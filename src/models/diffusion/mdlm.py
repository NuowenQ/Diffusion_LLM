"""
MDLM (Masked Diffusion Language Model) with Personality Conditioning

Main model architecture combining:
- Transformer backbone
- Personality conditioning (AdaLN + Cross-Attention)
- Discrete diffusion for text generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from dataclasses import dataclass
import math

from ..conditioning.adaln import AdaLNTransformerBlock
from ..conditioning.cross_attention import HybridConditioningBlock
from .noise_schedule import create_noise_schedule, NoiseSchedule
from .forward_process import ForwardDiffusion
from .reverse_process import ReverseDiffusion


@dataclass
class MDLMConfig:
    """
    Configuration for MDLM model.
    
    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Hidden dimension of transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        conditioning_dim: Dimension of personality conditioning
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        conditioning_type: Type of conditioning ('adaln', 'cross_attn', 'hybrid')
        use_adaln_zero: Whether to use AdaLN-Zero gating
        num_personality_tokens: Number of pseudo-tokens for cross-attention
        num_timesteps: Number of diffusion timesteps
        noise_schedule: Type of noise schedule ('cosine', 'linear', 'learned')
    """
    vocab_size: int = 50257  # GPT-2 vocab size
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    conditioning_dim: int = 64
    max_seq_len: int = 512
    dropout: float = 0.1
    conditioning_type: str = 'hybrid'  # 'adaln', 'cross_attn', or 'hybrid'
    use_adaln_zero: bool = True
    num_personality_tokens: int = 8
    num_timesteps: int = 1000
    noise_schedule: str = 'cosine'
    mlp_ratio: float = 4.0


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding.
    
    Args:
        embedding_dim: Dimension of timestep embedding
    """
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Project to hidden dimension
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: Timesteps [batch]
            
        Returns:
            Embeddings [batch, embedding_dim]
        """
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # Project
        emb = self.mlp(emb)
        
        return emb


class MDLM(nn.Module):
    """
    Masked Diffusion Language Model with Personality Conditioning.
    
    Combines discrete diffusion with personality-conditioned transformers
    for controllable text generation.
    
    Args:
        config: Model configuration
        mask_token_id: ID of mask token (usually from tokenizer)
    """
    
    def __init__(self, config: MDLMConfig, mask_token_id: int):
        super().__init__()
        self.config = config
        self.mask_token_id = mask_token_id
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        
        # Timestep embedding
        self.time_embedding = TimestepEmbedding(config.hidden_dim)
        
        # Combine timestep and personality conditioning
        self.conditioning_proj = nn.Sequential(
            nn.Linear(config.hidden_dim + config.conditioning_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.conditioning_dim),
        )
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            if config.conditioning_type == 'adaln':
                layer = AdaLNTransformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    conditioning_dim=config.conditioning_dim,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    use_adaln_zero=config.use_adaln_zero,
                )
            elif config.conditioning_type == 'cross_attn' or config.conditioning_type == 'hybrid':
                layer = HybridConditioningBlock(
                    hidden_dim=config.hidden_dim,
                    conditioning_dim=config.conditioning_dim,
                    num_heads=config.num_heads,
                    num_personality_tokens=config.num_personality_tokens,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    use_adaln=config.use_adaln_zero and config.conditioning_type == 'hybrid',
                )
            else:
                raise ValueError(f"Unknown conditioning type: {config.conditioning_type}")
            
            self.layers.append(layer)
        
        # Output layer
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights with token embedding
        self.head.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Noise schedule and diffusion processes
        self.noise_schedule = create_noise_schedule(
            config.noise_schedule,
            config.num_timesteps
        )
        self.forward_diffusion = ForwardDiffusion(
            self.noise_schedule,
            mask_token_id,
            config.vocab_size,
        )
        self.reverse_diffusion = ReverseDiffusion(
            self.noise_schedule,
            mask_token_id,
            config.vocab_size,
        )
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        personality: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict logits for masked tokens.
        
        Args:
            input_ids: Token IDs (possibly masked) [batch, seq_len]
            timesteps: Diffusion timesteps [batch]
            personality: Personality conditioning [batch, conditioning_dim]
            attention_mask: Optional attention mask [batch, seq_len]
            
        Returns:
            logits: Token logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, hidden_dim]
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        
        # Timestep embeddings
        t_emb = self.time_embedding(timesteps)  # [batch, hidden_dim]
        
        # Combine timestep and personality conditioning
        combined_cond = torch.cat([t_emb, personality], dim=-1)
        conditioning = self.conditioning_proj(combined_cond)  # [batch, conditioning_dim]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Prepare attention mask for transformer
        if attention_mask is not None:
            # Convert to attention mask format (1 = attend, 0 = don't attend)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, conditioning, self_attn_mask=attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.head(x)
        
        return logits
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        personality: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.
        
        Args:
            input_ids: Original token IDs [batch, seq_len]
            personality: Personality conditioning [batch, conditioning_dim]
            attention_mask: Optional padding mask [batch, seq_len]
            
        Returns:
            Dictionary with loss and metrics
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Sample random timesteps
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=device)
        
        # Forward diffusion: mask tokens
        x_t, mask = self.forward_diffusion.q_sample(input_ids, t)
        
        # Model prediction
        logits = self(x_t, t, personality, attention_mask)
        
        # Compute loss
        loss_dict = self.forward_diffusion.compute_loss(
            logits,
            input_ids,
            x_t,
            t,
            loss_mask=attention_mask,
        )
        
        return loss_dict
    
    @torch.no_grad()
    def generate(
        self,
        personality: torch.Tensor,
        seq_len: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        guidance_scale: float = 0.0,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text conditioned on personality.
        
        Args:
            personality: Personality conditioning [batch, conditioning_dim]
            seq_len: Length of sequence to generate
            temperature: Sampling temperature
            top_k: Optional top-k sampling
            top_p: Optional nucleus sampling
            guidance_scale: Classifier-free guidance scale (0 = no guidance)
            num_steps: Number of diffusion steps (if None, use full schedule)
            
        Returns:
            Generated token IDs [batch, seq_len]
        """
        batch_size = personality.size(0)
        device = personality.device
        
        # Start from fully masked sequence
        x_t = torch.full((batch_size, seq_len), self.mask_token_id, device=device)
        
        # Determine timesteps to use
        if num_steps is None:
            num_steps = self.config.num_timesteps
        
        timesteps = torch.linspace(
            self.config.num_timesteps - 1,
            0,
            num_steps,
            dtype=torch.long,
            device=device
        )
        
        # Iterative denoising
        for t_idx in timesteps:
            t = torch.full((batch_size,), t_idx, device=device)
            
            if guidance_scale > 0:
                # Classifier-free guidance
                # Conditional prediction
                logits_cond = self(x_t, t, personality)
                
                # Unconditional prediction
                personality_null = torch.zeros_like(personality)
                logits_uncond = self(x_t, t, personality_null)
                
                # Guided logits
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
                
                # Sample
                x_t = self._sample_from_logits(
                    x_t, logits, temperature, top_k, top_p
                )
            else:
                # Standard sampling
                x_t = self.reverse_diffusion.p_sample(
                    self, x_t, t, personality, temperature, top_k, top_p
                )
        
        return x_t
    
    def _sample_from_logits(
        self,
        x_t: torch.Tensor,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample tokens from logits."""
        # Apply temperature
        logits = logits / temperature
        
        # Apply filtering
        if top_k is not None:
            logits = self.reverse_diffusion._top_k_filtering(logits, top_k)
        if top_p is not None:
            logits = self.reverse_diffusion._top_p_filtering(logits, top_p)
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probs.view(-1, self.config.vocab_size),
            num_samples=1
        ).view(x_t.shape)
        
        # Only update masked positions
        is_masked = (x_t == self.mask_token_id)
        x_t_next = x_t.clone()
        x_t_next[is_masked] = sampled[is_masked]
        
        return x_t_next


if __name__ == '__main__':
    """Test MDLM model."""
    print("Testing MDLM Model...\n")
    
    # Configuration
    config = MDLMConfig(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        conditioning_dim=64,
        max_seq_len=128,
        num_timesteps=100,
    )
    
    mask_token_id = 999
    
    # Create model
    model = MDLM(config, mask_token_id)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(0, config.vocab_size - 1, (batch_size, seq_len))
    timesteps = torch.randint(0, config.num_timesteps, (batch_size,))
    personality = torch.randn(batch_size, config.conditioning_dim)
    
    logits = model(input_ids, timesteps, personality)
    print(f"\nForward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    
    # Test loss computation
    loss_dict = model.compute_loss(input_ids, personality)
    print(f"\nLoss computation:")
    print(f"  Loss: {loss_dict['loss']:.4f}")
    print(f"  Accuracy: {loss_dict['accuracy']:.4f}")
    
    # Test generation
    print(f"\nGeneration test:")
    generated = model.generate(personality, seq_len=64, num_steps=10)
    print(f"  Generated shape: {generated.shape}")
    print(f"  Num masked: {(generated == mask_token_id).float().sum():.0f}")
    
    print("\n✓ MDLM test passed!")
