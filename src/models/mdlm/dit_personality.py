"""
Extended DiT with Personality Conditioning

Extends the original MDLM DiT backbone to accept personality conditioning
from CausalVAE using AdaLN and Cross-Attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dit import DIT, DDiTBlock
from ..conditioning.adaln import AdaLNZero
from ..conditioning.cross_attention import PersonalityCrossAttention


class PersonalityDiTBlock(nn.Module):
    """
    DiT block extended with personality conditioning.

    Adds:
    - AdaLN-Zero for global personality influence
    - Cross-attention for token-level conditioning

    Args:
        Original DiT block args plus:
        conditioning_dim: Dimension of personality conditioning
        num_personality_tokens: Number of pseudo-tokens for cross-attention
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        conditioning_dim=64,
        num_personality_tokens=8,
        **kwargs
    ):
        super().__init__()

        # Original DiT components
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=0.0,
            batch_first=True
        )

        # AdaLN-Zero for modulation (returns y, gate)
        self.adaln = AdaLNZero(hidden_size, conditioning_dim)

        # Cross-attention with personality
        self.cross_attn = PersonalityCrossAttention(
            hidden_dim=hidden_size,
            conditioning_dim=conditioning_dim,
            num_heads=num_heads,
            num_personality_tokens=num_personality_tokens,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

    def forward(self, x, personality_cond, attention_mask=None):
        """
        Forward pass with personality conditioning.

        Args:
            x: Input tokens [batch, seq_len, hidden]
            personality_cond: Personality conditioning [batch, conditioning_dim]
            attention_mask: Optional attention mask

        Returns:
            Output tokens [batch, seq_len, hidden]
        """
        # Self-attention with AdaLN-Zero modulation
        normed = self.norm1(x)
        normed_modulated, gate = self.adaln(normed, personality_cond)

        attn_out, _ = self.attn(
            normed_modulated,
            normed_modulated,
            normed_modulated,
            attn_mask=attention_mask,
            need_weights=False
        )
        # Apply gate to residual
        x = x + gate.unsqueeze(1) * attn_out

        # Cross-attention with personality
        cross_attn_out = self.cross_attn(x, personality_cond, attention_mask=attention_mask)
        x = x + cross_attn_out

        # MLP
        normed = self.norm2(x)
        mlp_out = self.mlp(normed)
        x = x + mlp_out

        return x


class PersonalityDiT(DIT):
    """
    DiT extended with personality conditioning.

    Replaces standard DiT blocks with PersonalityDiTBlocks that accept
    personality conditioning from CausalVAE.

    Args:
        config: MDLM config (must have config.model.hidden_size, n_heads, etc.)
        conditioning_dim: Dimension of personality conditioning (default: 64)
        num_personality_tokens: Number of pseudo-tokens for cross-attention
        vocab_size: Vocabulary size
    """

    def __init__(
        self,
        config,
        conditioning_dim=64,
        num_personality_tokens=8,
        vocab_size=None,
    ):
        # Initialize parent DIT (sets self.vocab_embed, self.sigma_map,
        # self.rotary_emb, self.blocks, self.output_layer, self.config)
        super().__init__(config, vocab_size=vocab_size)

        self.conditioning_dim = conditioning_dim

        # Access hidden_size and n_heads from config (DIT stores config, not attrs)
        hidden_size = config.model.hidden_size
        n_heads = config.model.n_heads

        # Project personality to hidden dimension
        self.personality_proj = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Add cross-attention modules for each layer
        self.personality_cross_attns = nn.ModuleList([
            PersonalityCrossAttention(
                hidden_dim=hidden_size,
                conditioning_dim=conditioning_dim,
                num_heads=n_heads,
                num_personality_tokens=num_personality_tokens,
            )
            for _ in range(len(self.blocks))
        ])

    def forward(
        self,
        indices,
        sigma,
        personality_cond=None,
        mask=None
    ):
        """
        Forward pass with optional personality conditioning.

        Args:
            indices: Token indices [batch, seq_len]
            sigma: Diffusion timesteps [batch] (noise level)
            personality_cond: Personality conditioning [batch, conditioning_dim]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # If no personality conditioning, use zeros (classifier-free guidance)
        if personality_cond is None:
            personality_cond = torch.zeros(
                indices.size(0),
                self.conditioning_dim,
                device=indices.device
            )

        # Token embeddings (DIT uses self.vocab_embed, not self.tok_embeddings)
        x = self.vocab_embed(indices)

        # Time conditioning (same as DIT.forward)
        c = F.silu(self.sigma_map(sigma))

        # Rotary embeddings
        rotary_cos_sin = self.rotary_emb(x)

        # Apply transformer blocks with personality conditioning injection
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i, block in enumerate(self.blocks):
                # Original DDiTBlock forward: (x, rotary_cos_sin, c, seqlens)
                x = block(x, rotary_cos_sin, c, seqlens=None)

                # Add personality cross-attention after each block
                if i < len(self.personality_cross_attns):
                    cross_attn_out = self.personality_cross_attns[i](
                        x,
                        personality_cond,
                        attention_mask=mask
                    )
                    x = x + cross_attn_out

            # Output projection: DDitFinalLayer expects (x, c)
            x = self.output_layer(x, c)

        return x


# Helper function to create personality-conditioned DiT
def create_personality_dit(config, conditioning_dim=64, vocab_size=None):
    """
    Factory function to create PersonalityDiT.

    Args:
        config: MDLM config
        conditioning_dim: Personality conditioning dimension
        vocab_size: Vocabulary size

    Returns:
        PersonalityDiT model
    """
    return PersonalityDiT(
        config,
        conditioning_dim=conditioning_dim,
        vocab_size=vocab_size
    )
