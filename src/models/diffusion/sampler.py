"""
Sampling Strategies for MDLM

Implements various sampling methods including:
- DDPM (standard denoising)
- DDIM (deterministic, faster)
- Ancestral sampling
- Classifier-free guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from tqdm import tqdm


class DiffusionSampler:
    """
    Sampler for discrete diffusion models.
    
    Supports multiple sampling strategies with optional classifier-free guidance.
    
    Args:
        model: MDLM model
        sampler_type: Type of sampler ('ddpm', 'ddim', 'ancestral')
        num_steps: Number of sampling steps (can be less than training steps)
    """
    
    def __init__(
        self,
        model: nn.Module,
        sampler_type: str = 'ddpm',
        num_steps: Optional[int] = None,
    ):
        self.model = model
        self.sampler_type = sampler_type
        self.num_steps = num_steps or model.config.num_timesteps
        
        # Create sampling schedule
        self.timesteps = self._create_timestep_schedule()
    
    def _create_timestep_schedule(self) -> torch.Tensor:
        """Create timestep schedule for sampling."""
        if self.sampler_type == 'ddim':
            # Uniform spacing for DDIM
            timesteps = torch.linspace(
                self.model.config.num_timesteps - 1,
                0,
                self.num_steps,
                dtype=torch.long
            )
        else:
            # Standard schedule
            timesteps = torch.arange(
                self.model.config.num_timesteps - 1,
                -1,
                -self.model.config.num_timesteps // self.num_steps,
                dtype=torch.long
            )[:self.num_steps]
        
        return timesteps
    
    @torch.no_grad()
    def sample(
        self,
        personality: torch.Tensor,
        seq_len: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        guidance_scale: float = 0.0,
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using the configured sampling strategy.
        
        Args:
            personality: Personality conditioning [batch, conditioning_dim]
            seq_len: Sequence length to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling
            guidance_scale: CFG scale (0 = no guidance)
            progress_bar: Show progress bar
            
        Returns:
            Generated tokens [batch, seq_len]
        """
        batch_size = personality.size(0)
        device = personality.device
        
        # Start from fully masked
        x_t = torch.full(
            (batch_size, seq_len),
            self.model.mask_token_id,
            device=device
        )
        
        # Sampling loop
        iterator = tqdm(self.timesteps, desc="Sampling") if progress_bar else self.timesteps
        
        for t_idx in iterator:
            t = torch.full((batch_size,), t_idx, device=device)
            
            # Get logits (with optional CFG)
            if guidance_scale > 0:
                logits = self._guided_logits(x_t, t, personality, guidance_scale)
            else:
                logits = self.model(x_t, t, personality)
            
            # Sample next state
            if self.sampler_type == 'ddpm':
                x_t = self._ddpm_step(x_t, logits, t, temperature, top_k, top_p)
            elif self.sampler_type == 'ddim':
                x_t = self._ddim_step(x_t, logits, t)
            elif self.sampler_type == 'ancestral':
                x_t = self._ancestral_step(x_t, logits, t, temperature)
            else:
                raise ValueError(f"Unknown sampler type: {self.sampler_type}")
        
        return x_t
    
    def _guided_logits(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        personality: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Compute classifier-free guided logits."""
        # Conditional
        logits_cond = self.model(x_t, t, personality)
        
        # Unconditional
        personality_null = torch.zeros_like(personality)
        logits_uncond = self.model(x_t, t, personality_null)
        
        # Guidance
        logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        
        return logits
    
    def _ddpm_step(
        self,
        x_t: torch.Tensor,
        logits: torch.Tensor,
        t: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:
        """Standard DDPM sampling step."""
        # Apply temperature
        logits = logits / temperature
        
        # Apply filtering
        if top_k is not None:
            logits = self._top_k_filtering(logits, top_k)
        if top_p is not None:
            logits = self._top_p_filtering(logits, top_p)
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probs.view(-1, self.model.config.vocab_size),
            num_samples=1
        ).view(x_t.shape)
        
        # Only update masked positions
        is_masked = (x_t == self.model.mask_token_id)
        x_t_next = x_t.clone()
        x_t_next[is_masked] = sampled[is_masked]
        
        return x_t_next
    
    def _ddim_step(
        self,
        x_t: torch.Tensor,
        logits: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        DDIM sampling step (deterministic).
        
        For discrete diffusion, we use the greedy selection.
        """
        pred_tokens = logits.argmax(dim=-1)
        
        # Only update masked positions
        is_masked = (x_t == self.model.mask_token_id)
        x_t_next = x_t.clone()
        x_t_next[is_masked] = pred_tokens[is_masked]
        
        return x_t_next
    
    def _ancestral_step(
        self,
        x_t: torch.Tensor,
        logits: torch.Tensor,
        t: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Ancestral sampling: unmask a subset based on confidence.
        
        More stable than full DDPM for discrete diffusion.
        """
        batch_size, seq_len = x_t.shape
        
        # Get confidence scores
        probs = F.softmax(logits / temperature, dim=-1)
        confidence, pred_tokens = probs.max(dim=-1)
        
        # Determine fraction to unmask
        alpha_bar_t = self.model.noise_schedule.get_alpha_bar(t)
        t_prev = torch.clamp(t - 1, min=0)
        alpha_bar_prev = self.model.noise_schedule.get_alpha_bar(t_prev)
        
        unmask_fraction = (alpha_bar_t - alpha_bar_prev) / (1 - alpha_bar_t + 1e-8)
        
        # Unmask top-k most confident tokens
        is_masked = (x_t == self.model.mask_token_id)
        num_masked = is_masked.float().sum(dim=1)
        num_to_unmask = (unmask_fraction.unsqueeze(-1) * num_masked.unsqueeze(-1)).long()
        
        x_t_next = x_t.clone()
        
        for i in range(batch_size):
            if num_to_unmask[i] > 0 and is_masked[i].any():
                # Get masked positions
                masked_confidence = confidence[i].clone()
                masked_confidence[~is_masked[i]] = -1.0
                
                # Select top-k
                k = min(num_to_unmask[i].item(), is_masked[i].sum().item())
                _, top_indices = torch.topk(masked_confidence, k)
                
                # Unmask
                x_t_next[i, top_indices] = pred_tokens[i, top_indices]
        
        return x_t_next
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Top-k filtering."""
        vocab_size = logits.size(-1)
        top_k = min(top_k, vocab_size)
        
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, top_k_indices, top_k_logits)
        
        return mask
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus (top-p) filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, sorted_indices, sorted_logits)
        mask[sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)] = float('-inf')
        
        return mask


class CFGSampler(DiffusionSampler):
    """
    Sampler with built-in classifier-free guidance.
    
    Simplifies usage by always applying CFG.
    
    Args:
        model: MDLM model
        guidance_scale: Default guidance scale
        sampler_type: Base sampling strategy
        num_steps: Number of sampling steps
    """
    
    def __init__(
        self,
        model: nn.Module,
        guidance_scale: float = 2.0,
        sampler_type: str = 'ancestral',
        num_steps: Optional[int] = None,
    ):
        super().__init__(model, sampler_type, num_steps)
        self.guidance_scale = guidance_scale
    
    @torch.no_grad()
    def sample(
        self,
        personality: torch.Tensor,
        seq_len: int,
        temperature: float = 1.0,
        guidance_scale: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """Sample with CFG."""
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        return super().sample(
            personality,
            seq_len,
            temperature=temperature,
            guidance_scale=guidance_scale,
            **kwargs
        )


if __name__ == '__main__':
    """Test samplers."""
    print("Testing Diffusion Samplers...\n")
    
    from .mdlm import MDLM, MDLMConfig
    
    # Create model
    config = MDLMConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_timesteps=50,
    )
    
    model = MDLM(config, mask_token_id=999)
    model.eval()
    
    # Test different samplers
    batch_size = 2
    seq_len = 32
    personality = torch.randn(batch_size, config.conditioning_dim)
    
    for sampler_type in ['ddpm', 'ddim', 'ancestral']:
        print(f"Testing {sampler_type} sampler:")
        sampler = DiffusionSampler(model, sampler_type=sampler_type, num_steps=10)
        
        samples = sampler.sample(
            personality,
            seq_len,
            temperature=1.0,
            progress_bar=False,
        )
        
        print(f"  Generated shape: {samples.shape}")
        print(f"  Num masked: {(samples == 999).sum()}")
    
    # Test CFG sampler
    print(f"\nTesting CFG sampler:")
    cfg_sampler = CFGSampler(model, guidance_scale=2.0, num_steps=10)
    samples = cfg_sampler.sample(personality, seq_len, progress_bar=False)
    print(f"  Generated shape: {samples.shape}")
    
    print("\n✓ Sampler test passed!")
