"""
Classifier-Free Guidance (CFG) for Personality-Conditioned Generation

Enables stronger personality conditioning during sampling by interpolating
between conditional and unconditional predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassifierFreeGuidance:
    """
    Classifier-Free Guidance wrapper for personality-conditioned models.

    During training, randomly drops personality conditioning to learn
    both conditional p(x|c) and unconditional p(x) distributions.

    During sampling, interpolates predictions:
    x_guided = (1 + w) * x_cond - w * x_uncond

    Args:
        model: The conditioned diffusion model
        dropout_prob: Probability of dropping conditioning during training (default: 0.1)
        guidance_scale: Guidance strength during sampling (default: 2.0)
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_prob: float = 0.1,
        guidance_scale: float = 2.0,
    ):
        self.model = model
        self.dropout_prob = dropout_prob
        self.guidance_scale = guidance_scale

    def apply_conditioning_dropout(
        self,
        personality: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Randomly zero out personality conditioning during training.

        Args:
            personality: Personality embedding [batch, conditioning_dim]
            training: Whether in training mode

        Returns:
            Personality embedding with random dropout [batch, conditioning_dim]
        """
        if not training:
            return personality

        batch_size = personality.size(0)
        device = personality.device

        # Sample dropout mask
        dropout_mask = torch.rand(batch_size, device=device) > self.dropout_prob
        dropout_mask = dropout_mask.float().unsqueeze(-1)  # [batch, 1]

        # Apply mask (multiply by 0 to drop, 1 to keep)
        return personality * dropout_mask

    def get_null_conditioning(
        self,
        batch_size: int,
        conditioning_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create null/unconditional personality embedding.

        Args:
            batch_size: Batch size
            conditioning_dim: Dimension of conditioning vector
            device: Device to create tensor on

        Returns:
            Null conditioning [batch, conditioning_dim] (all zeros)
        """
        return torch.zeros(batch_size, conditioning_dim, device=device)

    def guided_prediction(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        personality_cond: torch.Tensor,
        guidance_scale: Optional[float] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Make guided prediction during sampling.

        prediction = (1 + w) * pred_cond - w * pred_uncond

        Args:
            x: Noised input [batch, ...]
            timestep: Diffusion timestep [batch] or scalar
            personality_cond: Personality conditioning [batch, conditioning_dim]
            guidance_scale: Override default guidance scale
            **model_kwargs: Additional arguments to model

        Returns:
            Guided prediction [batch, ...]
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        # Conditional prediction
        pred_cond = self.model(
            x,
            timestep,
            personality=personality_cond,
            **model_kwargs
        )

        # Unconditional prediction (null conditioning)
        batch_size = x.size(0)
        conditioning_dim = personality_cond.size(-1)
        personality_null = self.get_null_conditioning(
            batch_size,
            conditioning_dim,
            x.device,
        )

        pred_uncond = self.model(
            x,
            timestep,
            personality=personality_null,
            **model_kwargs
        )

        # Guidance interpolation
        pred_guided = (1 + guidance_scale) * pred_cond - guidance_scale * pred_uncond

        return pred_guided


class CFGWrapper(nn.Module):
    """
    Wrapper module that handles CFG training and inference.

    Wraps any personality-conditioned model to add CFG capabilities.

    Args:
        model: Base model to wrap
        dropout_prob: CFG dropout probability during training
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.model = model
        self.dropout_prob = dropout_prob

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        personality: torch.Tensor,
        use_cfg_dropout: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with optional CFG dropout.

        Args:
            x: Input tensor
            timestep: Diffusion timestep
            personality: Personality conditioning
            use_cfg_dropout: Whether to apply CFG dropout (default: True in training)
            **kwargs: Additional model arguments

        Returns:
            Model output
        """
        # Apply conditioning dropout during training
        if use_cfg_dropout and self.training:
            personality = self.apply_conditioning_dropout(personality)

        return self.model(x, timestep, personality, **kwargs)

    def apply_conditioning_dropout(self, personality: torch.Tensor) -> torch.Tensor:
        """Apply CFG dropout to personality conditioning."""
        batch_size = personality.size(0)
        device = personality.device

        # Sample dropout mask
        dropout_mask = torch.rand(batch_size, device=device) > self.dropout_prob
        dropout_mask = dropout_mask.float().unsqueeze(-1)

        return personality * dropout_mask

    @torch.no_grad()
    def guided_sample(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        personality: torch.Tensor,
        guidance_scale: float = 2.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample with classifier-free guidance.

        Args:
            x: Noised input
            timestep: Diffusion timestep
            personality: Personality conditioning
            guidance_scale: Guidance strength
            **kwargs: Additional model arguments

        Returns:
            Guided prediction
        """
        # Conditional prediction
        pred_cond = self.model(x, timestep, personality, **kwargs)

        # Unconditional prediction
        batch_size = x.size(0)
        conditioning_dim = personality.size(-1)
        personality_null = torch.zeros(batch_size, conditioning_dim, device=x.device)

        pred_uncond = self.model(x, timestep, personality_null, **kwargs)

        # Guidance
        pred_guided = (1 + guidance_scale) * pred_cond - guidance_scale * pred_uncond

        return pred_guided


def create_cfg_model(
    base_model: nn.Module,
    dropout_prob: float = 0.1,
) -> CFGWrapper:
    """
    Convenience function to wrap a model with CFG.

    Args:
        base_model: Model to wrap
        dropout_prob: CFG dropout probability

    Returns:
        CFG-wrapped model
    """
    return CFGWrapper(base_model, dropout_prob)


if __name__ == '__main__':
    """Test CFG functionality."""
    print("Testing Classifier-Free Guidance...\n")

    # Create a dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self, hidden_dim=256, conditioning_dim=64):
            super().__init__()
            self.proj = nn.Linear(hidden_dim + conditioning_dim + 1, hidden_dim)

        def forward(self, x, timestep, personality):
            # Concatenate inputs
            t = timestep.view(-1, 1).expand(-1, x.size(1))  # Expand timestep
            combined = torch.cat([x, personality.unsqueeze(1).expand(-1, x.size(1), -1), t.unsqueeze(-1)], dim=-1)
            return self.proj(combined)

    # Create model and wrap with CFG
    base_model = DummyModel()
    cfg_model = CFGWrapper(base_model, dropout_prob=0.1)

    print(f"CFG Model created with dropout_prob=0.1")

    # Test forward pass with dropout
    batch_size = 4
    seq_len = 16
    hidden_dim = 256
    conditioning_dim = 64

    x = torch.randn(batch_size, seq_len, hidden_dim)
    timestep = torch.rand(batch_size)
    personality = torch.randn(batch_size, conditioning_dim)

    # Training mode (with dropout)
    cfg_model.train()
    output_train = cfg_model(x, timestep, personality, use_cfg_dropout=True)
    print(f"\nTraining forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output_train.shape}")

    # Test guided sampling
    cfg_model.eval()
    output_guided = cfg_model.guided_sample(x, timestep, personality, guidance_scale=2.0)
    print(f"\nGuided sampling:")
    print(f"  Output shape: {output_guided.shape}")
    print(f"  Guidance scale: 2.0")

    # Test with different guidance scales
    print(f"\nTesting different guidance scales:")
    for scale in [1.0, 2.0, 5.0]:
        output = cfg_model.guided_sample(x, timestep, personality, guidance_scale=scale)
        print(f"  Scale {scale}: output norm = {output.norm().item():.4f}")

    print("\n✓ CFG test passed!")
