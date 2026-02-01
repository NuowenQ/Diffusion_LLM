"""
Noise Schedules for MDLM

Defines the masking probability schedule over diffusion timesteps.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import math


class NoiseSchedule(nn.Module):
    """
    Base class for noise schedules in discrete diffusion.

    The schedule defines alpha_t (probability of keeping original token)
    at each timestep t.

    Args:
        num_timesteps: Total number of diffusion steps
    """

    def __init__(self, num_timesteps: int = 1000):
        super().__init__()
        self.num_timesteps = num_timesteps

    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get alpha_t for given timestep(s).

        Args:
            t: Timestep(s) [batch] or scalar

        Returns:
            alpha_t: Probability of keeping original token
        """
        raise NotImplementedError

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get cumulative product alpha_bar_t = prod(alpha_s for s <= t).

        Args:
            t: Timestep(s) [batch] or scalar

        Returns:
            alpha_bar_t: Cumulative masking probability
        """
        raise NotImplementedError


class CosineSchedule(NoiseSchedule):
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).

    Better for longer sequences and smoother transitions.

    alpha_bar_t = cos((t/T + s) / (1 + s) * pi/2)^2

    Args:
        num_timesteps: Total diffusion steps
        s: Small offset to prevent alpha_bar_T = 0 (default: 0.008)
    """

    def __init__(self, num_timesteps: int = 1000, s: float = 0.008):
        super().__init__(num_timesteps)
        self.s = s

        # Precompute alpha_bar for all timesteps
        timesteps = torch.arange(num_timesteps + 1, dtype=torch.float32)
        alpha_bar = self._cosine_alpha_bar(timesteps)

        # Compute alpha_t = alpha_bar_t / alpha_bar_{t-1}
        alpha = torch.zeros(num_timesteps + 1)
        alpha[0] = alpha_bar[0]
        alpha[1:] = alpha_bar[1:] / alpha_bar[:-1]

        # Register as buffers (not trainable)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

    def _cosine_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Compute cosine schedule."""
        f_t = torch.cos(((t / self.num_timesteps + self.s) / (1 + self.s)) * math.pi * 0.5) ** 2
        f_0 = torch.cos((self.s / (1 + self.s)) * math.pi * 0.5) ** 2
        return f_t / f_0

    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_t for timestep t."""
        if t.dim() == 0:
            return self.alpha[t]
        return self.alpha[t.long()]

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_bar_t for timestep t."""
        if t.dim() == 0:
            return self.alpha_bar[t]
        return self.alpha_bar[t.long()]


class LinearSchedule(NoiseSchedule):
    """
    Linear noise schedule.

    alpha_bar_t = 1 - t/T

    Simpler but can be unstable for long sequences.

    Args:
        num_timesteps: Total diffusion steps
        beta_start: Starting beta (default: 0.0001)
        beta_end: Ending beta (default: 0.02)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__(num_timesteps)

        # Linear schedule for beta
        betas = torch.linspace(beta_start, beta_end, num_timesteps + 1)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('alpha', alphas)
        self.register_buffer('alpha_bar', alpha_bar)

    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_t for timestep t."""
        if t.dim() == 0:
            return self.alpha[t]
        return self.alpha[t.long()]

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_bar_t for timestep t."""
        if t.dim() == 0:
            return self.alpha_bar[t]
        return self.alpha_bar[t.long()]


class LearnedSchedule(NoiseSchedule):
    """
    Learnable noise schedule.

    Parameterizes alpha_bar_t with a neural network.
    Can adapt to the specific data distribution.

    Args:
        num_timesteps: Total diffusion steps
        hidden_dim: Hidden dimension for MLP (default: 128)
    """

    def __init__(self, num_timesteps: int = 1000, hidden_dim: int = 128):
        super().__init__(num_timesteps)

        # MLP to predict alpha_bar from timestep
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Ensure alpha_bar in [0, 1]
        )

        # Initialize to approximate cosine schedule
        self._init_to_cosine()

    def _init_to_cosine(self):
        """Initialize to mimic cosine schedule."""
        cosine_schedule = CosineSchedule(self.num_timesteps)
        timesteps = torch.linspace(0, self.num_timesteps, 100).unsqueeze(-1)
        targets = cosine_schedule.get_alpha_bar(timesteps.squeeze(-1))

        # Simple supervised pretraining
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=1e-3)
        for _ in range(1000):
            pred = self.mlp(timesteps / self.num_timesteps).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get learned alpha_bar_t for timestep t."""
        t_normalized = t.float() / self.num_timesteps
        if t.dim() == 0:
            t_normalized = t_normalized.unsqueeze(0).unsqueeze(0)
        else:
            t_normalized = t_normalized.unsqueeze(-1)
        return self.mlp(t_normalized).squeeze(-1)

    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha_t from alpha_bar_t and alpha_bar_{t-1}."""
        alpha_bar_t = self.get_alpha_bar(t)

        # Handle t = 0 case
        if t.dim() == 0:
            if t == 0:
                return alpha_bar_t
            alpha_bar_prev = self.get_alpha_bar(t - 1)
        else:
            t_prev = torch.clamp(t - 1, min=0)
            alpha_bar_prev = self.get_alpha_bar(t_prev)
            # For t = 0, use alpha_bar_0
            mask = (t == 0).float()
            alpha_bar_prev = alpha_bar_prev * (1 - mask) + alpha_bar_t * mask

        return alpha_bar_t / (alpha_bar_prev + 1e-8)


def create_noise_schedule(
    schedule_type: str = 'cosine',
    num_timesteps: int = 1000,
    **kwargs
) -> NoiseSchedule:
    """
    Factory function to create noise schedules.

    Args:
        schedule_type: Type of schedule ('cosine', 'linear', 'learned')
        num_timesteps: Total diffusion steps
        **kwargs: Additional arguments for specific schedules

    Returns:
        NoiseSchedule instance
    """
    if schedule_type == 'cosine':
        return CosineSchedule(num_timesteps, **kwargs)
    elif schedule_type == 'linear':
        return LinearSchedule(num_timesteps, **kwargs)
    elif schedule_type == 'learned':
        return LearnedSchedule(num_timesteps, **kwargs)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


if __name__ == '__main__':
    """Test noise schedules."""
    print("Testing Noise Schedules...\n")

    num_timesteps = 1000

    # Test Cosine Schedule
    print("Cosine Schedule:")
    cosine = CosineSchedule(num_timesteps)
    t = torch.tensor([0, 100, 500, 999])
    alpha_bar = cosine.get_alpha_bar(t)
    print(f"  alpha_bar at t={t.tolist()}: {alpha_bar.tolist()}")
    print(f"  alpha_bar_0 (should be ~1.0): {cosine.get_alpha_bar(torch.tensor(0)):.4f}")
    print(f"  alpha_bar_T (should be ~0.0): {cosine.get_alpha_bar(torch.tensor(999)):.4f}")

    # Test Linear Schedule
    print("\nLinear Schedule:")
    linear = LinearSchedule(num_timesteps)
    alpha_bar = linear.get_alpha_bar(t)
    print(f"  alpha_bar at t={t.tolist()}: {alpha_bar.tolist()}")

    # Test Learned Schedule
    print("\nLearned Schedule (initialized to cosine):")
    learned = LearnedSchedule(num_timesteps)
    alpha_bar_learned = learned.get_alpha_bar(t)
    alpha_bar_cosine = cosine.get_alpha_bar(t)
    print(f"  Learned alpha_bar: {alpha_bar_learned.tolist()}")
    print(f"  Cosine alpha_bar: {alpha_bar_cosine.tolist()}")
    print(f"  MSE: {torch.nn.functional.mse_loss(alpha_bar_learned, alpha_bar_cosine):.6f}")

    print("\n✓ Noise schedule test passed!")
