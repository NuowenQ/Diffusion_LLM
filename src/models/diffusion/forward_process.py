"""
Forward Diffusion Process for MDLM

Implements the masking process q(x_t | x_0) for discrete text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .noise_schedule import NoiseSchedule


class ForwardDiffusion(nn.Module):
    """
    Forward diffusion process for discrete text.

    At each timestep t, tokens are masked with probability (1 - alpha_bar_t).

    q(x_t | x_0) = alpha_bar_t * x_0 + (1 - alpha_bar_t) * [MASK]

    Args:
        noise_schedule: Noise schedule defining alpha_t
        mask_token_id: ID of the mask token (default: from tokenizer)
        vocab_size: Size of vocabulary
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        mask_token_id: int,
        vocab_size: int,
    ):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t | x_0) by masking tokens.

        Args:
            x_0: Original tokens [batch, seq_len]
            t: Timesteps [batch]
            noise: Optional random mask (if None, sample uniformly)

        Returns:
            x_t: Masked tokens [batch, seq_len]
            mask: Boolean mask indicating which tokens were masked [batch, seq_len]
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device

        # Get masking probability
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t)  # [batch]

        # Sample which tokens to mask
        if noise is None:
            noise = torch.rand(batch_size, seq_len, device=device)

        # Mask tokens where noise > alpha_bar_t
        mask = noise > alpha_bar_t.unsqueeze(-1)  # [batch, seq_len]

        # Apply masking
        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id

        return x_t, mask

    def q_posterior(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).

        For discrete diffusion:
        q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)

        Args:
            x_t: Current masked tokens [batch, seq_len]
            x_0: Original tokens [batch, seq_len]
            t: Current timestep [batch]

        Returns:
            Dictionary with posterior probabilities
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device

        # Get alpha values
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t)
        alpha_bar_t_minus_1 = self.noise_schedule.get_alpha_bar(torch.clamp(t - 1, min=0))
        alpha_t = alpha_bar_t / (alpha_bar_t_minus_1 + 1e-8)

        # For masked tokens, posterior depends on x_0
        # For unmasked tokens, they stay the same
        is_masked = (x_t == self.mask_token_id)

        # Probability of unmasking: (alpha_bar_{t-1} - alpha_bar_t * alpha_t) / (1 - alpha_bar_t)
        unmask_prob = (alpha_bar_t_minus_1 - alpha_bar_t * alpha_t) / (1 - alpha_bar_t + 1e-8)
        unmask_prob = unmask_prob.unsqueeze(-1).expand(-1, seq_len)

        return {
            'unmask_prob': unmask_prob,
            'is_masked': is_masked,
        }

    def compute_loss(
        self,
        model_logits: torch.Tensor,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cross-entropy loss for masked token prediction.

        L = E_t[∑_{i: x_t[i] = [MASK]} -log p_θ(x_0[i] | x_t, t)]

        Args:
            model_logits: Model predictions [batch, seq_len, vocab_size]
            x_0: Original tokens [batch, seq_len]
            x_t: Masked tokens [batch, seq_len]
            t: Timesteps [batch]
            loss_mask: Optional mask for loss computation (e.g., padding mask)

        Returns:
            Dictionary with loss and metrics
        """
        batch_size, seq_len = x_0.shape

        # Only compute loss on masked positions
        is_masked = (x_t == self.mask_token_id)

        # Apply additional loss mask if provided (e.g., for padding)
        if loss_mask is not None:
            is_masked = is_masked & loss_mask

        # Cross-entropy loss
        loss = F.cross_entropy(
            model_logits.view(-1, self.vocab_size),
            x_0.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)

        # Mask out non-masked positions
        loss = loss * is_masked.float()

        # Average over masked positions
        num_masked = is_masked.float().sum() + 1e-8
        loss = loss.sum() / num_masked

        # Compute accuracy on masked positions
        pred_tokens = model_logits.argmax(dim=-1)
        correct = (pred_tokens == x_0) & is_masked
        accuracy = correct.float().sum() / num_masked

        return {
            'loss': loss,
            'accuracy': accuracy,
            'num_masked': num_masked,
        }


class UniformMaskingStrategy(nn.Module):
    """
    Uniform random masking strategy.

    Masks each token independently with probability (1 - alpha_bar_t).
    """

    def __init__(self, mask_token_id: int):
        super().__init__()
        self.mask_token_id = mask_token_id

    def forward(
        self,
        x_0: torch.Tensor,
        mask_prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply uniform masking.

        Args:
            x_0: Original tokens [batch, seq_len]
            mask_prob: Masking probability [batch] or [batch, seq_len]

        Returns:
            x_masked: Masked tokens [batch, seq_len]
            mask: Boolean mask [batch, seq_len]
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device

        # Sample random values
        noise = torch.rand(batch_size, seq_len, device=device)

        # Expand mask_prob if needed
        if mask_prob.dim() == 1:
            mask_prob = mask_prob.unsqueeze(-1)

        # Create mask
        mask = noise < mask_prob

        # Apply masking
        x_masked = x_0.clone()
        x_masked[mask] = self.mask_token_id

        return x_masked, mask


class SpanMaskingStrategy(nn.Module):
    """
    Span masking strategy (like BERT/SpanBERT).

    Masks contiguous spans of tokens rather than independent tokens.
    Can improve local coherence.

    Args:
        mask_token_id: ID of mask token
        mean_span_length: Mean length of masked spans (default: 3)
    """

    def __init__(self, mask_token_id: int, mean_span_length: float = 3.0):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.mean_span_length = mean_span_length

    def forward(
        self,
        x_0: torch.Tensor,
        mask_prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply span masking.

        Args:
            x_0: Original tokens [batch, seq_len]
            mask_prob: Target masking probability [batch]

        Returns:
            x_masked: Masked tokens [batch, seq_len]
            mask: Boolean mask [batch, seq_len]
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device

        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        for i in range(batch_size):
            current_masked = 0
            target_masked = int(mask_prob[i].item() * seq_len)

            while current_masked < target_masked:
                # Sample span start
                start = torch.randint(0, seq_len, (1,)).item()

                # Sample span length from geometric distribution
                span_length = torch.distributions.Geometric(1.0 / self.mean_span_length).sample().int().item() + 1
                span_length = min(span_length, seq_len - start)

                # Mask span
                mask[i, start:start + span_length] = True
                current_masked = mask[i].sum().item()

        # Apply masking
        x_masked = x_0.clone()
        x_masked[mask] = self.mask_token_id

        return x_masked, mask


if __name__ == '__main__':
    """Test forward diffusion process."""
    print("Testing Forward Diffusion Process...\n")

    from .noise_schedule import CosineSchedule

    # Setup
    batch_size = 4
    seq_len = 16
    vocab_size = 1000
    mask_token_id = 999
    num_timesteps = 100

    # Create schedule and forward process
    schedule = CosineSchedule(num_timesteps)
    forward_diffusion = ForwardDiffusion(schedule, mask_token_id, vocab_size)

    # Test q_sample
    x_0 = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
    t = torch.randint(0, num_timesteps, (batch_size,))

    x_t, mask = forward_diffusion.q_sample(x_0, t)
    print(f"q_sample test:")
    print(f"  x_0 shape: {x_0.shape}")
    print(f"  x_t shape: {x_t.shape}")
    print(f"  mask shape: {mask.shape}")
    print(f"  Fraction masked: {mask.float().mean():.3f}")

    # Test at different timesteps
    print(f"\nMasking at different timesteps:")
    for t_val in [10, 50, 90]:
        t_batch = torch.full((batch_size,), t_val)
        x_t, mask = forward_diffusion.q_sample(x_0, t_batch)
        print(f"  t={t_val}: {mask.float().mean():.3f} masked")

    # Test loss computation
    model_logits = torch.randn(batch_size, seq_len, vocab_size)
    loss_dict = forward_diffusion.compute_loss(model_logits, x_0, x_t, t)
    print(f"\nLoss computation:")
    print(f"  Loss: {loss_dict['loss']:.4f}")
    print(f"  Accuracy: {loss_dict['accuracy']:.4f}")
    print(f"  Num masked: {loss_dict['num_masked']:.1f}")

    # Test uniform masking strategy
    print(f"\nUniform masking strategy:")
    uniform = UniformMaskingStrategy(mask_token_id)
    x_masked, mask = uniform(x_0, torch.tensor([0.3] * batch_size))
    print(f"  Fraction masked: {mask.float().mean():.3f}")

    # Test span masking strategy
    print(f"\nSpan masking strategy:")
    span = SpanMaskingStrategy(mask_token_id, mean_span_length=3.0)
    x_masked, mask = span(x_0, torch.tensor([0.3] * batch_size))
    print(f"  Fraction masked: {mask.float().mean():.3f}")

    print("\n✓ Forward diffusion test passed!")
