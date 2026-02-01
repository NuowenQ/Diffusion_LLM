"""
Reverse Diffusion Process for MDLM

Implements the denoising process p_θ(x_{t-1} | x_t) for discrete text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .noise_schedule import NoiseSchedule


class ReverseDiffusion(nn.Module):
    """
    Reverse diffusion process for discrete text.

    Learns to predict the original tokens from masked versions:
    p_θ(x_{t-1} | x_t) = categorical(logits_θ(x_t, t))

    Args:
        noise_schedule: Noise schedule defining alpha_t
        mask_token_id: ID of the mask token
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

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        personality: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p_θ(x_{t-1} | x_t).

        Args:
            model: Denoising model
            x_t: Current masked tokens [batch, seq_len]
            t: Current timestep [batch]
            personality: Optional personality conditioning [batch, conditioning_dim]
            temperature: Sampling temperature (default: 1.0)
            top_k: Optional top-k sampling
            top_p: Optional nucleus sampling

        Returns:
            x_{t-1}: Partially denoised tokens [batch, seq_len]
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device

        # Get model predictions
        with torch.no_grad():
            logits = model(x_t, t, personality=personality)  # [batch, seq_len, vocab_size]

        # Apply temperature
        logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None:
            logits = self._top_k_filtering(logits, top_k)

        # Apply nucleus (top-p) filtering
        if top_p is not None:
            logits = self._top_p_filtering(logits, top_p)

        # Sample from categorical distribution
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(
            probs.view(-1, self.vocab_size),
            num_samples=1
        ).view(batch_size, seq_len)

        # Only update masked positions
        is_masked = (x_t == self.mask_token_id)
        x_t_minus_1 = x_t.clone()
        x_t_minus_1[is_masked] = sampled_tokens[is_masked]

        return x_t_minus_1

    def p_sample_greedy(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        personality: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedy decoding: select argmax at each position.

        Args:
            model: Denoising model
            x_t: Current masked tokens [batch, seq_len]
            t: Current timestep [batch]
            personality: Optional personality conditioning

        Returns:
            x_{t-1}: Partially denoised tokens [batch, seq_len]
        """
        with torch.no_grad():
            logits = model(x_t, t, personality=personality)

        # Greedy selection
        pred_tokens = logits.argmax(dim=-1)

        # Only update masked positions
        is_masked = (x_t == self.mask_token_id)
        x_t_minus_1 = x_t.clone()
        x_t_minus_1[is_masked] = pred_tokens[is_masked]

        return x_t_minus_1

    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """
        Filter logits to keep only top-k tokens.

        Args:
            logits: Token logits [batch, seq_len, vocab_size]
            top_k: Number of top tokens to keep

        Returns:
            Filtered logits
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Get top-k values and indices
        top_k = min(top_k, vocab_size)
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

        # Create mask for top-k
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, top_k_indices, top_k_logits)

        return mask

    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        """
        Nucleus sampling: keep top tokens with cumulative probability >= top_p.

        Args:
            logits: Token logits [batch, seq_len, vocab_size]
            top_p: Cumulative probability threshold

        Returns:
            Filtered logits
        """
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the mask to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Create mask
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, sorted_indices, sorted_logits)
        mask[sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)] = float('-inf')

        return mask

    def compute_vlb(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        personality: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Variational Lower Bound (VLB) for model evaluation.

        VLB = E_q[log p_θ(x_{t-1} | x_t) - log q(x_{t-1} | x_t, x_0)]

        Args:
            model: Denoising model
            x_0: Original tokens [batch, seq_len]
            x_t: Masked tokens [batch, seq_len]
            t: Timesteps [batch]
            personality: Optional personality conditioning

        Returns:
            VLB loss
        """
        batch_size, seq_len = x_0.shape

        # Get model predictions
        logits = model(x_t, t, personality=personality)
        log_probs = F.log_softmax(logits, dim=-1)

        # Get ground truth log probabilities
        log_p_theta = log_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)

        # For discrete diffusion, posterior is deterministic given x_0
        # q(x_{t-1} | x_t, x_0) is 1 at the correct unmasking position
        # So KL term is just whether the model predicts correctly

        # Only compute on masked positions
        is_masked = (x_t == self.mask_token_id)
        vlb = -log_p_theta * is_masked.float()

        return vlb.sum() / (is_masked.float().sum() + 1e-8)


class AnalyticUnmasking(nn.Module):
    """
    Analytic unmasking strategy.

    Instead of sampling, analytically compute which tokens to unmask
    based on the noise schedule.

    This can provide more stable generation in some cases.
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        mask_token_id: int,
    ):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id

    def forward(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        personality: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Analytically unmask tokens.

        Unmask the top (alpha_{t-1} - alpha_t) fraction of tokens
        based on model confidence.

        Args:
            model: Denoising model
            x_t: Current masked tokens [batch, seq_len]
            t: Current timestep [batch]
            personality: Optional personality conditioning

        Returns:
            x_{t-1}: Partially denoised tokens
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device

        # Get model predictions
        with torch.no_grad():
            logits = model(x_t, t, personality=personality)

        # Get confidence scores (max probability)
        probs = F.softmax(logits, dim=-1)
        confidence, pred_tokens = probs.max(dim=-1)

        # Determine how many tokens to unmask
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t)
        alpha_bar_t_minus_1 = self.noise_schedule.get_alpha_bar(torch.clamp(t - 1, min=0))

        # Fraction to unmask
        unmask_fraction = (alpha_bar_t - alpha_bar_t_minus_1) / (1 - alpha_bar_t + 1e-8)
        unmask_fraction = unmask_fraction.unsqueeze(-1)  # [batch, 1]

        # Count currently masked tokens
        is_masked = (x_t == self.mask_token_id)
        num_masked = is_masked.float().sum(dim=1, keepdim=True)  # [batch, 1]

        # Number to unmask
        num_to_unmask = (unmask_fraction * num_masked).long()

        # Unmask top-k most confident positions
        x_t_minus_1 = x_t.clone()
        for i in range(batch_size):
            if num_to_unmask[i] > 0 and is_masked[i].any():
                # Get confidence of masked positions
                masked_confidence = confidence[i].clone()
                masked_confidence[~is_masked[i]] = -1.0  # Exclude unmasked

                # Select top-k
                k = min(num_to_unmask[i].item(), is_masked[i].sum().item())
                _, top_indices = torch.topk(masked_confidence, k)

                # Unmask
                x_t_minus_1[i, top_indices] = pred_tokens[i, top_indices]

        return x_t_minus_1


if __name__ == '__main__':
    """Test reverse diffusion process."""
    print("Testing Reverse Diffusion Process...\n")

    from .noise_schedule import CosineSchedule

    # Setup
    batch_size = 4
    seq_len = 16
    vocab_size = 1000
    mask_token_id = 999
    num_timesteps = 100

    # Create schedule and reverse process
    schedule = CosineSchedule(num_timesteps)
    reverse_diffusion = ReverseDiffusion(schedule, mask_token_id, vocab_size)

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(vocab_size, vocab_size)

        def forward(self, x, t, personality=None):
            # Simple embedding + projection
            x_emb = F.one_hot(x, vocab_size).float()
            return self.proj(x_emb)

    model = DummyModel()

    # Test sampling
    x_t = torch.full((batch_size, seq_len), mask_token_id)
    x_t[:, :4] = torch.randint(0, vocab_size - 1, (batch_size, 4))  # Some unmasked
    t = torch.full((batch_size,), 50)

    x_t_minus_1 = reverse_diffusion.p_sample(model, x_t, t, temperature=1.0)
    print(f"Stochastic sampling:")
    print(f"  x_t shape: {x_t.shape}")
    print(f"  x_{{t-1}} shape: {x_t_minus_1.shape}")
    print(f"  Num masked in x_t: {(x_t == mask_token_id).float().sum():.0f}")
    print(f"  Num masked in x_{{t-1}}: {(x_t_minus_1 == mask_token_id).float().sum():.0f}")

    # Test greedy sampling
    x_t_minus_1_greedy = reverse_diffusion.p_sample_greedy(model, x_t, t)
    print(f"\nGreedy sampling:")
    print(f"  Num masked: {(x_t_minus_1_greedy == mask_token_id).float().sum():.0f}")

    # Test analytic unmasking
    analytic = AnalyticUnmasking(schedule, mask_token_id)
    x_t_minus_1_analytic = analytic(model, x_t, t)
    print(f"\nAnalytic unmasking:")
    print(f"  Num masked: {(x_t_minus_1_analytic == mask_token_id).float().sum():.0f}")

    print("\n✓ Reverse diffusion test passed!")
