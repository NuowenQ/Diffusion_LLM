"""
Adaptive Layer Normalization (AdaLN) for Personality Conditioning

Injects personality embeddings into transformer blocks via learned
scale and shift parameters.

Based on DiT (Diffusion Transformers) conditioning approach.
"""

import torch
import torch.nn as nn
from typing import Optional


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on personality embedding.

    Applies: y = γ(c) * LayerNorm(x) + β(c)

    where c is the conditioning vector (personality embedding), and
    γ, β are learned affine parameters predicted from c.

    Args:
        normalized_shape: Shape of the input to be normalized (e.g., hidden_dim)
        conditioning_dim: Dimension of conditioning vector (personality embedding)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        normalized_shape: int,
        conditioning_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.conditioning_dim = conditioning_dim
        self.eps = eps

        # Layer normalization (without learnable affine parameters)
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)

        # Project conditioning to scale and shift parameters
        self.adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_dim, 2 * normalized_shape),
        )

        # Initialize projection to output (1, 0) initially (identity transformation)
        nn.init.zeros_(self.adaln_proj[1].weight)
        nn.init.zeros_(self.adaln_proj[1].bias)
        # Bias for scale should be 1, for shift should be 0
        self.adaln_proj[1].bias.data[:normalized_shape] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply adaptive layer normalization.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            conditioning: Conditioning vector [batch, conditioning_dim]

        Returns:
            y: Normalized and conditioned tensor [batch, seq_len, hidden_dim]
        """
        # Layer normalization
        x_norm = self.ln(x)

        # Compute scale and shift from conditioning
        adaln_params = self.adaln_proj(conditioning)  # [batch, 2 * hidden_dim]
        scale, shift = adaln_params.chunk(2, dim=-1)  # Each: [batch, hidden_dim]

        # Apply affine transformation
        # Expand scale and shift to match x_norm shape
        scale = scale.unsqueeze(1)  # [batch, 1, hidden_dim]
        shift = shift.unsqueeze(1)  # [batch, 1, hidden_dim]

        y = scale * x_norm + shift

        return y


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero: Adaptive Layer Normalization with gating.

    Extends AdaLN with a learned gate parameter that allows the model
    to gradually incorporate the conditioning signal during training.

    Applies: y = γ(c) * LayerNorm(x) + β(c)
    where the residual connection is gated: x + gate(c) * sublayer(y)

    Args:
        normalized_shape: Shape of the input to be normalized
        conditioning_dim: Dimension of conditioning vector
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        normalized_shape: int,
        conditioning_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.conditioning_dim = conditioning_dim
        self.eps = eps

        # Layer normalization
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)

        # Project conditioning to scale, shift, and gate
        self.adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_dim, 3 * normalized_shape),
        )

        # Initialize to identity
        nn.init.zeros_(self.adaln_proj[1].weight)
        nn.init.zeros_(self.adaln_proj[1].bias)
        # Scale = 1, Shift = 0, Gate = 0 (initially no conditioning)
        self.adaln_proj[1].bias.data[:normalized_shape] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply AdaLN-Zero.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            conditioning: Conditioning vector [batch, conditioning_dim]

        Returns:
            y: Normalized tensor [batch, seq_len, hidden_dim]
            gate: Gate value [batch, hidden_dim] for residual connection
        """
        # Layer normalization
        x_norm = self.ln(x)

        # Compute scale, shift, and gate from conditioning
        adaln_params = self.adaln_proj(conditioning)  # [batch, 3 * hidden_dim]
        scale, shift, gate = adaln_params.chunk(3, dim=-1)

        # Apply affine transformation
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        y = scale * x_norm + shift

        return y, gate  # gate is [batch, hidden_dim] for per-channel gating


class AdaLNTransformerBlock(nn.Module):
    """
    Transformer block with AdaLN conditioning.

    Incorporates personality conditioning via AdaLN before self-attention
    and feedforward layers.

    Args:
        hidden_dim: Hidden dimension of transformer
        num_heads: Number of attention heads
        conditioning_dim: Dimension of conditioning vector
        mlp_ratio: Ratio of MLP hidden dim to hidden_dim (default: 4.0)
        dropout: Dropout rate
        use_adaln_zero: Whether to use AdaLN-Zero with gating
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        conditioning_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_adaln_zero: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_adaln_zero = use_adaln_zero

        # AdaLN before attention
        if use_adaln_zero:
            self.adaln1 = AdaLNZero(hidden_dim, conditioning_dim)
        else:
            self.adaln1 = AdaptiveLayerNorm(hidden_dim, conditioning_dim)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # AdaLN before MLP
        if use_adaln_zero:
            self.adaln2 = AdaLNZero(hidden_dim, conditioning_dim)
        else:
            self.adaln2 = AdaptiveLayerNorm(hidden_dim, conditioning_dim)

        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block with AdaLN conditioning.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            conditioning: Conditioning vector [batch, conditioning_dim]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        # Self-attention with AdaLN
        if self.use_adaln_zero:
            x_norm, gate1 = self.adaln1(x, conditioning)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attention_mask)
            # Apply gate to residual
            x = x + gate1.unsqueeze(1) * attn_out
        else:
            x_norm = self.adaln1(x, conditioning)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attention_mask)
            x = x + attn_out

        # MLP with AdaLN
        if self.use_adaln_zero:
            x_norm, gate2 = self.adaln2(x, conditioning)
            mlp_out = self.mlp(x_norm)
            x = x + gate2.unsqueeze(1) * mlp_out
        else:
            x_norm = self.adaln2(x, conditioning)
            mlp_out = self.mlp(x_norm)
            x = x + mlp_out

        return x


if __name__ == '__main__':
    """Test AdaLN modules."""
    print("Testing Adaptive Layer Normalization...\n")

    batch_size = 4
    seq_len = 16
    hidden_dim = 256
    conditioning_dim = 64

    # Test AdaptiveLayerNorm
    print("Testing AdaptiveLayerNorm:")
    adaln = AdaptiveLayerNorm(hidden_dim, conditioning_dim)

    x = torch.randn(batch_size, seq_len, hidden_dim)
    c = torch.randn(batch_size, conditioning_dim)

    y = adaln(x, c)
    print(f"  Input shape: {x.shape}")
    print(f"  Conditioning shape: {c.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Output mean: {y.mean().item():.4f}, std: {y.std().item():.4f}")

    # Test AdaLNZero
    print("\nTesting AdaLNZero:")
    adaln_zero = AdaLNZero(hidden_dim, conditioning_dim)

    y, gate = adaln_zero(x, c)
    print(f"  Output shape: {y.shape}")
    print(f"  Gate shape: {gate.shape}")
    print(f"  Gate mean: {gate.mean().item():.4f}, std: {gate.std().item():.4f}")

    # Test AdaLNTransformerBlock
    print("\nTesting AdaLNTransformerBlock:")
    block = AdaLNTransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=8,
        conditioning_dim=conditioning_dim,
        use_adaln_zero=True,
    )

    y = block(x, c)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")

    # Test with attention mask
    print("\nTesting with attention mask:")
    attn_mask = torch.zeros(seq_len, seq_len).bool()
    attn_mask = torch.triu(attn_mask, diagonal=1)  # Causal mask

    y = block(x, c, attention_mask=attn_mask)
    print(f"  Output shape with mask: {y.shape}")

    print("\n✓ AdaLN test passed!")
