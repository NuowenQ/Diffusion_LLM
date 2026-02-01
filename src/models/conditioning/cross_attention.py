"""
Cross-Attention Conditioning for Personality Embeddings

Allows text tokens to attend to personality embedding "tokens" for
fine-grained, token-level personality conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class PersonalityCrossAttention(nn.Module):
    """
    Cross-attention module where text attends to personality embeddings.

    Projects personality embedding to K pseudo-tokens that text tokens
    can attend to.

    Args:
        hidden_dim: Hidden dimension of text tokens
        conditioning_dim: Dimension of personality embedding
        num_heads: Number of attention heads
        num_personality_tokens: Number of pseudo-tokens to project personality to
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        conditioning_dim: int,
        num_heads: int = 8,
        num_personality_tokens: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.conditioning_dim = conditioning_dim
        self.num_heads = num_heads
        self.num_personality_tokens = num_personality_tokens
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Project personality embedding to K pseudo-tokens
        self.personality_proj = nn.Sequential(
            nn.Linear(conditioning_dim, num_personality_tokens * hidden_dim),
            nn.LayerNorm(num_personality_tokens * hidden_dim),
        )

        # Query projection (from text)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)

        # Key and Value projections (from personality)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        personality: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention: text tokens attend to personality tokens.

        Args:
            x: Text tokens [batch, seq_len, hidden_dim]
            personality: Personality embedding [batch, conditioning_dim]
            attention_mask: Optional mask [batch, seq_len, num_personality_tokens]

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project personality to pseudo-tokens
        personality_tokens = self.personality_proj(personality)  # [batch, K * hidden_dim]
        personality_tokens = personality_tokens.view(
            batch_size, self.num_personality_tokens, self.hidden_dim
        )  # [batch, K, hidden_dim]

        # Compute Q, K, V
        Q = self.q_proj(x)  # [batch, seq_len, hidden_dim]
        K = self.k_proj(personality_tokens)  # [batch, K, hidden_dim]
        V = self.v_proj(personality_tokens)  # [batch, K, hidden_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.num_personality_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.num_personality_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, num_heads, seq_len/K, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [batch, num_heads, seq_len, K]

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )

        output = self.out_proj(attn_output)
        output = self.dropout(output)

        return output


class CrossAttentionTransformerBlock(nn.Module):
    """
    Transformer block with both self-attention and personality cross-attention.

    Args:
        hidden_dim: Hidden dimension
        conditioning_dim: Dimension of personality embedding
        num_heads: Number of attention heads
        num_personality_tokens: Number of personality pseudo-tokens
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        conditioning_dim: int,
        num_heads: int = 8,
        num_personality_tokens: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention to personality
        self.cross_attn = PersonalityCrossAttention(
            hidden_dim,
            conditioning_dim,
            num_heads,
            num_personality_tokens,
            dropout,
        )

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
        personality: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with self-attention and cross-attention.

        Args:
            x: Input tokens [batch, seq_len, hidden_dim]
            personality: Personality embedding [batch, conditioning_dim]
            self_attn_mask: Optional self-attention mask
            cross_attn_mask: Optional cross-attention mask

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        # Self-attention
        x_norm = self.ln1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=self_attn_mask)
        x = x + attn_out

        # Cross-attention to personality
        x_norm = self.ln2(x)
        cross_out = self.cross_attn(x_norm, personality, cross_attn_mask)
        x = x + cross_out

        # MLP
        x_norm = self.ln3(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


class HybridConditioningBlock(nn.Module):
    """
    Hybrid conditioning combining AdaLN and cross-attention.

    Uses AdaLN for global personality influence and cross-attention
    for fine-grained token-level conditioning.

    Args:
        hidden_dim: Hidden dimension
        conditioning_dim: Dimension of personality embedding
        num_heads: Number of attention heads
        num_personality_tokens: Number of personality pseudo-tokens
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
        use_adaln: Whether to use AdaLN (in addition to cross-attention)
    """

    def __init__(
        self,
        hidden_dim: int,
        conditioning_dim: int,
        num_heads: int = 8,
        num_personality_tokens: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_adaln: bool = True,
    ):
        super().__init__()

        self.use_adaln = use_adaln

        if use_adaln:
            from .adaln import AdaLNZero
            self.adaln1 = AdaLNZero(hidden_dim, conditioning_dim)
            self.adaln2 = AdaLNZero(hidden_dim, conditioning_dim)
            self.adaln3 = AdaLNZero(hidden_dim, conditioning_dim)
        else:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.ln3 = nn.LayerNorm(hidden_dim)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention to personality
        self.cross_attn = PersonalityCrossAttention(
            hidden_dim,
            conditioning_dim,
            num_heads,
            num_personality_tokens,
            dropout,
        )

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
        personality: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with hybrid conditioning.

        Args:
            x: Input tokens [batch, seq_len, hidden_dim]
            personality: Personality embedding [batch, conditioning_dim]
            self_attn_mask: Optional self-attention mask

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        if self.use_adaln:
            # Self-attention with AdaLN
            x_norm, gate1 = self.adaln1(x, personality)
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=self_attn_mask)
            x = x + gate1.unsqueeze(1) * attn_out

            # Cross-attention with AdaLN
            x_norm, gate2 = self.adaln2(x, personality)
            cross_out = self.cross_attn(x_norm, personality)
            x = x + gate2.unsqueeze(1) * cross_out

            # MLP with AdaLN
            x_norm, gate3 = self.adaln3(x, personality)
            mlp_out = self.mlp(x_norm)
            x = x + gate3.unsqueeze(1) * mlp_out
        else:
            # Standard LayerNorm
            x_norm = self.ln1(x)
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=self_attn_mask)
            x = x + attn_out

            x_norm = self.ln2(x)
            cross_out = self.cross_attn(x_norm, personality)
            x = x + cross_out

            x_norm = self.ln3(x)
            mlp_out = self.mlp(x_norm)
            x = x + mlp_out

        return x


if __name__ == '__main__':
    """Test cross-attention modules."""
    print("Testing Cross-Attention Conditioning...\n")

    batch_size = 4
    seq_len = 16
    hidden_dim = 256
    conditioning_dim = 64

    # Test PersonalityCrossAttention
    print("Testing PersonalityCrossAttention:")
    cross_attn = PersonalityCrossAttention(
        hidden_dim=hidden_dim,
        conditioning_dim=conditioning_dim,
        num_heads=8,
        num_personality_tokens=8,
    )

    x = torch.randn(batch_size, seq_len, hidden_dim)
    personality = torch.randn(batch_size, conditioning_dim)

    y = cross_attn(x, personality)
    print(f"  Input shape: {x.shape}")
    print(f"  Personality shape: {personality.shape}")
    print(f"  Output shape: {y.shape}")

    # Test CrossAttentionTransformerBlock
    print("\nTesting CrossAttentionTransformerBlock:")
    block = CrossAttentionTransformerBlock(
        hidden_dim=hidden_dim,
        conditioning_dim=conditioning_dim,
        num_heads=8,
        num_personality_tokens=8,
    )

    y = block(x, personality)
    print(f"  Output shape: {y.shape}")

    # Test HybridConditioningBlock
    print("\nTesting HybridConditioningBlock:")
    hybrid_block = HybridConditioningBlock(
        hidden_dim=hidden_dim,
        conditioning_dim=conditioning_dim,
        num_heads=8,
        num_personality_tokens=8,
        use_adaln=True,
    )

    y = hybrid_block(x, personality)
    print(f"  Output shape: {y.shape}")

    print("\n✓ Cross-attention test passed!")
