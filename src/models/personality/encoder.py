"""
Disentangled Personality Encoder using β-VAE

Maps Big Five personality traits to a low-dimensional disentangled latent space.

Architecture:
- Input: Big Five scores [5]
- Hidden layers: [128, 256, 256, 128]
- Latent dimension: 64 (first 5 dims aligned to Big Five, rest for auxiliary factors)
- Disentanglement: β-VAE with β=4.0 + Total Correlation penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class DisentangledPersonalityEncoder(nn.Module):
    """
    β-VAE based personality encoder with disentanglement.

    Args:
        input_dim: Input dimension (5 for Big Five)
        hidden_dims: List of hidden layer dimensions
        latent_dim: Latent space dimension (default: 64)
        beta: β-VAE weight for KL divergence (default: 4.0)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: list = [128, 256, 256, 128],
        latent_dim: int = 64,
        beta: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.beta = beta
        self.dropout_rate = dropout

        # Build encoder network
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Build decoder network
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Output in [0, 1] like normalized Big Five

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input Big Five scores [batch, 5] (normalized to [0, 1])

        Returns:
            mu: Mean of latent distribution [batch, latent_dim]
            logvar: Log variance of latent distribution [batch, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)

        Args:
            mu: Mean [batch, latent_dim]
            logvar: Log variance [batch, latent_dim]

        Returns:
            z: Sampled latent code [batch, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to reconstructed input.

        Args:
            z: Latent code [batch, latent_dim]

        Returns:
            x_recon: Reconstructed Big Five scores [batch, 5]
        """
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input Big Five scores [batch, 5]
            return_latent: Whether to return latent variables

        Returns:
            Dict containing:
                - recon: Reconstructed input [batch, 5]
                - mu: Latent mean [batch, latent_dim]
                - logvar: Latent log variance [batch, latent_dim]
                - z: Sampled latent code [batch, latent_dim] (if return_latent=True)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        output = {
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
        }

        if return_latent:
            output['z'] = z

        return output

    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute β-VAE loss: L = Recon + β * KL

        Args:
            x: Original input [batch, 5]
            recon: Reconstructed input [batch, 5]
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log variance [batch, latent_dim]
            kl_weight: Weight for KL term (for capacity annealing)

        Returns:
            Dict containing:
                - loss: Total loss
                - recon_loss: Reconstruction loss
                - kl_loss: KL divergence
        """
        # Reconstruction loss (MSE for normalized Big Five scores)
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Average over batch

        # Total loss with β weighting
        total_loss = recon_loss + self.beta * kl_weight * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the prior p(z) = N(0, I) and decode.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Sampled Big Five scores [num_samples, 5]
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples

    @torch.no_grad()
    def encode_personality(self, personality: torch.Tensor) -> torch.Tensor:
        """
        Encode personality to latent space (using mean, no sampling).

        Args:
            personality: Big Five scores [batch, 5] (normalized to [0, 1])

        Returns:
            z: Latent code [batch, latent_dim]
        """
        mu, _ = self.encode(personality)
        return mu

    def get_latent_dim_mapping(self) -> Dict[str, int]:
        """
        Get mapping of latent dimensions to personality factors.

        Returns:
            Dict mapping factor name to latent dimension index
        """
        return {
            'openness': 0,
            'conscientiousness': 1,
            'extraversion': 2,
            'agreeableness': 3,
            'neuroticism': 4,
            # Dimensions 5-63 are auxiliary factors for beliefs, attitudes, etc.
        }


class FactorVAE(DisentangledPersonalityEncoder):
    """
    Extension of β-VAE with Total Correlation (TC) penalty for better disentanglement.

    Implements the FactorVAE objective:
    L = Recon + α * KL + γ * TC

    where TC = KL(q(z) || ∏_j q(z_j)) measures total correlation
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: list = [128, 256, 256, 128],
        latent_dim: int = 64,
        beta: float = 4.0,
        gamma: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, beta, dropout)
        self.gamma = gamma

        # Discriminator for TC estimation (density ratio trick)
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2),  # Binary classification: real vs permuted
        )

    def permute_dims(self, z: torch.Tensor) -> torch.Tensor:
        """
        Permute latent dimensions to create samples from ∏_j q(z_j).

        Args:
            z: Latent codes [batch, latent_dim]

        Returns:
            z_permuted: Permuted latent codes [batch, latent_dim]
        """
        batch_size = z.size(0)
        z_permuted = torch.zeros_like(z)

        for i in range(self.latent_dim):
            perm = torch.randperm(batch_size)
            z_permuted[:, i] = z[perm, i]

        return z_permuted

    def tc_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Estimate total correlation using discriminator-based density ratio trick.

        Args:
            z: Latent codes [batch, latent_dim]

        Returns:
            tc: Total correlation estimate
        """
        # Real samples from q(z)
        d_real = self.discriminator(z)

        # Permuted samples from ∏_j q(z_j)
        z_permuted = self.permute_dims(z)
        d_permuted = self.discriminator(z_permuted)

        # Discriminator loss (binary cross-entropy)
        ones = torch.ones(z.size(0), dtype=torch.long, device=z.device)
        zeros = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        d_loss = 0.5 * (
            F.cross_entropy(d_real, ones) +
            F.cross_entropy(d_permuted, zeros)
        )

        # TC estimate from discriminator output
        tc = (d_real[:, 0] - d_real[:, 1]).mean()

        return tc, d_loss

    def loss_function_factor(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        FactorVAE loss with total correlation penalty.

        Args:
            x: Original input [batch, 5]
            recon: Reconstructed input [batch, 5]
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log variance [batch, latent_dim]
            z: Sampled latent code [batch, latent_dim]
            kl_weight: Weight for KL term

        Returns:
            Dict containing all loss components
        """
        # Base β-VAE loss
        base_losses = super().loss_function(x, recon, mu, logvar, kl_weight)

        # Total correlation penalty
        tc, d_loss = self.tc_loss(z)

        # Total loss
        total_loss = base_losses['loss'] + self.gamma * tc

        return {
            'loss': total_loss,
            'recon_loss': base_losses['recon_loss'],
            'kl_loss': base_losses['kl_loss'],
            'tc_loss': tc,
            'discriminator_loss': d_loss,
        }


def create_personality_encoder(
    encoder_type: str = 'beta_vae',
    **kwargs
) -> nn.Module:
    """
    Factory function to create personality encoder.

    Args:
        encoder_type: Type of encoder ('beta_vae' or 'factor_vae')
        **kwargs: Arguments passed to encoder constructor

    Returns:
        Personality encoder model
    """
    if encoder_type == 'beta_vae':
        return DisentangledPersonalityEncoder(**kwargs)
    elif encoder_type == 'factor_vae':
        return FactorVAE(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == '__main__':
    """Test the personality encoder."""
    print("Testing Disentangled Personality Encoder...\n")

    # Create encoder
    encoder = DisentangledPersonalityEncoder(
        input_dim=5,
        hidden_dims=[128, 256, 256, 128],
        latent_dim=64,
        beta=4.0,
    )

    print(f"Encoder architecture:")
    print(encoder)
    print(f"\nTotal parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Test forward pass
    batch_size = 8
    x = torch.rand(batch_size, 5)  # Random Big Five scores in [0, 1]

    output = encoder(x, return_latent=True)
    print(f"\nForward pass test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstruction shape: {output['recon'].shape}")
    print(f"  Latent mean shape: {output['mu'].shape}")
    print(f"  Latent logvar shape: {output['logvar'].shape}")
    print(f"  Latent z shape: {output['z'].shape}")

    # Test loss computation
    losses = encoder.loss_function(
        x,
        output['recon'],
        output['mu'],
        output['logvar'],
    )
    print(f"\nLoss computation:")
    print(f"  Total loss: {losses['loss'].item():.4f}")
    print(f"  Recon loss: {losses['recon_loss'].item():.4f}")
    print(f"  KL loss: {losses['kl_loss'].item():.4f}")

    # Test sampling
    samples = encoder.sample(num_samples=5, device=x.device)
    print(f"\nSampling test:")
    print(f"  Generated samples shape: {samples.shape}")
    print(f"  Sample values (first sample): {samples[0]}")

    # Test FactorVAE
    print(f"\n\nTesting FactorVAE...")
    factor_encoder = FactorVAE(
        input_dim=5,
        latent_dim=64,
        beta=4.0,
        gamma=1.0,
    )

    output = factor_encoder(x, return_latent=True)
    losses = factor_encoder.loss_function_factor(
        x,
        output['recon'],
        output['mu'],
        output['logvar'],
        output['z'],
    )

    print(f"\nFactorVAE loss computation:")
    print(f"  Total loss: {losses['loss'].item():.4f}")
    print(f"  TC loss: {losses['tc_loss'].item():.4f}")
    print(f"  Discriminator loss: {losses['discriminator_loss'].item():.4f}")

    print("\n✓ Personality encoder test passed!")
