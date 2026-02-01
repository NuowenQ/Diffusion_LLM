"""
Causal Structural Causal Model (SCM) Layer

Implements the causal pathway:
Personality → Beliefs/Attitudes → Behavioral Intentions → Actions

Uses DAG-constrained linear SCM with Pearl's do-calculus for interventions.

Based on:
- CausalVAE (Yang et al., CVPR 2021)
- NOTEARS for acyclicity constraint (Zheng et al., NeurIPS 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class CausalSCMLayer(nn.Module):
    """
    Structural Causal Model layer with DAG constraints.

    Implements: z_causal = (I - A^T)^{-1} @ z_exogenous

    where A is a strictly lower-triangular adjacency matrix representing
    the causal DAG structure.

    Causal Structure:
        z[0:15]   - Personality (exogenous)
        z[16:31]  - Beliefs/Attitudes (caused by personality)
        z[32:47]  - Behavioral Intentions (caused by beliefs + personality)
        z[48:63]  - Actions (caused by all above)

    Args:
        num_vars: Total number of latent variables (default: 64)
        structure_config: Dict specifying variable groups
        learnable_structure: Whether to learn the DAG structure (default: True)
        prior_adjacency: Optional prior adjacency matrix to initialize with
    """

    def __init__(
        self,
        num_vars: int = 64,
        structure_config: Optional[Dict] = None,
        learnable_structure: bool = True,
        prior_adjacency: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.num_vars = num_vars
        self.learnable_structure = learnable_structure

        # Default structure if not provided
        if structure_config is None:
            structure_config = {
                'personality': (0, 16),
                'beliefs': (16, 32),
                'intentions': (32, 48),
                'actions': (48, 64),
            }

        self.structure_config = structure_config

        # Initialize adjacency matrix A
        if prior_adjacency is not None:
            assert prior_adjacency.shape == (num_vars, num_vars)
            A_init = prior_adjacency.clone()
        else:
            A_init = self._create_prior_structure()

        if learnable_structure:
            # Learnable adjacency matrix (unconstrained)
            self.A_raw = nn.Parameter(A_init)
        else:
            # Fixed structure
            self.register_buffer('A_raw', A_init)

    def _create_prior_structure(self) -> torch.Tensor:
        """
        Create prior adjacency matrix based on structure config.

        Creates a lower-triangular structure where:
        - Personality → Beliefs
        - Personality, Beliefs → Intentions
        - Personality, Beliefs, Intentions → Actions
        """
        A = torch.zeros(self.num_vars, self.num_vars)

        # Extract ranges
        p_start, p_end = self.structure_config['personality']
        b_start, b_end = self.structure_config['beliefs']
        i_start, i_end = self.structure_config['intentions']
        a_start, a_end = self.structure_config['actions']

        # Personality → Beliefs (small random initialization)
        A[b_start:b_end, p_start:p_end] = torch.randn(b_end - b_start, p_end - p_start) * 0.01

        # Personality, Beliefs → Intentions
        A[i_start:i_end, p_start:p_end] = torch.randn(i_end - i_start, p_end - p_start) * 0.01
        A[i_start:i_end, b_start:b_end] = torch.randn(i_end - i_start, b_end - b_start) * 0.01

        # All → Actions
        A[a_start:a_end, p_start:p_end] = torch.randn(a_end - a_start, p_end - p_start) * 0.01
        A[a_start:a_end, b_start:b_end] = torch.randn(a_end - a_start, b_end - b_start) * 0.01
        A[a_start:a_end, i_start:i_end] = torch.randn(a_end - a_start, i_end - i_start) * 0.01

        return A

    def get_adjacency_matrix(self, apply_threshold: bool = True) -> torch.Tensor:
        """
        Get the adjacency matrix A.

        Args:
            apply_threshold: Whether to threshold small values to 0 (default: True)

        Returns:
            A: Adjacency matrix [num_vars, num_vars]
        """
        A = self.A_raw

        # Apply strict lower-triangular constraint (ensure acyclicity)
        A = torch.tril(A, diagonal=-1)

        # Threshold small values for sparsity
        if apply_threshold:
            A = torch.where(torch.abs(A) > 0.01, A, torch.zeros_like(A))

        return A

    def forward(self, z_exogenous: torch.Tensor) -> torch.Tensor:
        """
        Apply structural causal model to transform exogenous variables.

        z_causal = (I - A^T)^{-1} @ z_exogenous

        Args:
            z_exogenous: Exogenous latent variables [batch, num_vars]

        Returns:
            z_causal: Causal latent variables [batch, num_vars]
        """
        batch_size = z_exogenous.size(0)
        A = self.get_adjacency_matrix(apply_threshold=False)

        # Compute (I - A^T)
        I = torch.eye(self.num_vars, device=z_exogenous.device)
        I_minus_AT = I - A.T

        # Solve linear system: z_causal = (I - A^T)^{-1} @ z_exogenous
        # Use torch.linalg.solve for efficiency
        z_causal = torch.linalg.solve(
            I_minus_AT.unsqueeze(0).expand(batch_size, -1, -1),
            z_exogenous.unsqueeze(-1)
        ).squeeze(-1)

        return z_causal

    def do_intervention(
        self,
        z_exogenous: torch.Tensor,
        intervention_dims: torch.Tensor,
        intervention_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform do-intervention: do(z_i = v)

        This implements Pearl's truncated factorization by:
        1. Setting z[intervention_dims] = intervention_values
        2. Zeroing incoming edges to intervention_dims in A
        3. Propagating through the modified SCM

        Args:
            z_exogenous: Exogenous latent variables [batch, num_vars]
            intervention_dims: Dimensions to intervene on [num_interventions]
            intervention_values: Values to set [batch, num_interventions]

        Returns:
            z_counterfactual: Counterfactual latent variables [batch, num_vars]
        """
        batch_size = z_exogenous.size(0)
        A = self.get_adjacency_matrix(apply_threshold=False)

        # Create intervened adjacency matrix (zero out incoming edges)
        A_intervened = A.clone()
        A_intervened[intervention_dims, :] = 0

        # Set intervention values
        z_exo_intervened = z_exogenous.clone()
        for i, dim in enumerate(intervention_dims):
            z_exo_intervened[:, dim] = intervention_values[:, i]

        # Apply modified SCM
        I = torch.eye(self.num_vars, device=z_exogenous.device)
        I_minus_AT = I - A_intervened.T

        z_counterfactual = torch.linalg.solve(
            I_minus_AT.unsqueeze(0).expand(batch_size, -1, -1),
            z_exo_intervened.unsqueeze(-1)
        ).squeeze(-1)

        return z_counterfactual

    def compute_acyclicity_constraint(self) -> torch.Tensor:
        """
        Compute acyclicity constraint h(A) using NOTEARS formulation.

        h(A) = tr(e^{A ⊙ A}) - d

        For a DAG, h(A) = 0. We minimize |h(A)| during training.

        Returns:
            h: Acyclicity measure (should be 0 for valid DAG)
        """
        A = self.get_adjacency_matrix(apply_threshold=False)

        # Element-wise square
        A_squared = A * A

        # Matrix exponential
        expm_A_squared = torch.matrix_exp(A_squared)

        # Trace
        h = torch.trace(expm_A_squared) - self.num_vars

        return h

    def compute_sparsity_regularization(self) -> torch.Tensor:
        """
        Compute L1 sparsity regularization on adjacency matrix.

        Returns:
            sparsity: L1 norm of adjacency matrix
        """
        A = self.get_adjacency_matrix(apply_threshold=False)
        return torch.sum(torch.abs(A))

    def get_causal_graph_edges(self, threshold: float = 0.1) -> torch.Tensor:
        """
        Extract edges from the causal graph.

        Args:
            threshold: Minimum edge weight to consider (default: 0.1)

        Returns:
            edges: Tensor of shape [num_edges, 3] with columns [from, to, weight]
        """
        A = self.get_adjacency_matrix(apply_threshold=True)

        # Find non-zero edges
        edge_indices = torch.nonzero(torch.abs(A) > threshold)
        edge_weights = A[edge_indices[:, 0], edge_indices[:, 1]]

        edges = torch.cat([
            edge_indices.float(),
            edge_weights.unsqueeze(1)
        ], dim=1)

        return edges


class CausalVAE(nn.Module):
    """
    Combined Causal VAE: Personality Encoder + SCM Layer

    Integrates the disentangled personality encoder with the causal SCM layer.

    Args:
        personality_encoder: Pre-trained personality encoder
        scm_config: Configuration for SCM layer
        freeze_encoder: Whether to freeze the personality encoder (default: True)
    """

    def __init__(
        self,
        personality_encoder: nn.Module,
        scm_config: Optional[Dict] = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.personality_encoder = personality_encoder

        if freeze_encoder:
            # Freeze personality encoder weights
            for param in self.personality_encoder.parameters():
                param.requires_grad = False

        # Create SCM layer
        latent_dim = personality_encoder.latent_dim
        self.scm_layer = CausalSCMLayer(
            num_vars=latent_dim,
            structure_config=scm_config,
        )

    def forward(
        self,
        personality: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: Personality → Encoder → SCM → Causal Latent

        Args:
            personality: Big Five scores [batch, 5]
            return_all: Whether to return all intermediate outputs

        Returns:
            Dict containing causal latent variables and optionally intermediate outputs
        """
        # Encode personality to exogenous latent space
        encoder_output = self.personality_encoder(personality, return_latent=True)
        z_exogenous = encoder_output['z']

        # Apply causal SCM
        z_causal = self.scm_layer(z_exogenous)

        output = {'z_causal': z_causal}

        if return_all:
            output.update({
                'z_exogenous': z_exogenous,
                'mu': encoder_output['mu'],
                'logvar': encoder_output['logvar'],
                'recon': encoder_output['recon'],
            })

        return output

    def generate_counterfactual(
        self,
        personality: torch.Tensor,
        intervention_dims: torch.Tensor,
        intervention_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate counterfactual by intervening on specific personality dimensions.

        Args:
            personality: Original Big Five scores [batch, 5]
            intervention_dims: Dimensions to intervene on
            intervention_values: Values to set for intervention

        Returns:
            z_counterfactual: Counterfactual causal latent [batch, latent_dim]
        """
        # Encode to exogenous space
        z_exogenous = self.personality_encoder.encode_personality(personality)

        # Apply intervention
        z_counterfactual = self.scm_layer.do_intervention(
            z_exogenous,
            intervention_dims,
            intervention_values,
        )

        return z_counterfactual

    def loss_function(
        self,
        personality: torch.Tensor,
        lambda_acyclicity: float = 1.0,
        lambda_sparsity: float = 0.01,
        kl_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full CausalVAE loss.

        L = L_encoder + λ_acyc * h(A) + λ_sparse * ||A||_1

        Args:
            personality: Big Five scores [batch, 5]
            lambda_acyclicity: Weight for acyclicity constraint
            lambda_sparsity: Weight for sparsity regularization
            kl_weight: Weight for KL divergence (for annealing)

        Returns:
            Dict of loss components
        """
        # Encoder loss
        encoder_output = self.personality_encoder(personality)
        encoder_losses = self.personality_encoder.loss_function(
            personality,
            encoder_output['recon'],
            encoder_output['mu'],
            encoder_output['logvar'],
            kl_weight=kl_weight,
        )

        # SCM constraints
        acyclicity = self.scm_layer.compute_acyclicity_constraint()
        sparsity = self.scm_layer.compute_sparsity_regularization()

        # Total loss
        total_loss = (
            encoder_losses['loss'] +
            lambda_acyclicity * torch.abs(acyclicity) +
            lambda_sparsity * sparsity
        )

        return {
            'loss': total_loss,
            'recon_loss': encoder_losses['recon_loss'],
            'kl_loss': encoder_losses['kl_loss'],
            'acyclicity': acyclicity,
            'sparsity': sparsity,
        }


if __name__ == '__main__':
    """Test the Causal SCM Layer."""
    print("Testing Causal SCM Layer...\n")

    # Create SCM layer
    scm = CausalSCMLayer(num_vars=64)

    print(f"SCM Layer created")
    print(f"  Number of variables: {scm.num_vars}")
    print(f"  Structure config: {scm.structure_config}")

    # Test forward pass
    batch_size = 8
    z_exo = torch.randn(batch_size, 64)

    z_causal = scm(z_exo)
    print(f"\nForward pass:")
    print(f"  Input shape: {z_exo.shape}")
    print(f"  Output shape: {z_causal.shape}")

    # Test intervention
    intervention_dims = torch.tensor([2])  # Intervene on extraversion
    intervention_values = torch.ones(batch_size, 1) * 0.9

    z_counterfactual = scm.do_intervention(
        z_exo,
        intervention_dims,
        intervention_values,
    )
    print(f"\nIntervention test:")
    print(f"  Intervened dims: {intervention_dims}")
    print(f"  Counterfactual shape: {z_counterfactual.shape}")
    print(f"  Intervention value at dim 2: {z_counterfactual[0, 2].item():.4f}")

    # Test acyclicity constraint
    h = scm.compute_acyclicity_constraint()
    print(f"\nAcyclicity constraint:")
    print(f"  h(A) = {h.item():.6f} (should be ~0 for valid DAG)")

    # Test sparsity
    sparsity = scm.compute_sparsity_regularization()
    print(f"\nSparsity:")
    print(f"  ||A||_1 = {sparsity.item():.4f}")

    # Test edge extraction
    edges = scm.get_causal_graph_edges(threshold=0.01)
    print(f"\nCausal graph edges:")
    print(f"  Number of edges: {edges.shape[0]}")
    if edges.shape[0] > 0:
        print(f"  Sample edges (from, to, weight):")
        for i in range(min(5, edges.shape[0])):
            print(f"    {edges[i, 0].int()} → {edges[i, 1].int()}: {edges[i, 2]:.4f}")

    print("\n✓ Causal SCM Layer test passed!")
