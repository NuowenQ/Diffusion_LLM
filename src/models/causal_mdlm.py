"""
Causal MDLM: Integrated Model

Combines CausalVAE (personality encoder + SCM) with MDLM diffusion
for personality-conditioned text generation.

Architecture:
    Big Five Personality → CausalVAE → Causal Latent → MDLM → Text

This module implements the MDLM diffusion loss directly (rather than
wrapping the standalone Diffusion Lightning module) so that personality
conditioning flows correctly through the forward pass.
"""

import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import transformers
from typing import Optional, Dict

from .causal.scm_layer import CausalVAE
from .personality.encoder import create_personality_encoder
from .mdlm import noise_schedule
from .mdlm.dit_personality import PersonalityDiT


def _sample_categorical(categorical_probs):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


class CausalMDLM(L.LightningModule):
    """
    Integrated Causal MDLM model.

    Combines:
    1. Personality Encoder (β-VAE or FactorVAE)
    2. Causal SCM Layer (DAG-constrained)
    3. MDLM with personality conditioning (PersonalityDiT backbone)

    Implements MDLM diffusion loss directly with personality conditioning
    injected through the PersonalityDiT backbone.

    Args:
        config: OmegaConf config with model/noise/training/sampling sections
        encoder_config: Dict for personality encoder
        tokenizer: Text tokenizer
        freeze_causal_vae: Whether to freeze CausalVAE during training
        use_cfg: Use classifier-free guidance
        cfg_dropout: CFG dropout probability
    """

    def __init__(
        self,
        config,
        encoder_config: Dict,
        tokenizer,
        freeze_causal_vae: bool = False,
        use_cfg: bool = True,
        cfg_dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])

        self.config = config
        self.encoder_config = encoder_config
        self.tokenizer = tokenizer
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout

        # Vocab setup
        self.vocab_size = tokenizer.vocab_size
        if (not hasattr(tokenizer, 'mask_token')
                or tokenizer.mask_token is None):
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id

        # Diffusion config
        self.parameterization = config.parameterization
        self.T = config.get('T', 0)
        self.subs_masking = config.get('subs_masking', False)
        self.antithetic_sampling = config.training.antithetic_sampling
        self.importance_sampling = config.training.importance_sampling
        self.change_of_variables = config.training.change_of_variables
        self.sampling_eps = config.training.get('sampling_eps', 1e-5)
        self.time_conditioning = config.get('time_conditioning', True)
        self.neg_infinity = -1000000.0

        # Noise schedule
        self.noise = noise_schedule.get_noise(config, dtype=self.dtype)

        # Create personality encoder
        conditioning_dim = encoder_config.get('latent_dim', 64)

        personality_encoder = create_personality_encoder(
            encoder_type=encoder_config.get('encoder_type', 'beta_vae'),
            input_dim=5,
            hidden_dims=encoder_config.get('hidden_dims', [128, 256, 256, 128]),
            latent_dim=conditioning_dim,
            beta=encoder_config.get('beta', 4.0),
            gamma=encoder_config.get('gamma'),
            dropout=encoder_config.get('dropout', 0.1),
        )

        # Create CausalVAE
        self.causal_vae = CausalVAE(
            personality_encoder,
            scm_config=None,
            freeze_encoder=freeze_causal_vae,
        )

        if freeze_causal_vae:
            for param in self.causal_vae.parameters():
                param.requires_grad = False

        # Create PersonalityDiT backbone
        self.backbone = PersonalityDiT(
            config,
            conditioning_dim=conditioning_dim,
            num_personality_tokens=encoder_config.get(
                'num_personality_tokens', 8),
            vocab_size=self.vocab_size,
        )

        self.softplus = nn.Softplus()

        # Validate config
        self._validate_configuration()

    def _validate_configuration(self):
        assert not (self.change_of_variables
                    and self.importance_sampling)
        if self.parameterization == 'sedd':
            assert not self.importance_sampling
            assert not self.change_of_variables
        if self.parameterization == 'd3pm':
            assert self.T > 0
        if self.T > 0:
            assert self.parameterization in {'d3pm', 'subs'}
        if self.subs_masking:
            assert self.parameterization == 'd3pm'

    def _get_personality_cond(self, personality, batch_size, device):
        """Get personality conditioning with optional CFG dropout."""
        if self.training and self.use_cfg:
            dropout_mask = torch.rand(batch_size, device=device) > self.cfg_dropout
            personality = personality * dropout_mask.unsqueeze(-1).float()

        with torch.set_grad_enabled(
                not self.hparams.freeze_causal_vae):
            causal_output = self.causal_vae(personality, return_all=False)
            personality_cond = causal_output['z_causal']

        return personality_cond

    def _backbone_forward(self, x, sigma, personality_cond=None):
        """Forward pass through backbone with personality conditioning."""
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.backbone(
                x, sigma,
                personality_cond=personality_cond)

        if self.parameterization == 'subs':
            return self._subs_parameterization(logits, x)
        elif self.parameterization == 'd3pm':
            return self._d3pm_parameterization(logits)
        return logits

    def _process_sigma(self, sigma):
        if sigma is None:
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _subs_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _d3pm_parameterization(self, logits):
        if self.subs_masking:
            logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return logits

    def q_xt(self, x, move_chance):
        """Compute noisy sample xt by masking tokens."""
        move_indices = torch.rand(
            *x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def _forward_pass_diffusion(self, x0, personality_cond):
        """Core MDLM diffusion loss with personality conditioning."""
        t = self._sample_t(x0.shape[0], x0.device)
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            t += (1 / self.T)

        if self.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
            f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance)
        model_output = self._backbone_forward(
            xt, unet_conditioning, personality_cond=personality_cond)

        if self.T > 0:
            diffusion_loss = self._d3pm_loss(
                model_output=model_output, xt=xt, x0=x0, t=t)
            if self.parameterization == 'subs':
                return diffusion_loss
            reconstruction_loss = self._reconstruction_loss(
                x0, personality_cond)
            return reconstruction_loss + diffusion_loss

        # SUBS parameterization, continuous time
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)

        if self.change_of_variables or self.importance_sampling:
            return log_p_theta * torch.log1p(
                - torch.exp(- self.noise.sigma_min))

        return - log_p_theta * (
            dsigma / torch.expm1(sigma))[:, None]

    def _d3pm_loss(self, model_output, xt, x0, t):
        dt = 1 / self.T
        if torch.is_tensor(t):
            t = t[:, None]
            assert t.ndim == 2
            t = t.clamp(0., 1. - 1e-4)
        alpha_t = 1 - t + torch.zeros_like(xt)
        alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

        log_x_theta_at_x0 = torch.gather(
            model_output, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = model_output[:, :, self.mask_index]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(
            alpha_s * x_theta_at_m / (t - dt) + 1)

        L_vb_masked = (
            term_1_coef * (term_1_log_nr - term_1_log_dr)
            + term_2_coef * (term_2_log_nr - term_2_log_dr))

        L_vb = L_vb_masked * (xt == self.mask_index)
        return self.T * L_vb

    def _reconstruction_loss(self, x0, personality_cond):
        t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                          device=self.device)
        unet_conditioning = self.noise(t0)[0][:, None]
        model_output_t0 = self._backbone_forward(
            x0, unet_conditioning, personality_cond=personality_cond)
        return - torch.gather(
            input=model_output_t0,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)

    def _compute_loss(self, x0, attention_mask, personality_cond):
        """Full loss computation with attention masking."""
        loss = self._forward_pass_diffusion(x0, personality_cond)

        nlls = loss * attention_mask
        count = attention_mask.sum()
        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return token_nll

    def training_step(self, batch, batch_idx):
        """Training step with proper MDLM diffusion loss."""
        input_ids = batch['input_ids']
        personality = batch['personality']
        attention_mask = batch.get(
            'attention_mask',
            torch.ones_like(input_ids, dtype=torch.float))

        personality_cond = self._get_personality_cond(
            personality, input_ids.size(0), input_ids.device)

        loss = self._compute_loss(
            input_ids, attention_mask, personality_cond)

        self.log('train/loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch['input_ids']
        personality = batch['personality']
        attention_mask = batch.get(
            'attention_mask',
            torch.ones_like(input_ids, dtype=torch.float))

        personality_cond = self._get_personality_cond(
            personality, input_ids.size(0), input_ids.device)

        loss = self._compute_loss(
            input_ids, attention_mask, personality_cond)

        self.log('val/loss', loss, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def generate(
        self,
        personality: torch.Tensor,
        seq_len: int = 128,
        num_steps: int = 256,
        guidance_scale: float = 2.0,
        intervention_dim: Optional[int] = None,
        intervention_value: Optional[float] = None,
    ):
        """
        Generate text conditioned on personality using MDLM sampling.

        Args:
            personality: Big Five scores [batch, 5]
            seq_len: Length of sequence to generate
            num_steps: Number of denoising steps
            guidance_scale: CFG guidance scale (>1 for stronger conditioning)
            intervention_dim: Optional causal intervention dimension
            intervention_value: Optional intervention value

        Returns:
            Generated token IDs [batch, seq_len]
        """
        batch_size = personality.size(0)
        device = personality.device
        eps = 1e-5

        # Get personality conditioning
        if intervention_dim is not None:
            z_exo = self.causal_vae.personality_encoder.encode_personality(
                personality)
            intervention_dims = torch.tensor(
                [intervention_dim], device=device)
            intervention_values = torch.full(
                (batch_size, 1), intervention_value, device=device)
            personality_cond = self.causal_vae.scm_layer.do_intervention(
                z_exo, intervention_dims, intervention_values)
        else:
            causal_output = self.causal_vae(personality, return_all=False)
            personality_cond = causal_output['z_causal']

        # Start from all mask tokens
        x = torch.full(
            (batch_size, seq_len),
            self.mask_index,
            dtype=torch.long,
            device=device)

        # DDPM caching sampling (for loglinear noise)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(
                batch_size, 1, device=device)
            sigma_t, _ = self.noise(t)
            if t.ndim > 1:
                t_squeezed = t.squeeze(-1)
            else:
                t_squeezed = t

            move_chance_t = t_squeezed[:, None, None]
            move_chance_s = (t_squeezed - dt)[:, None, None]

            if p_x0_cache is None:
                # Conditioned prediction
                p_x0 = self._backbone_forward(
                    x, sigma_t,
                    personality_cond=personality_cond).exp()

                # CFG: also compute unconditioned prediction
                if guidance_scale > 1.0:
                    null_cond = torch.zeros_like(personality_cond)
                    p_x0_uncond = self._backbone_forward(
                        x, sigma_t,
                        personality_cond=null_cond).exp()
                    # Guided prediction
                    p_x0 = p_x0_uncond + guidance_scale * (
                        p_x0 - p_x0_uncond)
                    p_x0 = p_x0.clamp(min=0)
                    p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)
            else:
                p_x0 = p_x0_cache

            q_xs = p_x0 * (move_chance_t - move_chance_s)
            q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
            x_next = _sample_categorical(q_xs)

            copy_flag = (x != self.mask_index).to(x.dtype)
            x_new = copy_flag * x + (1 - copy_flag) * x_next

            if (not torch.allclose(x_new, x)
                    or self.time_conditioning):
                p_x0_cache = None
            else:
                p_x0_cache = p_x0

            x = x_new.long()

        # Final denoising step
        t = timesteps[-1] * torch.ones(
            batch_size, 1, device=device)
        sigma_t = self.noise(t)[0]
        logits = self._backbone_forward(
            x, sigma_t, personality_cond=personality_cond)
        x = logits.argmax(dim=-1)

        return x

    def configure_optimizers(self):
        """Configure optimizers with separate LRs for components."""
        lr = self.config.optim.get('lr', 1e-4)

        if self.hparams.freeze_causal_vae:
            params = itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters())
            optimizer = torch.optim.AdamW(
                params, lr=lr,
                weight_decay=self.config.optim.get('weight_decay', 0.01))
        else:
            optimizer = torch.optim.AdamW([
                {'params': self.causal_vae.parameters(),
                 'lr': lr * 0.1},
                {'params': self.backbone.parameters(),
                 'lr': lr},
                {'params': self.noise.parameters(),
                 'lr': lr},
            ], weight_decay=self.config.optim.get('weight_decay', 0.01))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

    def load_pretrained_causal_vae(self, checkpoint_path: str):
        """Load pretrained CausalVAE weights."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device)
        self.causal_vae.load_state_dict(
            checkpoint['model_state_dict'])


if __name__ == '__main__':
    """Test CausalMDLM."""
    print("Testing CausalMDLM integration...\n")

    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    # Full config matching Diffusion requirements
    config = OmegaConf.create({
        'model': {
            'length': 128,
            'hidden_size': 256,
            'n_heads': 8,
            'n_blocks': 6,
            'cond_dim': 256,
            'dropout': 0.1,
            'scale_by_sigma': False,
        },
        'parameterization': 'subs',
        'T': 0,
        'subs_masking': False,
        'time_conditioning': True,
        'noise': {
            'type': 'loglinear',
        },
        'training': {
            'antithetic_sampling': True,
            'importance_sampling': False,
            'change_of_variables': True,
            'sampling_eps': 1e-5,
        },
        'optim': {
            'lr': 1e-4,
            'weight_decay': 0.01,
        },
        'sampling': {
            'predictor': 'ddpm_cache',
            'steps': 256,
        },
    })

    encoder_config = {
        'encoder_type': 'beta_vae',
        'latent_dim': 64,
        'hidden_dims': [128, 256, 256, 128],
        'beta': 4.0,
        'dropout': 0.1,
    }

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CausalMDLM(
        config,
        encoder_config,
        tokenizer,
        freeze_causal_vae=False,
    )

    # Test forward pass
    batch_size = 4
    seq_len = 32

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    personality = torch.rand(batch_size, 5)
    attention_mask = torch.ones(batch_size, seq_len)

    batch = {
        'input_ids': input_ids,
        'personality': personality,
        'attention_mask': attention_mask,
    }

    print(f"Input shape: {input_ids.shape}")
    print(f"Personality shape: {personality.shape}")

    try:
        loss = model.training_step(batch, 0)
        print(f"Loss: {loss.item():.4f}")
        print("\nCausalMDLM integration test passed!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
