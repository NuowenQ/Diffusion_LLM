"""
Causal MDLM: Integrated Model

Combines CausalVAE (personality encoder + SCM) with MDLM diffusion
for personality-conditioned text generation.

Architecture:
    Big Five Personality → CausalVAE → Causal Latent → MDLM → Text
"""

import torch
import torch.nn as nn
import lightning as L
from typing import Optional, Dict

from .causal.scm_layer import CausalVAE
from .personality.encoder import create_personality_encoder
from .mdlm.diffusion import Diffusion
from .mdlm.dit_personality import PersonalityDiT


class CausalMDLM(L.LightningModule):
    """
    Integrated Causal MDLM model.
    
    Combines:
    1. Personality Encoder (β-VAE or FactorVAE)
    2. Causal SCM Layer (DAG-constrained)
    3. MDLM with personality conditioning
    
    Args:
        mdlm_config: Config for MDLM diffusion model
        encoder_config: Config for personality encoder
        tokenizer: Text tokenizer
        freeze_causal_vae: Whether to freeze CausalVAE during training
        use_cfg: Use classifier-free guidance
        cfg_dropout: CFG dropout probability
    """
    
    def __init__(
        self,
        mdlm_config,
        encoder_config: Dict,
        tokenizer,
        freeze_causal_vae: bool = False,
        use_cfg: bool = True,
        cfg_dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        
        self.mdlm_config = mdlm_config
        self.encoder_config = encoder_config
        self.tokenizer = tokenizer
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        
        # Create personality encoder
        print("Creating personality encoder...")
        personality_encoder = create_personality_encoder(
            encoder_type=encoder_config.get('encoder_type', 'beta_vae'),
            input_dim=5,
            hidden_dims=encoder_config.get('hidden_dims', [128, 256, 256, 128]),
            latent_dim=encoder_config.get('latent_dim', 64),
            beta=encoder_config.get('beta', 4.0),
            gamma=encoder_config.get('gamma'),
            dropout=encoder_config.get('dropout', 0.1),
        )
        
        # Create CausalVAE
        print("Creating CausalVAE...")
        self.causal_vae = CausalVAE(
            personality_encoder,
            scm_config=None,  # Use default structure
            freeze_encoder=freeze_causal_vae,
        )
        
        if freeze_causal_vae:
            for param in self.causal_vae.parameters():
                param.requires_grad = False
            print("CausalVAE frozen")
        
        # Create MDLM with personality conditioning
        print("Creating MDLM diffusion model...")
        
        # Modify config to use PersonalityDiT
        self.mdlm = Diffusion(mdlm_config, tokenizer)
        
        # Replace backbone with PersonalityDiT
        conditioning_dim = encoder_config.get('latent_dim', 64)
        self.mdlm.backbone = PersonalityDiT(
            mdlm_config,
            conditioning_dim=conditioning_dim,
            num_personality_tokens=8,
            vocab_size=self.mdlm.vocab_size,
        )
        
        print(f"Model created:")
        print(f"  CausalVAE parameters: {sum(p.numel() for p in self.causal_vae.parameters()):,}")
        print(f"  MDLM parameters: {sum(p.numel() for p in self.mdlm.parameters()):,}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        personality: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass: personality → causal embedding → diffusion.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            personality: Big Five scores [batch, 5]
            timesteps: Diffusion timesteps [batch] (if None, sampled randomly)
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Model logits [batch, seq_len, vocab_size]
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Apply CFG dropout during training
        if self.training and self.use_cfg:
            dropout_mask = torch.rand(batch_size, device=device) > self.cfg_dropout
            personality_masked = personality * dropout_mask.unsqueeze(-1).float()
        else:
            personality_masked = personality
        
        # Get causal personality embedding
        with torch.set_grad_enabled(not self.hparams.freeze_causal_vae):
            causal_output = self.causal_vae(personality_masked, return_all=False)
            personality_cond = causal_output['z_causal']
        
        # Sample timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.mdlm_config.model.num_timesteps,
                (batch_size,),
                device=device
            )
        
        # MDLM forward pass with personality conditioning
        logits = self.mdlm.backbone(
            input_ids,
            timesteps,
            personality_cond=personality_cond,
            mask=attention_mask
        )
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch['input_ids']
        personality = batch['personality']
        attention_mask = batch.get('attention_mask')
        
        # Forward pass
        logits = self(input_ids, personality, attention_mask=attention_mask)
        
        # Compute loss (using MDLM's loss computation)
        # Note: We need to adapt this to MDLM's loss interface
        # For now, we'll use a simplified cross-entropy loss
        
        # TODO: Integrate with MDLM's proper loss computation
        # which handles the diffusion objective
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch['input_ids']
        personality = batch['personality']
        attention_mask = batch.get('attention_mask')
        
        # Forward pass
        logits = self(input_ids, personality, attention_mask=attention_mask)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        personality: torch.Tensor,
        seq_len: int = 128,
        num_steps: int = 1000,
        temperature: float = 1.0,
        guidance_scale: float = 2.0,
        intervention_dim: Optional[int] = None,
        intervention_value: Optional[float] = None,
    ):
        """
        Generate text conditioned on personality.
        
        Args:
            personality: Big Five scores [batch, 5]
            seq_len: Length of sequence to generate
            num_steps: Number of diffusion steps
            temperature: Sampling temperature
            guidance_scale: CFG scale
            intervention_dim: Optional causal intervention dimension
            intervention_value: Optional intervention value
            
        Returns:
            Generated token IDs [batch, seq_len]
        """
        batch_size = personality.size(0)
        device = personality.device
        
        # Get causal personality embedding
        if intervention_dim is not None:
            # Apply intervention
            z_exo = self.causal_vae.personality_encoder.encode_personality(personality)
            intervention_dims = torch.tensor([intervention_dim], device=device)
            intervention_values = torch.full(
                (batch_size, 1),
                intervention_value,
                device=device
            )
            personality_cond = self.causal_vae.scm_layer.do_intervention(
                z_exo,
                intervention_dims,
                intervention_values,
            )
        else:
            causal_output = self.causal_vae(personality, return_all=False)
            personality_cond = causal_output['z_causal']
        
        # Use MDLM's sampling procedure
        # Note: We need to integrate with MDLM's sampler
        # For now, this is a placeholder
        
        # Start from mask tokens
        generated_ids = torch.full(
            (batch_size, seq_len),
            self.mdlm.mask_index,
            device=device
        )
        
        # TODO: Implement proper MDLM sampling with personality conditioning
        # This would use the ddpm_cache or analytic sampler from MDLM
        
        return generated_ids
    
    def configure_optimizers(self):
        """Configure optimizers."""
        # Use different learning rates for frozen/unfrozen components
        if self.hparams.freeze_causal_vae:
            # Only optimize MDLM
            optimizer = torch.optim.AdamW(
                self.mdlm.parameters(),
                lr=1e-4,
                weight_decay=0.01
            )
        else:
            # Optimize both with different LRs
            optimizer = torch.optim.AdamW([
                {'params': self.causal_vae.parameters(), 'lr': 1e-5},
                {'params': self.mdlm.parameters(), 'lr': 1e-4},
            ], weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
    
    def load_pretrained_causal_vae(self, checkpoint_path: str):
        """Load pretrained CausalVAE weights."""
        print(f"Loading pretrained CausalVAE from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.causal_vae.load_state_dict(checkpoint['model_state_dict'])
        print("CausalVAE loaded successfully!")


if __name__ == '__main__':
    """Test CausalMDLM."""
    print("Testing CausalMDLM integration...\n")
    
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    
    # Create dummy config
    mdlm_config = OmegaConf.create({
        'model': {
            'length': 128,
            'num_timesteps': 1000,
            'hidden_size': 768,
            'n_heads': 12,
            'n_layers': 12,
        },
        'parameterization': 'subs',
        'backbone': 'dit',
        'sampling': {'predictor': 'ddpm_cache'},
        'training': {
            'antithetic_sampling': True,
            'importance_sampling': False,
            'change_of_variables': True,
        },
        'eval': {'gen_ppl_eval_model_name_or_path': 'gpt2'},
    })
    
    encoder_config = {
        'encoder_type': 'beta_vae',
        'latent_dim': 64,
        'hidden_dims': [128, 256, 256, 128],
        'beta': 4.0,
        'dropout': 0.1,
    }
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model = CausalMDLM(
        mdlm_config,
        encoder_config,
        tokenizer,
        freeze_causal_vae=False,
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    personality = torch.rand(batch_size, 5)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Personality shape: {personality.shape}")
    
    try:
        logits = model(input_ids, personality)
        print(f"Output shape: {logits.shape}")
        print("\n✓ CausalMDLM integration test passed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
