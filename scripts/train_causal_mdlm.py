"""
Train Causal MDLM - Integrated Training Script

Trains the complete Causal MDLM system:
- CausalVAE (personality encoder + SCM)
- MDLM diffusion with personality conditioning

This uses the official MDLM implementation as the backbone.

Usage:
    # Train from scratch
    python scripts/train_causal_mdlm.py

    # Load pretrained CausalVAE
    python scripts/train_causal_mdlm.py \
        --causal_vae_path experiments/stage2_causal_scm/best_model.pt \
        --freeze_causal_vae
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import json
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.models.causal_mdlm import CausalMDLM
from src.data.pandora_mdlm_dataloader import create_mdlm_dataloaders


def create_mdlm_config(args):
    """Create MDLM configuration."""
    config = OmegaConf.create({
        'model': {
            'length': args.max_length,
            'num_timesteps': args.num_timesteps,
            'hidden_size': args.hidden_dim,
            'n_heads': args.num_heads,
            'n_layers': args.num_layers,
            'dropout': args.dropout,
        },
        'parameterization': 'subs',  # MDLM's SUBStitution parameterization
        'backbone': 'dit',
        'sampling': {
            'predictor': 'ddpm_cache',  # Fast MDLM sampler
            'steps': 1000,
        },
        'training': {
            'antithetic_sampling': True,
            'importance_sampling': False,
            'change_of_variables': True,
        },
        'eval': {
            'gen_ppl_eval_model_name_or_path': 'gpt2',
            'compute_generative_perplexity': False,
        },
        'time_conditioning': True,
    })
    
    return config


def create_encoder_config(args):
    """Create personality encoder configuration."""
    config = {
        'encoder_type': args.encoder_type,
        'latent_dim': args.latent_dim,
        'hidden_dims': args.encoder_hidden_dims,
        'beta': args.beta,
        'gamma': args.gamma if args.encoder_type == 'factor_vae' else None,
        'dropout': args.dropout,
    }
    
    return config


def main(args):
    """Main training function."""
    # Set random seed
    L.seed_everything(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*60)
    print("Training Causal MDLM")
    print("="*60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add mask token if needed
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    # Create dataloaders
    print("\nCreating PANDORA dataloaders...")
    train_loader, val_loader, test_loader = create_mdlm_dataloaders(
        batch_size=args.batch_size,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )
    
    # Create configs
    mdlm_config = create_mdlm_config(args)
    encoder_config = create_encoder_config(args)
    
    # Create model
    print("\nCreating Causal MDLM model...")
    model = CausalMDLM(
        mdlm_config=mdlm_config,
        encoder_config=encoder_config,
        tokenizer=tokenizer,
        freeze_causal_vae=args.freeze_causal_vae,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
    )
    
    # Load pretrained CausalVAE if provided
    if args.causal_vae_path:
        model.load_pretrained_causal_vae(args.causal_vae_path)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='causal-mdlm-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    # Setup logger
    if args.use_wandb:
        logger = WandbLogger(
            project='causal-mdlm',
            name=args.wandb_name or f'causal-mdlm-{args.output_dir.split("/")[-1]}',
            save_dir=output_dir,
        )
    else:
        logger = True  # Default logger
    
    # Create trainer
    print("\nInitializing PyTorch Lightning Trainer...")
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=args.gpus,
        strategy='ddp' if args.gpus > 1 else 'auto',
        precision=args.precision,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best model saved to: {output_dir / 'checkpoints'}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Causal MDLM')
    
    # Pretrained models
    parser.add_argument('--causal_vae_path', type=str, default=None,
                       help='Path to pretrained CausalVAE checkpoint')
    parser.add_argument('--freeze_causal_vae', action='store_true',
                       help='Freeze CausalVAE weights during training')
    
    # Personality encoder config
    parser.add_argument('--encoder_type', type=str, default='beta_vae',
                       choices=['beta_vae', 'factor_vae'],
                       help='Type of personality encoder')
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Latent dimension for personality encoding')
    parser.add_argument('--encoder_hidden_dims', type=int, nargs='+',
                       default=[128, 256, 256, 128],
                       help='Hidden dimensions for encoder')
    parser.add_argument('--beta', type=float, default=4.0,
                       help='Beta parameter for β-VAE')
    parser.add_argument('--gamma', type=float, default=10.0,
                       help='Gamma parameter for FactorVAE')
    
    # MDLM model config
    parser.add_argument('--hidden_dim', type=int, default=768,
                       help='Hidden dimension for transformer')
    parser.add_argument('--num_layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    
    # CFG config
    parser.add_argument('--use_cfg', action='store_true', default=True,
                       help='Use classifier-free guidance')
    parser.add_argument('--cfg_dropout', type=float, default=0.1,
                       help='CFG conditioning dropout probability')
    
    # Data config
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer name')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Training config
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--precision', type=str, default='16-mixed',
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    
    # Logging config
    parser.add_argument('--output_dir', type=str, default='experiments/causal_mdlm',
                       help='Output directory')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                       help='Log every N steps')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                       help='Validation check interval (epoch fraction)')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='W&B run name')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)
