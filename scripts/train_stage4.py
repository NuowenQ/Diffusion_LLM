"""
Stage 4: Joint Fine-tuning of Full Pipeline

Fine-tunes the complete model end-to-end:
Personality Encoder → CausalVAE → MDLM

Usage:
    python scripts/train_stage4.py \
        --mdlm_path experiments/stage3_mdlm/best_model.pt \
        --causal_vae_path experiments/stage2_causal_scm/best_model.pt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

from src.data.pandora_dataset import create_dataloaders
from src.models.personality.encoder import create_personality_encoder
from src.models.causal.scm_layer import CausalVAE
from src.models.diffusion.mdlm import MDLM, MDLMConfig
from src.models.conditioning.cfg import CFGWrapper


class EndToEndModel(nn.Module):
    """
    End-to-end model combining all components.
    
    Args:
        causal_vae: CausalVAE (personality encoder + SCM)
        mdlm: MDLM diffusion model
    """
    
    def __init__(self, causal_vae: nn.Module, mdlm: nn.Module):
        super().__init__()
        self.causal_vae = causal_vae
        self.mdlm = mdlm
    
    def forward(
        self,
        input_ids: torch.Tensor,
        personality: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        Forward pass: personality → causal embedding → diffusion loss.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            personality: Big Five scores [batch, 5]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Dictionary with losses
        """
        # Get causal personality embedding
        causal_output = self.causal_vae(personality, return_all=True)
        personality_cond = causal_output['z_causal']
        
        # Compute diffusion loss
        diffusion_loss_dict = self.mdlm.compute_loss(
            input_ids,
            personality_cond,
            attention_mask=attention_mask,
        )
        
        # Compute causal VAE loss (for joint training)
        causal_loss_dict = self.causal_vae.loss_function(
            personality,
            lambda_acyclicity=0.1,  # Smaller weight for fine-tuning
            lambda_sparsity=0.01,
            kl_weight=1.0,
        )
        
        # Combined loss
        total_loss = diffusion_loss_dict['loss'] + 0.1 * causal_loss_dict['loss']
        
        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss_dict['loss'],
            'diffusion_accuracy': diffusion_loss_dict['accuracy'],
            'causal_loss': causal_loss_dict['loss'],
            'recon_loss': causal_loss_dict['recon_loss'],
            'acyclicity': causal_loss_dict['acyclicity'],
        }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_diffusion_loss = 0
    total_causal_loss = 0
    total_accuracy = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        personality = batch['personality'].to(device)
        
        # Forward pass
        loss_dict = model(input_ids, personality, attention_mask)
        
        # Backward pass
        loss = loss_dict['loss']
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_diffusion_loss += loss_dict['diffusion_loss'].item()
        total_causal_loss += loss_dict['causal_loss'].item()
        total_accuracy += loss_dict['diffusion_accuracy'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'diff': loss_dict['diffusion_loss'].item(),
            'acc': loss_dict['diffusion_accuracy'].item(),
        })
    
    # Compute averages
    n_batches = len(train_loader)
    return {
        'train_loss': total_loss / n_batches,
        'train_diffusion_loss': total_diffusion_loss / n_batches,
        'train_causal_loss': total_causal_loss / n_batches,
        'train_accuracy': total_accuracy / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    total_diffusion_loss = 0
    total_causal_loss = 0
    total_accuracy = 0
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        personality = batch['personality'].to(device)
        
        # Forward pass
        loss_dict = model(input_ids, personality, attention_mask)
        
        total_loss += loss_dict['loss'].item()
        total_diffusion_loss += loss_dict['diffusion_loss'].item()
        total_causal_loss += loss_dict['causal_loss'].item()
        total_accuracy += loss_dict['diffusion_accuracy'].item()
    
    n_batches = len(val_loader)
    return {
        'val_loss': total_loss / n_batches,
        'val_diffusion_loss': total_diffusion_loss / n_batches,
        'val_causal_loss': total_causal_loss / n_batches,
        'val_accuracy': total_accuracy / n_batches,
    }


def main(args):
    """Main training loop."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        mask_token_id = tokenizer.mask_token_id
    
    # Load pretrained CausalVAE
    print(f"\nLoading pretrained CausalVAE from {args.causal_vae_path}...")
    causal_vae_checkpoint = torch.load(args.causal_vae_path, map_location=device)
    
    encoder_args = causal_vae_checkpoint['args']
    personality_encoder = create_personality_encoder(
        encoder_type=encoder_args['encoder_type'],
        input_dim=5,
        hidden_dims=encoder_args['hidden_dims'],
        latent_dim=encoder_args['latent_dim'],
        beta=encoder_args['beta'],
        gamma=encoder_args.get('gamma'),
        dropout=encoder_args['dropout'],
    )
    
    causal_vae = CausalVAE(personality_encoder, scm_config=None, freeze_encoder=False)
    causal_vae.load_state_dict(causal_vae_checkpoint['model_state_dict'])
    causal_vae = causal_vae.to(device)
    
    # Unfreeze for joint training
    for param in causal_vae.parameters():
        param.requires_grad = True
    
    print(f"Loaded CausalVAE from epoch {causal_vae_checkpoint['epoch']}")
    
    # Load pretrained MDLM
    print(f"\nLoading pretrained MDLM from {args.mdlm_path}...")
    mdlm_checkpoint = torch.load(args.mdlm_path, map_location=device)
    
    config = mdlm_checkpoint['config']
    mdlm = MDLM(config, mask_token_id)
    
    # Handle CFG wrapper
    if 'module.model.token_embedding.weight' in mdlm_checkpoint['model_state_dict']:
        # Model was wrapped in CFGWrapper
        model_dict = {}
        for k, v in mdlm_checkpoint['model_state_dict'].items():
            if k.startswith('module.model.'):
                model_dict[k.replace('module.model.', '')] = v
            elif k.startswith('model.'):
                model_dict[k.replace('model.', '')] = v
        mdlm.load_state_dict(model_dict)
    else:
        mdlm.load_state_dict(mdlm_checkpoint['model_state_dict'])
    
    mdlm = mdlm.to(device)
    
    print(f"Loaded MDLM from epoch {mdlm_checkpoint['epoch']}")
    
    # Create end-to-end model
    print(f"\nCreating end-to-end model...")
    model = EndToEndModel(causal_vae, mdlm)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  CausalVAE: {sum(p.numel() for p in causal_vae.parameters()):,}")
    print(f"  MDLM: {sum(p.numel() for p in mdlm.parameters()):,}")
    
    # Create dataloaders
    print("\nLoading PANDORA dataset...")
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )
    
    # Create optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': causal_vae.parameters(), 'lr': args.lr * 0.1},  # Lower LR for pretrained
        {'params': mdlm.parameters(), 'lr': args.lr},
    ], weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\nStarting joint fine-tuning for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, args)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"    Diffusion: {train_metrics['train_diffusion_loss']:.4f}")
        print(f"    Causal: {train_metrics['train_causal_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'causal_vae_state_dict': causal_vae.state_dict(),
                'mdlm_state_dict': mdlm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config,
                'args': vars(args),
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'causal_vae_state_dict': causal_vae.state_dict(),
                'mdlm_state_dict': mdlm.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
                'args': vars(args),
            }, best_model_path)
            print(f"  ✓ New best model! Val loss: {best_val_loss:.4f}")
    
    print(f"\nJoint fine-tuning complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stage 4: Joint Fine-tuning')
    
    # Pretrained models
    parser.add_argument('--mdlm_path', type=str,
                       default='experiments/stage3_mdlm/best_model.pt',
                       help='Path to pretrained MDLM')
    parser.add_argument('--causal_vae_path', type=str,
                       default='experiments/stage2_causal_scm/best_model.pt',
                       help='Path to pretrained CausalVAE')
    
    # Data arguments
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer name')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (for MDLM; CausalVAE uses 0.1x)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='experiments/stage4_joint',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)
