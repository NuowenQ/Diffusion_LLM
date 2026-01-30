"""
Stage 1: Train Disentangled Personality Encoder

Trains the β-VAE or FactorVAE personality encoder on PANDORA dataset.

Usage:
    python scripts/train_stage1.py --encoder_type beta_vae --epochs 200
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

from src.data.pandora_dataset import create_dataloaders
from src.models.personality.encoder import create_personality_encoder, DisentangledPersonalityEncoder, FactorVAE


def capacity_annealing_schedule(epoch: int, start_epoch: int = 0, end_epoch: int = 50, max_capacity: float = 50.0):
    """
    Compute KL capacity for capacity annealing.

    Gradually increases KL capacity from 0 to max_capacity over the annealing period.
    """
    if epoch < start_epoch:
        return 0.0
    elif epoch >= end_epoch:
        return max_capacity
    else:
        return max_capacity * (epoch - start_epoch) / (end_epoch - start_epoch)


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
    total_recon_loss = 0
    total_kl_loss = 0
    total_tc_loss = 0
    total_disc_loss = 0

    # Capacity annealing
    kl_weight = capacity_annealing_schedule(
        epoch,
        start_epoch=0,
        end_epoch=args.anneal_epochs,
        max_capacity=1.0,
    )

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        personality = batch['personality'].to(device)

        # Forward pass
        output = model(personality, return_latent=True)

        # Compute loss
        if isinstance(model, FactorVAE):
            losses = model.loss_function_factor(
                personality,
                output['recon'],
                output['mu'],
                output['logvar'],
                output['z'],
                kl_weight=kl_weight,
            )
            total_tc_loss += losses['tc_loss'].item()
            total_disc_loss += losses['discriminator_loss'].item()

            # Update discriminator
            disc_loss = losses['discriminator_loss']
            optimizer.zero_grad()
            disc_loss.backward(retain_graph=True)
            optimizer.step()
        else:
            losses = model.loss_function(
                personality,
                output['recon'],
                output['mu'],
                output['logvar'],
                kl_weight=kl_weight,
            )

        # Backward pass
        loss = losses['loss']
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += losses['recon_loss'].item()
        total_kl_loss += losses['kl_loss'].item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'recon': losses['recon_loss'].item(),
            'kl': losses['kl_loss'].item(),
            'kl_weight': kl_weight,
        })

    # Compute averages
    n_batches = len(train_loader)
    metrics = {
        'train_loss': total_loss / n_batches,
        'train_recon_loss': total_recon_loss / n_batches,
        'train_kl_loss': total_kl_loss / n_batches,
        'kl_weight': kl_weight,
    }

    if isinstance(model, FactorVAE):
        metrics['train_tc_loss'] = total_tc_loss / n_batches
        metrics['train_disc_loss'] = total_disc_loss / n_batches

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    for batch in val_loader:
        personality = batch['personality'].to(device)

        # Forward pass
        output = model(personality)

        # Compute loss (with full KL weight)
        if isinstance(model, FactorVAE):
            losses = model.loss_function_factor(
                personality,
                output['recon'],
                output['mu'],
                output['logvar'],
                output['z'],
                kl_weight=1.0,
            )
        else:
            losses = model.loss_function(
                personality,
                output['recon'],
                output['mu'],
                output['logvar'],
                kl_weight=1.0,
            )

        total_loss += losses['loss'].item()
        total_recon_loss += losses['recon_loss'].item()
        total_kl_loss += losses['kl_loss'].item()

    n_batches = len(val_loader)
    return {
        'val_loss': total_loss / n_batches,
        'val_recon_loss': total_recon_loss / n_batches,
        'val_kl_loss': total_kl_loss / n_batches,
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

    # Create dataloaders
    print("\nLoading PANDORA dataset...")
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    # Create model
    print(f"\nCreating {args.encoder_type} model...")
    model = create_personality_encoder(
        encoder_type=args.encoder_type,
        input_dim=5,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        beta=args.beta,
        gamma=args.gamma if args.encoder_type == 'factor_vae' else None,
        dropout=args.dropout,
    )
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
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
        print(f"  Train Recon: {train_metrics['train_recon_loss']:.4f}")
        print(f"  Train KL: {train_metrics['train_kl_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Recon: {val_metrics['val_recon_loss']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': vars(args),
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args),
            }, best_model_path)
            print(f"  ✓ New best model! Val loss: {best_val_loss:.4f}")

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stage 1: Personality Encoder')

    # Model arguments
    parser.add_argument('--encoder_type', type=str, default='beta_vae',
                        choices=['beta_vae', 'factor_vae'],
                        help='Type of encoder to use')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 256, 256, 128],
                        help='Hidden layer dimensions')
    parser.add_argument('--beta', type=float, default=4.0,
                        help='Beta weight for KL divergence')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma weight for total correlation (FactorVAE only)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Data arguments
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='Tokenizer name')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm')
    parser.add_argument('--anneal_epochs', type=int, default=150,
                        help='Number of epochs for capacity annealing')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='experiments/stage1_personality_encoder',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()
    main(args)
