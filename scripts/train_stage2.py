"""
Stage 2: Train Causal SCM Layer

Trains the causal structural model on top of frozen personality encoder.
Learns the DAG structure: Personality → Beliefs → Intentions → Actions

Usage:
    python scripts/train_stage2.py --encoder_path experiments/stage1_personality_encoder/best_model.pt
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
from src.models.personality.encoder import create_personality_encoder
from src.models.causal.scm_layer import CausalVAE


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
    total_acyclicity = 0
    total_sparsity = 0
    
    # Gradually increase acyclicity weight
    lambda_acyclicity = min(args.lambda_acyclicity * (epoch / args.warmup_epochs), args.lambda_acyclicity)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        personality = batch['personality'].to(device)
        
        # Compute loss
        losses = model.loss_function(
            personality,
            lambda_acyclicity=lambda_acyclicity,
            lambda_sparsity=args.lambda_sparsity,
            kl_weight=1.0,  # Full KL weight (no annealing for Stage 2)
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
        total_acyclicity += losses['acyclicity'].item()
        total_sparsity += losses['sparsity'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acyc': losses['acyclicity'].item(),
            'sparse': losses['sparsity'].item(),
        })
    
    # Compute averages
    n_batches = len(train_loader)
    return {
        'train_loss': total_loss / n_batches,
        'train_recon_loss': total_recon_loss / n_batches,
        'train_kl_loss': total_kl_loss / n_batches,
        'train_acyclicity': total_acyclicity / n_batches,
        'train_sparsity': total_sparsity / n_batches,
        'lambda_acyclicity': lambda_acyclicity,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_acyclicity = 0
    total_sparsity = 0
    
    for batch in val_loader:
        personality = batch['personality'].to(device)
        
        # Compute loss
        losses = model.loss_function(
            personality,
            lambda_acyclicity=args.lambda_acyclicity,
            lambda_sparsity=args.lambda_sparsity,
            kl_weight=1.0,
        )
        
        total_loss += losses['loss'].item()
        total_recon_loss += losses['recon_loss'].item()
        total_kl_loss += losses['kl_loss'].item()
        total_acyclicity += losses['acyclicity'].item()
        total_sparsity += losses['sparsity'].item()
    
    n_batches = len(val_loader)
    return {
        'val_loss': total_loss / n_batches,
        'val_recon_loss': total_recon_loss / n_batches,
        'val_kl_loss': total_kl_loss / n_batches,
        'val_acyclicity': total_acyclicity / n_batches,
        'val_sparsity': total_sparsity / n_batches,
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
    
    # Load pretrained personality encoder
    print(f"\nLoading pretrained personality encoder from {args.encoder_path}...")
    encoder_checkpoint = torch.load(args.encoder_path, map_location=device)
    
    # Recreate encoder with same config
    encoder_args = encoder_checkpoint['args']
    personality_encoder = create_personality_encoder(
        encoder_type=encoder_args['encoder_type'],
        input_dim=5,
        hidden_dims=encoder_args['hidden_dims'],
        latent_dim=encoder_args['latent_dim'],
        beta=encoder_args['beta'],
        gamma=encoder_args.get('gamma'),
        dropout=encoder_args['dropout'],
    )
    
    # Load weights
    personality_encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    print(f"Loaded encoder from epoch {encoder_checkpoint['epoch']}")
    
    # Create dataloaders
    print("\nLoading PANDORA dataset...")
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )
    
    # Create CausalVAE
    print(f"\nCreating CausalVAE model...")
    model = CausalVAE(
        personality_encoder,
        scm_config=None,  # Use default structure
        freeze_encoder=args.freeze_encoder,
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create optimizer (only for SCM layer if encoder is frozen)
    if args.freeze_encoder:
        optimizer = optim.Adam(model.scm_layer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
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
        val_metrics = validate(model, val_loader, device, args)
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Train Acyclicity: {train_metrics['train_acyclicity']:.6f}")
        print(f"  Train Sparsity: {train_metrics['train_sparsity']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Acyclicity: {val_metrics['val_acyclicity']:.6f}")
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
    
    # Extract and visualize causal graph
    print(f"\nExtracting causal graph structure...")
    edges = model.scm_layer.get_causal_graph_edges(threshold=0.1)
    print(f"  Number of edges: {edges.shape[0]}")
    
    # Save edges
    torch.save(edges, output_dir / 'causal_graph_edges.pt')
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stage 2: Causal SCM Layer')
    
    # Pretrained encoder
    parser.add_argument('--encoder_path', type=str,
                       default='experiments/stage1_personality_encoder/best_model.pt',
                       help='Path to pretrained personality encoder')
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                       help='Freeze personality encoder weights')
    
    # Data arguments
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer name')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # SCM arguments
    parser.add_argument('--lambda_acyclicity', type=float, default=1.0,
                       help='Weight for acyclicity constraint')
    parser.add_argument('--lambda_sparsity', type=float, default=0.01,
                       help='Weight for sparsity regularization')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                       help='Epochs to warm up acyclicity constraint')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='experiments/stage2_causal_scm',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)
