"""
Stage 3: Train MDLM with Fixed Personality Conditioning

Trains the masked diffusion language model with frozen CausalVAE
providing personality conditioning.

Usage:
    python scripts/train_stage3.py --causal_vae_path experiments/stage2_causal_scm/best_model.pt
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


def train_epoch(
    model: nn.Module,
    causal_vae: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> dict:
    """Train for one epoch."""
    model.train()
    causal_vae.eval()  # Keep CausalVAE frozen
    
    total_loss = 0
    total_accuracy = 0
    total_num_masked = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        personality = batch['personality'].to(device)
        
        # Get causal personality embedding
        with torch.no_grad():
            causal_output = causal_vae(personality, return_all=False)
            personality_cond = causal_output['z_causal']
        
        # Compute diffusion loss
        loss_dict = model.compute_loss(
            input_ids,
            personality_cond,
            attention_mask=attention_mask,
        )
        
        # Backward pass
        loss = loss_dict['loss']
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_accuracy += loss_dict['accuracy'].item()
        total_num_masked += loss_dict['num_masked'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': loss_dict['accuracy'].item(),
        })
    
    # Compute averages
    n_batches = len(train_loader)
    return {
        'train_loss': total_loss / n_batches,
        'train_accuracy': total_accuracy / n_batches,
        'avg_num_masked': total_num_masked / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    causal_vae: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()
    causal_vae.eval()
    
    total_loss = 0
    total_accuracy = 0
    total_num_masked = 0
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        personality = batch['personality'].to(device)
        
        # Get causal personality embedding
        causal_output = causal_vae(personality, return_all=False)
        personality_cond = causal_output['z_causal']
        
        # Compute loss
        loss_dict = model.compute_loss(
            input_ids,
            personality_cond,
            attention_mask=attention_mask,
        )
        
        total_loss += loss_dict['loss'].item()
        total_accuracy += loss_dict['accuracy'].item()
        total_num_masked += loss_dict['num_masked'].item()
    
    n_batches = len(val_loader)
    return {
        'val_loss': total_loss / n_batches,
        'val_accuracy': total_accuracy / n_batches,
        'avg_num_masked': total_num_masked / n_batches,
    }


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    causal_vae: nn.Module,
    tokenizer,
    personalities: torch.Tensor,
    device: torch.device,
    num_samples: int = 4,
    seq_len: int = 64,
) -> list:
    """Generate sample outputs for inspection."""
    model.eval()
    causal_vae.eval()
    
    # Get causal embeddings
    causal_output = causal_vae(personalities[:num_samples], return_all=False)
    personality_cond = causal_output['z_causal']
    
    # Generate
    generated_ids = model.generate(
        personality_cond,
        seq_len=seq_len,
        temperature=1.0,
        num_steps=50,
    )
    
    # Decode
    samples = []
    for i in range(num_samples):
        text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
        samples.append({
            'personality': personalities[i].cpu().tolist(),
            'text': text,
        })
    
    return samples


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
    
    # Get special tokens
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        # Add mask token if not present
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        mask_token_id = tokenizer.mask_token_id
    
    vocab_size = len(tokenizer)
    
    # Load pretrained CausalVAE
    print(f"\nLoading pretrained CausalVAE from {args.causal_vae_path}...")
    causal_vae_checkpoint = torch.load(args.causal_vae_path, map_location=device)
    
    # Recreate CausalVAE
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
    
    causal_vae = CausalVAE(personality_encoder, scm_config=None, freeze_encoder=True)
    causal_vae.load_state_dict(causal_vae_checkpoint['model_state_dict'])
    causal_vae = causal_vae.to(device)
    
    # Freeze CausalVAE
    for param in causal_vae.parameters():
        param.requires_grad = False
    
    print(f"Loaded CausalVAE from epoch {causal_vae_checkpoint['epoch']}")
    print(f"Conditioning dimension: {personality_encoder.latent_dim}")
    
    # Create dataloaders
    print("\nLoading PANDORA dataset...")
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )
    
    # Create MDLM
    print(f"\nCreating MDLM model...")
    config = MDLMConfig(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        conditioning_dim=personality_encoder.latent_dim,
        max_seq_len=args.max_length,
        dropout=args.dropout,
        conditioning_type=args.conditioning_type,
        use_adaln_zero=args.use_adaln_zero,
        num_personality_tokens=args.num_personality_tokens,
        num_timesteps=args.num_timesteps,
        noise_schedule=args.noise_schedule,
    )
    
    mdlm = MDLM(config, mask_token_id)
    
    # Wrap with CFG if requested
    if args.use_cfg:
        print(f"Wrapping with CFG (dropout_prob={args.cfg_dropout})")
        model = CFGWrapper(mdlm, dropout_prob=args.cfg_dropout)
    else:
        model = mdlm
    
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, causal_vae, train_loader, optimizer, device, epoch, args)
        
        # Validate
        val_metrics = validate(model, causal_vae, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Train Accuracy: {train_metrics['train_accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Generate samples every N epochs
        if epoch % args.sample_every == 0:
            print(f"\nGenerating samples...")
            # Get some validation personalities
            val_batch = next(iter(val_loader))
            personalities = val_batch['personality'].to(device)
            
            samples = generate_samples(
                mdlm if args.use_cfg else model,
                causal_vae,
                tokenizer,
                personalities,
                device,
            )
            
            # Save samples
            samples_path = output_dir / f'samples_epoch_{epoch}.json'
            with open(samples_path, 'w') as f:
                json.dump(samples, f, indent=2)
            
            print(f"Sample generations:")
            for i, sample in enumerate(samples[:2]):
                print(f"  {i+1}. {sample['text'][:100]}...")
        
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
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
                'args': vars(args),
            }, best_model_path)
            print(f"  ✓ New best model! Val loss: {best_val_loss:.4f}")
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stage 3: MDLM')
    
    # Pretrained models
    parser.add_argument('--causal_vae_path', type=str,
                       default='experiments/stage2_causal_scm/best_model.pt',
                       help='Path to pretrained CausalVAE')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=768,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--conditioning_type', type=str, default='hybrid',
                       choices=['adaln', 'cross_attn', 'hybrid'],
                       help='Type of conditioning')
    parser.add_argument('--use_adaln_zero', action='store_true', default=True,
                       help='Use AdaLN-Zero gating')
    parser.add_argument('--num_personality_tokens', type=int, default=8,
                       help='Number of personality pseudo-tokens for cross-attention')
    
    # Diffusion arguments
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--noise_schedule', type=str, default='cosine',
                       choices=['cosine', 'linear', 'learned'],
                       help='Noise schedule type')
    
    # CFG arguments
    parser.add_argument('--use_cfg', action='store_true', default=True,
                       help='Use classifier-free guidance')
    parser.add_argument('--cfg_dropout', type=float, default=0.1,
                       help='CFG conditioning dropout probability')
    
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
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='experiments/stage3_mdlm',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every', type=int, default=5,
                       help='Generate samples every N epochs')
    
    args = parser.parse_args()
    main(args)
