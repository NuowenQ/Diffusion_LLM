"""
PANDORA DataLoader for MDLM Integration

Adapts PANDORA dataset to work with MDLM's training infrastructure.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Dict, Optional

from .pandora_dataset import PANDORADataset, create_dataloaders


class MDLMPANDORADataset(PANDORADataset):
    """
    PANDORA dataset adapted for MDLM.
    
    Returns data in format expected by MDLM:
    - 'input_ids': Token IDs
    - 'attention_mask': Attention mask  
    - 'personality': Big Five scores
    - 'labels': Same as input_ids (for language modeling)
    """
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item in MDLM format."""
        item = super().__getitem__(idx)
        
        # Add labels (for MDLM compatibility)
        item['labels'] = item['input_ids'].clone()
        
        return item


def create_mdlm_dataloaders(
    batch_size: int = 32,
    tokenizer_name: str = 'gpt2',
    max_length: int = 1024,
    num_workers: int = 4,
    normalize_scores: bool = True,
) -> tuple:
    """
    Create MDLM-compatible PANDORA dataloaders.

    Args:
        batch_size: Batch size
        tokenizer_name: Name of tokenizer
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        normalize_scores: Whether to normalize personality scores

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Create datasets using the HuggingFace splits
    train_dataset = MDLMPANDORADataset(
        split='train',
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        normalize_scores=normalize_scores,
    )

    val_dataset = MDLMPANDORADataset(
        split='validation',
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        normalize_scores=normalize_scores,
    )

    test_dataset = MDLMPANDORADataset(
        split='test',
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        normalize_scores=normalize_scores,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=PANDORADataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=PANDORADataset.collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=PANDORADataset.collate_fn,
    )

    print(f"MDLM PANDORA Dataloaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """Test MDLM dataloader."""
    print("Testing MDLM PANDORA DataLoader...\n")
    
    train_loader, val_loader, test_loader = create_mdlm_dataloaders(
        batch_size=16,
        max_length=128,
    )
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  personality: {batch['personality'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    
    print(f"\nPersonality range: [{batch['personality'].min():.3f}, {batch['personality'].max():.3f}]")
    
    print("\n✓ MDLM dataloader test passed!")
