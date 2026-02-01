"""
PANDORA Dataset Loader for Big Five Personality Prediction

This module loads the PANDORA subset dataset containing Reddit comments
and Big Five personality trait scores.

Dataset: https://huggingface.co/datasets/Fatima0923/Automated-Personality-Prediction
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np


class PANDORADataset(Dataset):
    """
    PyTorch Dataset for PANDORA personality data.

    Big Five Traits (scored 0-99):
    - Openness (O)
    - Conscientiousness (C)
    - Extraversion (E)
    - Agreeableness (A)
    - Neuroticism (N)

    Args:
        split: One of 'train', 'validation', or 'test'
        tokenizer_name: HuggingFace tokenizer name (default: 'gpt2')
        max_length: Maximum sequence length (default: 512)
        normalize_scores: Whether to normalize Big Five scores to [0, 1] (default: True)
        cache_dir: Directory to cache the dataset
    """

    BIG_FIVE_TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    def __init__(
        self,
        split: str = 'train',
        tokenizer_name: str = 'gpt2',
        max_length: int = 512,
        normalize_scores: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        assert split in ['train', 'validation', 'test'], \
            f"Split must be 'train', 'validation', or 'test', got {split}"

        self.split = split
        self.max_length = max_length
        self.normalize_scores = normalize_scores

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset from HuggingFace
        print(f"Loading PANDORA dataset (split: {split})...")
        self.dataset = load_dataset(
            "Fatima0923/Automated-Personality-Prediction",
            split=split,
            cache_dir=cache_dir
        )

        print(f"Loaded {len(self.dataset)} samples")

        # Compute normalization statistics if needed
        if self.normalize_scores:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute mean and std for Big Five scores (for normalization)."""
        scores = np.array([
            [item[trait] for trait in self.BIG_FIVE_TRAITS]
            for item in self.dataset
        ])

        # Scores are in [0, 99], normalize to [0, 1]
        self.score_min = 0.0
        self.score_max = 99.0

        print(f"Big Five score statistics:")
        for i, trait in enumerate(self.BIG_FIVE_TRAITS):
            print(f"  {trait:20s}: mean={scores[:, i].mean():.2f}, std={scores[:, i].std():.2f}")

    def _normalize_score(self, score: float) -> float:
        """Normalize score from [0, 99] to [0, 1]."""
        return (score - self.score_min) / (self.score_max - self.score_min)

    def _denormalize_score(self, normalized_score: float) -> float:
        """Denormalize score from [0, 1] to [0, 99]."""
        return normalized_score * (self.score_max - self.score_min) + self.score_min

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.

        Returns:
            Dict containing:
                - input_ids: Tokenized text [seq_len]
                - attention_mask: Attention mask [seq_len]
                - personality: Big Five scores [5]
                - text: Original text (str)
        """
        item = self.dataset[idx]

        # Extract text
        text = item['text']

        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Extract Big Five personality scores
        personality_scores = [
            item[trait] for trait in self.BIG_FIVE_TRAITS
        ]

        # Normalize if requested
        if self.normalize_scores:
            personality_scores = [
                self._normalize_score(score) for score in personality_scores
            ]

        personality_tensor = torch.tensor(personality_scores, dtype=torch.float32)

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'personality': personality_tensor,
            'text': text,
        }

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = None,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle (defaults to True for train, False for val/test)
            num_workers: Number of worker processes
            **kwargs: Additional arguments passed to DataLoader

        Returns:
            PyTorch DataLoader
        """
        if shuffle is None:
            shuffle = (self.split == 'train')

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            **kwargs
        )

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Batched dict with stacked tensors
        """
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'personality': torch.stack([item['personality'] for item in batch]),
            'text': [item['text'] for item in batch],
        }

    def get_personality_distribution(self) -> Dict[str, Tuple[float, float]]:
        """
        Get mean and std of each personality trait in the dataset.

        Returns:
            Dict mapping trait name to (mean, std)
        """
        scores = np.array([
            [item[trait] for trait in self.BIG_FIVE_TRAITS]
            for item in self.dataset
        ])

        if self.normalize_scores:
            scores = (scores - self.score_min) / (self.score_max - self.score_min)

        return {
            trait: (scores[:, i].mean(), scores[:, i].std())
            for i, trait in enumerate(self.BIG_FIVE_TRAITS)
        }


def create_dataloaders(
    batch_size: int = 32,
    tokenizer_name: str = 'gpt2',
    max_length: int = 512,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create train, validation, and test dataloaders.

    Args:
        batch_size: Batch size for all dataloaders
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        cache_dir: Cache directory for dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = PANDORADataset(
        split='train',
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        cache_dir=cache_dir,
    )

    val_dataset = PANDORADataset(
        split='validation',
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        cache_dir=cache_dir,
    )

    test_dataset = PANDORADataset(
        split='test',
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        cache_dir=cache_dir,
    )

    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = val_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = test_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """Test the dataset loader."""
    print("Testing PANDORA Dataset Loader...\n")

    # Create dataset
    dataset = PANDORADataset(split='train', max_length=256)

    # Print dataset info
    print(f"\nDataset size: {len(dataset)}")
    print(f"Personality distribution:")
    for trait, (mean, std) in dataset.get_personality_distribution().items():
        print(f"  {trait:20s}: {mean:.3f} ± {std:.3f}")

    # Get a sample
    sample = dataset[0]
    print(f"\nSample data:")
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Personality: {sample['personality']}")
    print(f"  Personality shape: {sample['personality'].shape}")

    # Test dataloader
    dataloader = dataset.get_dataloader(batch_size=4, num_workers=0)
    batch = next(iter(dataloader))
    print(f"\nBatch data:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Personality shape: {batch['personality'].shape}")
    print(f"  Number of texts: {len(batch['text'])}")

    print("\n✓ Dataset loader test passed!")
