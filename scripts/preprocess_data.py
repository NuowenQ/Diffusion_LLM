"""
Preprocess the PANDORA dataset for training.

This script prepares the downloaded PANDORA dataset for use in training by:
1. Validating data integrity
2. Computing dataset statistics
3. Creating train/val/test splits if needed
4. Tokenizing and caching preprocessed data
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class PANDORAPreprocessor:
    """Preprocess PANDORA dataset for training."""

    BIG_FIVE_TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    def __init__(
        self,
        dataset_name: str = "Fatima0923/Automated-Personality-Prediction",
        cache_dir: str = None,
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
    ):
        """
        Initialize preprocessor.

        Args:
            dataset_name: HuggingFace dataset name
            cache_dir: Cache directory for dataset
            tokenizer_name: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/datasets")
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length

        # Load tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load datasets
        self.datasets = {}
        self.stats = {}

    def load_datasets(self, splits: List[str] = None):
        """Load datasets from HuggingFace."""
        if splits is None:
            splits = ['train', 'validation', 'test']

        print(f"\nLoading datasets from {self.dataset_name}...")
        for split in splits:
            try:
                dataset = load_dataset(
                    self.dataset_name,
                    split=split,
                    cache_dir=self.cache_dir,
                )
                self.datasets[split] = dataset
                print(f"  ✓ Loaded {split:12s}: {len(dataset):6d} samples")
            except Exception as e:
                print(f"  ✗ Error loading {split}: {e}")
                raise

    def validate_data(self) -> bool:
        """Validate dataset integrity and structure."""
        print("\nValidating datasets...")

        for split, dataset in self.datasets.items():
            print(f"\n  {split.upper()}:")
            
            # Check required fields
            required_fields = ['text'] + self.BIG_FIVE_TRAITS
            missing_fields = [f for f in required_fields if f not in dataset.column_names]
            
            if missing_fields:
                print(f"    ✗ Missing fields: {missing_fields}")
                return False
            else:
                print(f"    ✓ All required fields present")

            # Check data types and value ranges
            sample = dataset[0]
            
            # Check text
            if not isinstance(sample['text'], str) or len(sample['text']) == 0:
                print(f"    ✗ Invalid text field")
                return False
            
            # Check personality scores
            for trait in self.BIG_FIVE_TRAITS:
                value = sample[trait]
                if not isinstance(value, (int, float)):
                    print(f"    ✗ Invalid type for {trait}: {type(value)}")
                    return False
                if not (0 <= value <= 99):
                    print(f"    ✗ {trait} score out of range [0-99]: {value}")
                    return False
            
            print(f"    ✓ Data types and value ranges valid")

        print("\n✓ All datasets validated successfully!")
        return True

    def compute_statistics(self):
        """Compute and print dataset statistics."""
        print("\nComputing dataset statistics...")

        for split, dataset in self.datasets.items():
            print(f"\n  {split.upper()}:")
            
            # Text statistics
            text_lengths = [len(item['text'].split()) for item in tqdm(dataset, desc=f"    Processing {split}", leave=False)]
            print(f"    Text length (words):")
            print(f"      Mean: {np.mean(text_lengths):.1f}")
            print(f"      Std:  {np.std(text_lengths):.1f}")
            print(f"      Min:  {np.min(text_lengths)}")
            print(f"      Max:  {np.max(text_lengths)}")

            # Personality statistics
            personality_scores = {trait: [] for trait in self.BIG_FIVE_TRAITS}
            for item in dataset:
                for trait in self.BIG_FIVE_TRAITS:
                    personality_scores[trait].append(item[trait])

            print(f"    Personality traits (score range: 0-99):")
            for trait in self.BIG_FIVE_TRAITS:
                scores = np.array(personality_scores[trait])
                print(f"      {trait:18s}: mean={np.mean(scores):6.2f}, std={np.std(scores):6.2f}")

            self.stats[split] = {
                'num_samples': len(dataset),
                'text_length_mean': float(np.mean(text_lengths)),
                'text_length_std': float(np.std(text_lengths)),
                'personality_stats': {
                    trait: {
                        'mean': float(np.mean(personality_scores[trait])),
                        'std': float(np.std(personality_scores[trait])),
                    }
                    for trait in self.BIG_FIVE_TRAITS
                },
            }

    def preprocess_texts(self):
        """Preprocess and tokenize texts."""
        print("\nPreprocessing texts...")

        for split, dataset in self.datasets.items():
            print(f"  Tokenizing {split} split...")
            
            def tokenize_function(examples):
                """Tokenize texts."""
                return self.tokenizer(
                    examples['text'],
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors=None,
                )

            # Apply tokenization
            dataset_tokenized = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=100,
                remove_columns=['text'],
                desc=f"    Tokenizing {split}",
            )

            self.datasets[split] = dataset_tokenized
            print(f"    ✓ Tokenized {len(dataset_tokenized)} samples")

    def save_statistics(self, output_dir: str = "./data"):
        """Save dataset statistics to file."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        stats_file = os.path.join(output_dir, "dataset_stats.json")
        
        print(f"\nSaving statistics to {stats_file}...")
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✓ Statistics saved")

    def run(self, output_dir: str = "./data"):
        """Run full preprocessing pipeline."""
        print("=" * 60)
        print("PANDORA Dataset Preprocessing Pipeline")
        print("=" * 60)

        # Load datasets
        self.load_datasets()

        # Validate
        if not self.validate_data():
            raise ValueError("Dataset validation failed!")

        # Compute statistics
        self.compute_statistics()

        # Preprocess texts
        self.preprocess_texts()

        # Save statistics
        self.save_statistics(output_dir)

        print("\n" + "=" * 60)
        print("✓ Preprocessing completed successfully!")
        print("=" * 60)

        return self.datasets


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PANDORA dataset for training"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Fatima0923/Automated-Personality-Prediction",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for dataset",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name (default: gpt2)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for preprocessed data and statistics",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to preprocess",
    )

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = PANDORAPreprocessor(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
    )

    # Load datasets
    preprocessor.load_datasets(splits=args.splits)

    # Run preprocessing
    preprocessor.run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
