"""
Download the PANDORA dataset from HuggingFace.

This script downloads the Automated-Personality-Prediction dataset (PANDORA subset)
from HuggingFace for local caching and preprocessing.

Dataset: https://huggingface.co/datasets/Fatima0923/Automated-Personality-Prediction
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset


def download_dataset(
    dataset_name: str = "Fatima0923/Automated-Personality-Prediction",
    cache_dir: str = None,
    splits: list = None,
):
    """
    Download the PANDORA dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset name/path
        cache_dir: Directory to cache the dataset (default: ~/.cache/huggingface/datasets)
        splits: List of splits to download (default: ['train', 'validation', 'test'])
    """
    if splits is None:
        splits = ['train', 'validation', 'test']

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")

    print(f"Downloading PANDORA dataset from HuggingFace...")
    print(f"Dataset: {dataset_name}")
    print(f"Cache directory: {cache_dir}")
    print(f"Splits: {splits}\n")

    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Download each split
    datasets = {}
    for split in splits:
        print(f"Downloading {split} split...")
        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
            )
            datasets[split] = dataset
            print(f"  ✓ Downloaded {len(dataset)} samples\n")
        except Exception as e:
            print(f"  ✗ Error downloading {split} split: {e}\n")
            raise

    # Print dataset info
    print("Dataset Summary:")
    print(f"  Dataset name: {dataset_name}")
    for split, dataset in datasets.items():
        print(f"  {split.capitalize():12s}: {len(dataset):6d} samples")

    # Print sample schema
    if datasets:
        first_split = next(iter(datasets.values()))
        print(f"\nDataset Schema:")
        print(f"  Columns: {first_split.column_names}")
        
        # Show first sample
        sample = first_split[0]
        print(f"\nFirst Sample:")
        for key, value in sample.items():
            if isinstance(value, str):
                print(f"  {key:20s}: {value[:100]}..." if len(value) > 100 else f"  {key:20s}: {value}")
            else:
                print(f"  {key:20s}: {value}")

    print("\n✓ Dataset download completed successfully!")
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download PANDORA dataset from HuggingFace"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for dataset (default: ~/.cache/huggingface/datasets)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to download (default: train validation test)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Fatima0923/Automated-Personality-Prediction",
        help="HuggingFace dataset name (default: Fatima0923/Automated-Personality-Prediction)",
    )

    args = parser.parse_args()

    download_dataset(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        splits=args.splits,
    )
