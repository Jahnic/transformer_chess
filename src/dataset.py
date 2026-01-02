"""
PyTorch Dataset and DataLoader for chess move sequences.

Character-level tokenization for training a transformer language model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Optional, Tuple


class ChessDataset(Dataset):
    """
    Character-level dataset for chess move sequences.
    
    Each sample is a (context, target) pair where:
    - context: sequence of `block_size` characters
    - target: the same sequence shifted by 1 (next-char prediction)
    """
    
    def __init__(
        self, 
        data_path: str, 
        vocab_path: str, 
        block_size: int = 128
    ):
        """
        Args:
            data_path: Path to text file (train.txt or val.txt)
            vocab_path: Path to vocab.json
            block_size: Context window size (number of characters)
        """
        self.block_size = block_size
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        
        # Create reverse mapping for decoding
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
        
        # Load text data
        with open(data_path, 'r') as f:
            self.text = f.read()
        
        # Encode entire text
        self.data = self.encode(self.text)
        
        print(f"Loaded {len(self.text):,} characters")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Block size: {self.block_size}")
        print(f"Number of training examples: {len(self):,}")
    
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to tensor of token indices."""
        # Handle unknown characters by mapping to a space
        unk_idx = self.vocab.get(' ', 0)
        indices = [self.vocab.get(c, unk_idx) for c in text]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """Convert tensor of indices back to text."""
        return ''.join(self.idx_to_char.get(i.item(), '?') for i in indices)
    
    def __len__(self) -> int:
        # Number of valid starting positions
        return max(0, len(self.data) - self.block_size - 1)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Returns:
            x: Input sequence of length block_size
            y: Target sequence of length block_size (shifted by 1)
        """
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def get_dataloaders(
    data_dir: str,
    block_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory containing train.txt, val.txt, vocab.json
        block_size: Context window size
        batch_size: Batch size for training
        num_workers: Number of data loading workers
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        vocab: Vocabulary dictionary
    """
    data_dir = Path(data_dir)
    
    train_dataset = ChessDataset(
        data_path=str(data_dir / 'train.txt'),
        vocab_path=str(data_dir / 'vocab.json'),
        block_size=block_size
    )
    
    val_dataset = ChessDataset(
        data_path=str(data_dir / 'val.txt'),
        vocab_path=str(data_dir / 'vocab.json'),
        block_size=block_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.vocab


def inspect_batch(dataloader: DataLoader, dataset: ChessDataset) -> None:
    """
    Inspect a single batch for debugging.
    """
    x, y = next(iter(dataloader))
    
    print(f"Batch shape: x={x.shape}, y={y.shape}")
    print(f"Data type: {x.dtype}")
    print()
    
    # Show first example
    print("First example in batch:")
    print(f"  Input:  '{dataset.decode(x[0])}'")
    print(f"  Target: '{dataset.decode(y[0])}'")
    print()
    
    # Verify alignment (target should be input shifted by 1)
    print("Verifying alignment (first 20 chars):")
    print(f"  Input[1:20]:  '{dataset.decode(x[0][1:20])}'")
    print(f"  Target[0:19]: '{dataset.decode(y[0][0:19])}'")


# Quick test
if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'processed'
    
    if not (data_dir / 'train.txt').exists():
        print("No processed data found. Run data.py first.")
        exit(1)
    
    print("=== Testing DataLoader ===\n")
    
    # Create datasets
    train_dataset = ChessDataset(
        data_path=str(data_dir / 'train.txt'),
        vocab_path=str(data_dir / 'vocab.json'),
        block_size=128
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    print("\n=== Batch Inspection ===\n")
    inspect_batch(train_loader, train_dataset)
