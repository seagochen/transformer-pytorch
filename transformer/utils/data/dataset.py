"""
IWSLT 2016 Dataset and DataLoader.

Provides PyTorch Dataset and DataLoader for the IWSLT 2016 De-En translation task.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .download import download_iwslt2016, load_iwslt_split
from .vocabulary import Vocabulary
from .tokenizer import BPETokenizer


class IWSLT2016Dataset(Dataset):
    """
    IWSLT 2016 German-English Translation Dataset.

    Args:
        src_texts: Source (German) texts
        tgt_texts: Target (English) texts
        vocab: Shared vocabulary
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        src_texts: List[str],
        tgt_texts: List[str],
        vocab: Vocabulary,
        max_seq_len: int = 256
    ):
        assert len(src_texts) == len(tgt_texts), \
            f"Source and target must have same length: {len(src_texts)} vs {len(tgt_texts)}"

        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - src: Source token indices
                - tgt: Target token indices (with SOS)
                - tgt_y: Target labels (with EOS, for loss computation)
        """
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Encode texts
        src_indices = self.vocab.encode(src_text, add_special_tokens=True)
        tgt_indices = self.vocab.encode(tgt_text, add_special_tokens=True)

        # Truncate if necessary
        if len(src_indices) > self.max_seq_len:
            src_indices = src_indices[:self.max_seq_len - 1] + [self.vocab.eos_idx]
        if len(tgt_indices) > self.max_seq_len:
            tgt_indices = tgt_indices[:self.max_seq_len - 1] + [self.vocab.eos_idx]

        # Create tensors
        src = torch.tensor(src_indices, dtype=torch.long)

        # tgt: input to decoder (SOS + tokens, without final EOS)
        # tgt_y: target labels (tokens + EOS, without initial SOS)
        tgt = torch.tensor(tgt_indices[:-1], dtype=torch.long)
        tgt_y = torch.tensor(tgt_indices[1:], dtype=torch.long)

        return {
            "src": src,
            "tgt": tgt,
            "tgt_y": tgt_y
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_idx: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads sequences to the same length within a batch.

    Args:
        batch: List of samples from dataset
        pad_idx: Padding token index

    Returns:
        Batched and padded tensors
    """
    src_list = [item["src"] for item in batch]
    tgt_list = [item["tgt"] for item in batch]
    tgt_y_list = [item["tgt_y"] for item in batch]

    # Pad sequences
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    tgt_y_padded = pad_sequence(tgt_y_list, batch_first=True, padding_value=pad_idx)

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "tgt_y": tgt_y_padded
    }


def create_dataloaders(
    data_dir: str = "./data",
    vocab_path: Optional[str] = None,
    vocab_size: int = 32000,
    min_freq: int = 2,
    max_seq_len: int = 256,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """
    Create train, valid, and test dataloaders.

    Args:
        data_dir: Directory containing the dataset
        vocab_path: Path to saved vocabulary (will build if not exists)
        vocab_size: Vocabulary size (if building new)
        min_freq: Minimum frequency (if building new)
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, valid_loader, test_loader, vocabulary)
    """
    data_dir = Path(data_dir)
    vocab_path = vocab_path or str(data_dir / "vocab.json")

    # Download dataset if needed
    print("Checking/downloading dataset...")
    extract_dir = download_iwslt2016(str(data_dir))

    # Load data splits
    print("Loading data splits...")
    train_src, train_tgt = load_iwslt_split(extract_dir, "train")
    valid_src, valid_tgt = load_iwslt_split(extract_dir, "valid")
    test_src, test_tgt = load_iwslt_split(extract_dir, "test")

    print(f"Train: {len(train_src)} pairs")
    print(f"Valid: {len(valid_src)} pairs")
    print(f"Test: {len(test_src)} pairs")

    # Build or load vocabulary
    if Path(vocab_path).exists():
        print(f"Loading vocabulary from {vocab_path}")
        vocab = Vocabulary.load(vocab_path)
    else:
        print(f"Building vocabulary (size={vocab_size}, min_freq={min_freq})...")
        vocab = Vocabulary.build_shared(
            train_src, train_tgt,
            vocab_size=vocab_size,
            min_freq=min_freq,
            save_path=vocab_path
        )

    print(f"Vocabulary size: {vocab.size}")

    # Create datasets
    train_dataset = IWSLT2016Dataset(train_src, train_tgt, vocab, max_seq_len)
    valid_dataset = IWSLT2016Dataset(valid_src, valid_tgt, vocab, max_seq_len)
    test_dataset = IWSLT2016Dataset(test_src, test_tgt, vocab, max_seq_len)

    # Create collate function with correct padding index
    def collate(batch):
        return collate_fn(batch, vocab.pad_idx)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader, vocab
