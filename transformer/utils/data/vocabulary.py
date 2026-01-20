"""
Vocabulary management for the Transformer.

Provides a simple vocabulary class that wraps the BPE tokenizer
for easier access to vocabulary properties.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .tokenizer import BPETokenizer


class Vocabulary:
    """
    Vocabulary wrapper for the tokenizer.

    Provides convenient access to vocabulary properties and special tokens.

    Args:
        tokenizer: BPE tokenizer instance
    """

    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer

    @property
    def size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    @property
    def pad_idx(self) -> int:
        """Get padding token index."""
        return self.tokenizer.pad_idx

    @property
    def unk_idx(self) -> int:
        """Get unknown token index."""
        return self.tokenizer.unk_idx

    @property
    def sos_idx(self) -> int:
        """Get start-of-sequence token index."""
        return self.tokenizer.sos_idx

    @property
    def eos_idx(self) -> int:
        """Get end-of-sequence token index."""
        return self.tokenizer.eos_idx

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode text to indices.

        Args:
            text: Input text
            add_special_tokens: Whether to add SOS/EOS

        Returns:
            List of token indices
        """
        return self.tokenizer.encode(text, add_special_tokens)

    def decode(
        self,
        indices: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode indices to text.

        Args:
            indices: Token indices
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(indices, skip_special_tokens)

    def token_to_idx(self, token: str) -> int:
        """Get index for a token."""
        return self.tokenizer.token2idx.get(token, self.unk_idx)

    def idx_to_token(self, idx: int) -> str:
        """Get token for an index."""
        return self.tokenizer.idx2token.get(idx, self.tokenizer.UNK_TOKEN)

    def save(self, path: str):
        """Save vocabulary to file."""
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from file."""
        tokenizer = BPETokenizer.load(path)
        return cls(tokenizer)

    @classmethod
    def build(
        cls,
        texts: List[str],
        vocab_size: int = 32000,
        min_freq: int = 2,
        save_path: Optional[str] = None
    ) -> "Vocabulary":
        """
        Build vocabulary from texts.

        Args:
            texts: Training texts
            vocab_size: Target vocabulary size
            min_freq: Minimum token frequency
            save_path: Optional path to save

        Returns:
            Built vocabulary
        """
        tokenizer = BPETokenizer(vocab_size=vocab_size, min_freq=min_freq)
        tokenizer.fit(texts)

        if save_path:
            tokenizer.save(save_path)

        return cls(tokenizer)

    @classmethod
    def build_shared(
        cls,
        src_texts: List[str],
        tgt_texts: List[str],
        vocab_size: int = 32000,
        min_freq: int = 2,
        save_path: Optional[str] = None
    ) -> "Vocabulary":
        """
        Build shared vocabulary from source and target texts.

        Args:
            src_texts: Source language texts
            tgt_texts: Target language texts
            vocab_size: Target vocabulary size
            min_freq: Minimum token frequency
            save_path: Optional path to save

        Returns:
            Shared vocabulary
        """
        all_texts = src_texts + tgt_texts

        tokenizer = BPETokenizer(vocab_size=vocab_size, min_freq=min_freq)
        tokenizer.fit(all_texts)

        if save_path:
            tokenizer.save(save_path)

        return cls(tokenizer)
