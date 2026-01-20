"""
BPE (Byte-Pair Encoding) Tokenizer.

A pure Python implementation of BPE tokenization without external dependencies.
"""

import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer.

    Implements BPE subword tokenization from scratch.

    Args:
        vocab_size: Target vocabulary size (including special tokens)
        min_freq: Minimum frequency for a token to be included
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

    def __init__(self, vocab_size: int = 32000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq

        # Token to index mapping
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}

        # BPE merge rules (pair -> merged token)
        self.merges: Dict[Tuple[str, str], str] = {}

        # Pre-tokenization regex pattern
        self.pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"
        )

        # Initialize with special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initialize special tokens."""
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.token2idx[token] = i
            self.idx2token[i] = token

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.UNK_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.token2idx[self.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.token2idx)

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of pre-tokenized words
        """
        # Convert to lowercase and find all matches
        text = text.lower()
        tokens = self.pattern.findall(text)
        return tokens

    def _get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        """
        Count word frequencies in corpus.

        Args:
            texts: List of text strings

        Returns:
            Dictionary mapping words to frequencies
        """
        word_freqs = Counter()
        for text in texts:
            words = self._pre_tokenize(text)
            word_freqs.update(words)
        return dict(word_freqs)

    def _get_pair_freqs(
        self, word_freqs: Dict[str, int], splits: Dict[str, List[str]]
    ) -> Dict[Tuple[str, str], int]:
        """
        Count pair frequencies.

        Args:
            word_freqs: Word frequencies
            splits: Current word splits

        Returns:
            Dictionary mapping pairs to frequencies
        """
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) < 2:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(
        self,
        pair: Tuple[str, str],
        splits: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Merge a pair of tokens in all words.

        Args:
            pair: Token pair to merge
            splits: Current word splits

        Returns:
            Updated word splits
        """
        new_splits = {}
        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    new_split.append(split[i] + split[i + 1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        return new_splits

    def fit(self, texts: List[str], verbose: bool = True):
        """
        Learn BPE vocabulary from texts.

        Args:
            texts: List of training texts
            verbose: Whether to print progress
        """
        if verbose:
            print(f"Learning BPE with target vocab size: {self.vocab_size}")

        # Get word frequencies
        word_freqs = self._get_word_freqs(texts)
        if verbose:
            print(f"Found {len(word_freqs)} unique words")

        # Filter by minimum frequency
        word_freqs = {w: f for w, f in word_freqs.items() if f >= self.min_freq}
        if verbose:
            print(f"After filtering (min_freq={self.min_freq}): {len(word_freqs)} words")

        # Initialize splits with characters
        splits = {}
        for word in word_freqs:
            # Add word boundary marker
            splits[word] = list(word) + ["</w>"]

        # Get initial vocabulary (all characters)
        vocab = set()
        for word in word_freqs:
            for char in word:
                vocab.add(char)
        vocab.add("</w>")

        # Add initial vocab to token2idx
        for token in sorted(vocab):
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

        # Learn merges
        num_merges = self.vocab_size - len(self.token2idx)
        if verbose:
            print(f"Learning {num_merges} merges...")

        for i in range(num_merges):
            # Get pair frequencies
            pair_freqs = self._get_pair_freqs(word_freqs, splits)
            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] < self.min_freq:
                break

            # Merge pair
            splits = self._merge_pair(best_pair, splits)

            # Add to vocabulary
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[new_token] = idx
                self.idx2token[idx] = new_token

            # Record merge
            self.merges[best_pair] = new_token

            if verbose and (i + 1) % 1000 == 0:
                print(f"  Learned {i + 1} merges, vocab size: {len(self.token2idx)}")

        if verbose:
            print(f"Final vocabulary size: {len(self.token2idx)}")

    def _apply_bpe(self, token: str) -> List[str]:
        """
        Apply BPE to a single token.

        Args:
            token: Input token

        Returns:
            List of BPE subtokens
        """
        word = list(token) + ["</w>"]

        while len(word) > 1:
            # Find the pair with the highest priority (earliest in merges)
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]

            # Find which pairs are in our merge rules
            mergeable = [(p, self.merges[p]) for p in pairs if p in self.merges]

            if not mergeable:
                break

            # Apply the first merge found
            # (In practice, should use merge priority, but this is simpler)
            pair_to_merge = mergeable[0][0]

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair_to_merge:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        return word

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode text to token indices.

        Args:
            text: Input text
            add_special_tokens: Whether to add SOS and EOS tokens

        Returns:
            List of token indices
        """
        # Pre-tokenize
        tokens = self._pre_tokenize(text)

        # Apply BPE to each token
        bpe_tokens = []
        for token in tokens:
            subtokens = self._apply_bpe(token)
            bpe_tokens.extend(subtokens)

        # Convert to indices
        indices = []
        for token in bpe_tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                indices.append(self.unk_idx)

        # Add special tokens
        if add_special_tokens:
            indices = [self.sos_idx] + indices + [self.eos_idx]

        return indices

    def decode(
        self,
        indices: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token indices to text.

        Args:
            indices: List of token indices
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        tokens = []
        for idx in indices:
            if idx in self.idx2token:
                token = self.idx2token[idx]
                if skip_special_tokens and token in self.SPECIAL_TOKENS:
                    continue
                tokens.append(token)

        # Join and clean up
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        text = text.strip()

        return text

    def save(self, path: str):
        """
        Save tokenizer to file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "token2idx": self.token2idx,
            "merges": {f"{a}|||{b}": c for (a, b), c in self.merges.items()}
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load tokenizer from file.

        Args:
            path: Path to saved tokenizer

        Returns:
            Loaded tokenizer
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            min_freq=data["min_freq"]
        )

        tokenizer.token2idx = data["token2idx"]
        tokenizer.idx2token = {int(v): k for k, v in data["token2idx"].items()}
        tokenizer.merges = {
            tuple(k.split("|||")): v
            for k, v in data["merges"].items()
        }

        return tokenizer


def build_shared_tokenizer(
    src_texts: List[str],
    tgt_texts: List[str],
    vocab_size: int = 32000,
    min_freq: int = 2,
    save_path: Optional[str] = None
) -> BPETokenizer:
    """
    Build a shared BPE tokenizer from source and target texts.

    Args:
        src_texts: Source language texts
        tgt_texts: Target language texts
        vocab_size: Target vocabulary size
        min_freq: Minimum frequency
        save_path: Optional path to save the tokenizer

    Returns:
        Trained BPE tokenizer
    """
    # Combine source and target texts
    all_texts = src_texts + tgt_texts

    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=vocab_size, min_freq=min_freq)
    tokenizer.fit(all_texts)

    # Save if path provided
    if save_path:
        tokenizer.save(save_path)

    return tokenizer
