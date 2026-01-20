"""
Embedding layers for the Transformer.

Implements Token Embedding and Positional Encoding
from "Attention Is All You Need" (Vaswani et al., 2017).
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Dimension of the model
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the division term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but should be saved/loaded)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Combined Token Embedding and Positional Encoding.

    Scales the token embedding by sqrt(d_model) as in the original paper,
    then adds positional encoding.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        padding_idx: Index of the padding token
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Scaling factor
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens and add positional encoding.

        Args:
            x: Input token indices of shape (batch_size, seq_len)

        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model)
        x = self.token_embedding(x) * self.scale

        # Add positional encoding
        x = self.positional_encoding(x)

        return x
