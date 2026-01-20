"""
Transformer Encoder.

Implements the Encoder stack from "Attention Is All You Need" (Vaswani et al., 2017).
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer.

    Each layer has two sub-layers:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network

    Each sub-layer is wrapped with residual connection and layer normalization.

    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        d_ff: Dimension of the feed-forward network
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Position-wise Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the encoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            src_mask: Optional mask for padding positions

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-Attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-Forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class Encoder(nn.Module):
    """
    Transformer Encoder stack.

    Composed of N identical encoder layers.

    Args:
        n_layers: Number of encoder layers
        d_model: Dimension of the model
        n_heads: Number of attention heads
        d_ff: Dimension of the feed-forward network
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all encoder layers.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            src_mask: Optional mask for padding positions

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)
