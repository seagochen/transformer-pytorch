"""
Transformer Decoder.

Implements the Decoder stack from "Attention Is All You Need" (Vaswani et al., 2017).
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer.

    Each layer has three sub-layers:
    1. Masked Multi-Head Self-Attention
    2. Multi-Head Cross-Attention (attending to encoder output)
    3. Position-wise Feed-Forward Network

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

        # Masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Multi-Head Cross-Attention
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Position-wise Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the decoder layer.

        Args:
            x: Target input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Mask for target self-attention (causal + padding)
            src_mask: Mask for source in cross-attention (padding)

        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked Self-Attention with residual connection and layer norm
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Cross-Attention with residual connection and layer norm
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Feed-Forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class Decoder(nn.Module):
    """
    Transformer Decoder stack.

    Composed of N identical decoder layers.

    Args:
        n_layers: Number of decoder layers
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
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all decoder layers.

        Args:
            x: Target input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Mask for target self-attention
            src_mask: Mask for source in cross-attention

        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)

        return self.norm(x)
