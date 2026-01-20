"""
Position-wise Feed-Forward Network.

Implements the feed-forward network used in each Transformer layer
from "Attention Is All You Need" (Vaswani et al., 2017).
"""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2

    This is applied to each position separately and identically.
    It consists of two linear transformations with a ReLU activation in between.

    Args:
        d_model: Dimension of the model (input and output dimension)
        d_ff: Dimension of the inner feed-forward layer (typically 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear layer + ReLU + Dropout
        x = self.dropout(torch.relu(self.linear1(x)))
        # Second linear layer
        x = self.linear2(x)

        return x
