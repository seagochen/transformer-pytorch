"""
Complete Transformer Model for Sequence-to-Sequence tasks.

Implements the full Transformer architecture from
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import torch
import torch.nn as nn
from typing import Optional

from .encoder import Encoder
from .decoder import Decoder
from .embedding import TransformerEmbedding


class Transformer(nn.Module):
    """
    Transformer Model for Sequence-to-Sequence tasks.

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Dimension of the model (default: 512)
        n_heads: Number of attention heads (default: 8)
        n_encoder_layers: Number of encoder layers (default: 6)
        n_decoder_layers: Number of decoder layers (default: 6)
        d_ff: Dimension of feed-forward network (default: 2048)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length (default: 256)
        padding_idx: Index of padding token (default: 0)
        share_embedding: Whether to share embedding between encoder and decoder (default: False)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        padding_idx: int = 0,
        share_embedding: bool = False
    ):
        super().__init__()

        self.padding_idx = padding_idx
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Source embedding
        self.src_embedding = TransformerEmbedding(
            src_vocab_size, d_model, max_seq_len, dropout, padding_idx
        )

        # Target embedding (shared with source if share_embedding is True)
        if share_embedding and src_vocab_size == tgt_vocab_size:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = TransformerEmbedding(
                tgt_vocab_size, d_model, max_seq_len, dropout, padding_idx
            )

        # Encoder
        self.encoder = Encoder(
            n_encoder_layers, d_model, n_heads, d_ff, dropout
        )

        # Decoder
        self.decoder = Decoder(
            n_decoder_layers, d_model, n_heads, d_ff, dropout
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create mask for source sequence (padding mask).

        Args:
            src: Source tensor of shape (batch_size, src_seq_len)

        Returns:
            Mask tensor of shape (batch_size, 1, 1, src_seq_len)
        """
        # 1 for padding positions, 0 for valid positions
        src_mask = (src == self.padding_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def create_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create mask for target sequence (causal mask + padding mask).

        Args:
            tgt: Target tensor of shape (batch_size, tgt_seq_len)

        Returns:
            Mask tensor of shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        batch_size, tgt_seq_len = tgt.size()
        device = tgt.device

        # Padding mask: (batch_size, 1, 1, tgt_seq_len)
        padding_mask = (tgt == self.padding_idx).unsqueeze(1).unsqueeze(2)

        # Causal mask (subsequent mask): (1, 1, tgt_seq_len, tgt_seq_len)
        causal_mask = torch.triu(
            torch.ones(tgt_seq_len, tgt_seq_len, device=device),
            diagonal=1
        ).bool().unsqueeze(0).unsqueeze(0)

        # Combine masks
        tgt_mask = padding_mask | causal_mask

        return tgt_mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer.

        Args:
            src: Source sequence of shape (batch_size, src_seq_len)
            tgt: Target sequence of shape (batch_size, tgt_seq_len)
            src_mask: Optional source mask
            tgt_mask: Optional target mask

        Returns:
            Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)

        # Embed source and target
        src_embedded = self.src_embedding(src)
        tgt_embedded = self.tgt_embedding(tgt)

        # Encode source
        encoder_output = self.encoder(src_embedded, src_mask)

        # Decode
        decoder_output = self.decoder(
            tgt_embedded, encoder_output, tgt_mask, src_mask
        )

        # Project to vocabulary
        output = self.output_projection(decoder_output)

        return output

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence (for inference).

        Args:
            src: Source sequence of shape (batch_size, src_seq_len)
            src_mask: Optional source mask

        Returns:
            Encoder output of shape (batch_size, src_seq_len, d_model)
        """
        if src_mask is None:
            src_mask = self.create_src_mask(src)

        src_embedded = self.src_embedding(src)
        return self.encoder(src_embedded, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence (for inference).

        Args:
            tgt: Target sequence of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output
            src_mask: Optional source mask
            tgt_mask: Optional target mask

        Returns:
            Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)

        tgt_embedded = self.tgt_embedding(tgt)
        decoder_output = self.decoder(
            tgt_embedded, encoder_output, tgt_mask, src_mask
        )
        return self.output_projection(decoder_output)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: int,
        sos_idx: int,
        eos_idx: int
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.

        Args:
            src: Source sequence of shape (batch_size, src_seq_len)
            max_len: Maximum decoding length
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index

        Returns:
            Decoded sequences of shape (batch_size, decoded_len)
        """
        batch_size = src.size(0)
        device = src.device

        # Encode source
        src_mask = self.create_src_mask(src)
        encoder_output = self.encode(src, src_mask)

        # Initialize target with SOS token
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

        # Track which sequences are done
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Decode
            output = self.decode(tgt, encoder_output, src_mask)

            # Get next token (greedy)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

            # Append to target
            tgt = torch.cat([tgt, next_token], dim=1)

            # Check for EOS
            done = done | (next_token.squeeze(-1) == eos_idx)
            if done.all():
                break

        return tgt

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        max_len: int,
        sos_idx: int,
        eos_idx: int,
        beam_size: int = 5
    ) -> torch.Tensor:
        """
        Beam search decoding for inference.

        Args:
            src: Source sequence of shape (1, src_seq_len) - single sample
            max_len: Maximum decoding length
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            beam_size: Beam size

        Returns:
            Best decoded sequence of shape (1, decoded_len)
        """
        assert src.size(0) == 1, "Beam search only supports batch size 1"

        device = src.device

        # Encode source
        src_mask = self.create_src_mask(src)
        encoder_output = self.encode(src, src_mask)

        # Expand for beam search
        encoder_output = encoder_output.repeat(beam_size, 1, 1)
        src_mask = src_mask.repeat(beam_size, 1, 1, 1)

        # Initialize beams: (beam_size, 1)
        beams = torch.full((beam_size, 1), sos_idx, dtype=torch.long, device=device)

        # Beam scores
        scores = torch.zeros(beam_size, device=device)
        scores[1:] = float('-inf')  # Only first beam is active initially

        # Completed sequences
        completed = []

        for step in range(max_len - 1):
            # Decode current beams
            output = self.decode(beams, encoder_output, src_mask)
            log_probs = torch.log_softmax(output[:, -1, :], dim=-1)

            # Calculate scores for all possible next tokens
            vocab_size = log_probs.size(-1)
            next_scores = scores.unsqueeze(1) + log_probs  # (beam_size, vocab_size)

            # Flatten and get top-k
            next_scores = next_scores.view(-1)  # (beam_size * vocab_size,)
            top_scores, top_indices = next_scores.topk(beam_size, dim=0)

            # Convert to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update beams
            beams = torch.cat([
                beams[beam_indices],
                token_indices.unsqueeze(1)
            ], dim=1)
            scores = top_scores

            # Check for completed sequences (EOS)
            eos_mask = token_indices == eos_idx
            for i in range(beam_size):
                if eos_mask[i]:
                    completed.append((scores[i].item(), beams[i].clone()))

            # Remove completed beams
            if eos_mask.any():
                keep_mask = ~eos_mask
                if keep_mask.any():
                    beams = beams[keep_mask]
                    scores = scores[keep_mask]
                    encoder_output = encoder_output[keep_mask]
                    src_mask = src_mask[keep_mask]

                    # Pad back to beam_size
                    n_keep = keep_mask.sum().item()
                    if n_keep < beam_size:
                        pad_size = beam_size - n_keep
                        beams = torch.cat([beams, beams[:pad_size]], dim=0)
                        scores = torch.cat([scores, torch.full((pad_size,), float('-inf'), device=device)])
                        encoder_output = torch.cat([encoder_output, encoder_output[:pad_size]], dim=0)
                        src_mask = torch.cat([src_mask, src_mask[:pad_size]], dim=0)
                else:
                    break

        # Add remaining beams to completed
        for i in range(beams.size(0)):
            completed.append((scores[i].item(), beams[i].clone()))

        # Return best sequence
        if completed:
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1].unsqueeze(0)
        else:
            return beams[0].unsqueeze(0)
