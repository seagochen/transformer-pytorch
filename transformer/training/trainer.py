"""
Transformer Trainer.

Training loop with label smoothing, gradient clipping, and BLEU evaluation.
"""

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from tqdm import tqdm

from ..models import Transformer
from ..utils.callbacks import TransformerLRScheduler
from ..utils.data import Vocabulary


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.

    Implements label smoothing as described in the Transformer paper.
    Smooths the target distribution by mixing it with a uniform distribution.

    Args:
        vocab_size: Size of the vocabulary
        padding_idx: Index of the padding token (ignored in loss)
        smoothing: Smoothing factor (0.0 = no smoothing, 1.0 = uniform)
    """

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        smoothing: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            logits: Model output logits of shape (batch_size, seq_len, vocab_size)
            target: Target indices of shape (batch_size, seq_len)

        Returns:
            Scalar loss tensor
        """
        # Reshape for easier computation
        logits = logits.contiguous().view(-1, self.vocab_size)
        target = target.contiguous().view(-1)

        # Create smoothed target distribution
        log_probs = torch.log_softmax(logits, dim=-1)

        # One-hot with smoothing
        smooth_target = torch.zeros_like(log_probs)
        smooth_target.fill_(self.smoothing / (self.vocab_size - 2))  # -2 for target and pad
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_target[:, self.padding_idx] = 0

        # Create mask for padding positions
        padding_mask = (target == self.padding_idx)
        smooth_target[padding_mask] = 0

        # Compute loss
        loss = -(smooth_target * log_probs).sum(dim=-1)

        # Average over non-padding tokens
        non_padding = (~padding_mask).sum()
        loss = loss.sum() / non_padding

        return loss


def compute_bleu(
    references: List[List[str]],
    hypotheses: List[List[str]],
    max_n: int = 4
) -> float:
    """
    Compute BLEU score.

    A simple implementation of BLEU without external dependencies.

    Args:
        references: List of reference token lists
        hypotheses: List of hypothesis token lists
        max_n: Maximum n-gram order

    Returns:
        BLEU score
    """
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        """Get n-gram counts."""
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def brevity_penalty(ref_len: int, hyp_len: int) -> float:
        """Compute brevity penalty."""
        if hyp_len >= ref_len:
            return 1.0
        return math.exp(1 - ref_len / hyp_len)

    # Compute n-gram precisions
    precisions = []
    total_ref_len = 0
    total_hyp_len = 0

    for n in range(1, max_n + 1):
        total_matches = 0
        total_count = 0

        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = get_ngrams(ref, n)
            hyp_ngrams = get_ngrams(hyp, n)

            # Count matches (clipped by reference count)
            for ngram, count in hyp_ngrams.items():
                total_matches += min(count, ref_ngrams.get(ngram, 0))

            total_count += max(len(hyp) - n + 1, 0)

            if n == 1:
                total_ref_len += len(ref)
                total_hyp_len += len(hyp)

        if total_count > 0:
            precisions.append(total_matches / total_count)
        else:
            precisions.append(0)

    # Compute BLEU
    if min(precisions) > 0:
        log_precision = sum(math.log(p) for p in precisions) / max_n
        bp = brevity_penalty(total_ref_len, total_hyp_len)
        bleu = bp * math.exp(log_precision)
    else:
        bleu = 0.0

    return bleu * 100  # Return as percentage


class Trainer:
    """
    Transformer Trainer.

    Handles training loop, validation, checkpointing, and evaluation.

    Args:
        model: Transformer model
        vocab: Vocabulary
        device: Training device
        learning_rate: Base learning rate (used with warmup)
        warmup_steps: Number of warmup steps
        label_smoothing: Label smoothing factor
        gradient_clip: Gradient clipping value
    """

    def __init__(
        self,
        model: Transformer,
        vocab: Vocabulary,
        device: str = "cuda",
        learning_rate: float = 0.0001,
        warmup_steps: int = 4000,
        label_smoothing: float = 0.1,
        gradient_clip: float = 1.0
    ):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = torch.device(device)
        self.gradient_clip = gradient_clip

        # Loss function with label smoothing
        self.criterion = LabelSmoothingLoss(
            vocab_size=vocab.size,
            padding_idx=vocab.pad_idx,
            smoothing=label_smoothing
        )

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        # Learning rate scheduler
        self.scheduler = TransformerLRScheduler(
            self.optimizer,
            d_model=model.d_model,
            warmup_steps=warmup_steps
        )

        # Training state
        self.global_step = 0
        self.best_bleu = 0.0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.

        Args:
            batch: Batch dictionary with src, tgt, tgt_y

        Returns:
            Loss value
        """
        self.model.train()

        src = batch["src"].to(self.device)
        tgt = batch["tgt"].to(self.device)
        tgt_y = batch["tgt_y"].to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(src, tgt)

        # Compute loss
        loss = self.criterion(output, tgt_y)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )

        # Update parameters
        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        return loss.item()

    @torch.no_grad()
    def validate(self, dataloader, n_samples: int = 1000) -> Tuple[float, float]:
        """
        Validate on validation set.

        Args:
            dataloader: Validation dataloader
            n_samples: Number of random samples to use for validation

        Returns:
            Tuple of (average loss, BLEU score)
        """
        self.model.eval()
        dataset = dataloader.dataset

        # Randomly sample indices
        all_indices = list(range(len(dataset)))
        sample_indices = random.sample(all_indices, min(n_samples, len(dataset)))

        total_loss = 0
        total_tokens = 0
        all_references = []
        all_hypotheses = []

        for idx in tqdm(sample_indices, desc="Validating", leave=False):
            sample = dataset[idx]
            src = sample["src"].unsqueeze(0).to(self.device)
            tgt = sample["tgt"].unsqueeze(0).to(self.device)
            tgt_y = sample["tgt_y"].unsqueeze(0).to(self.device)

            # Forward pass for loss
            output = self.model(src, tgt)
            loss = self.criterion(output, tgt_y)

            # Count non-padding tokens
            non_padding = (tgt_y != self.vocab.pad_idx).sum().item()
            total_loss += loss.item() * non_padding
            total_tokens += non_padding

            # Reference tokens
            ref_indices = tgt_y.squeeze(0).tolist()
            ref_tokens = []
            for token_idx in ref_indices:
                if token_idx == self.vocab.eos_idx:
                    break
                if token_idx != self.vocab.pad_idx:
                    ref_tokens.append(self.vocab.idx_to_token(token_idx))
            all_references.append(ref_tokens)

            # Hypothesis (greedy decode)
            decode_max_len = min(tgt_y.size(1) + 10, self.model.max_seq_len)
            hyp_indices = self.model.greedy_decode(
                src,
                max_len=decode_max_len,
                sos_idx=self.vocab.sos_idx,
                eos_idx=self.vocab.eos_idx
            ).squeeze(0).tolist()

            hyp_tokens = []
            for token_idx in hyp_indices:
                if token_idx == self.vocab.eos_idx:
                    break
                if token_idx not in [self.vocab.pad_idx, self.vocab.sos_idx]:
                    hyp_tokens.append(self.vocab.idx_to_token(token_idx))
            all_hypotheses.append(hyp_tokens)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        bleu = compute_bleu(all_references, all_hypotheses)

        return avg_loss, bleu

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 30,
        save_dir: str = "./checkpoints",
        log_interval: int = 100,
        save_interval: int = 5000
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
            log_interval: Steps between logging
            save_interval: Steps between checkpointing

        Returns:
            Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_bleu": [],
            "learning_rate": []
        }

        print("=" * 60)
        print("Starting training")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Steps per epoch: ~{len(train_loader)}")
        print("=" * 60)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0
            running_count = 0

            # Create progress bar for this epoch
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                dynamic_ncols=True
            )

            for batch in pbar:
                loss = self.train_step(batch)

                running_loss += loss
                running_count += 1
                history["train_loss"].append(loss)

                # Update progress bar
                avg_loss = running_loss / running_count
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                # Record learning rate periodically
                if self.global_step % log_interval == 0:
                    history["learning_rate"].append(lr)
                    running_loss = 0
                    running_count = 0

                # Checkpointing
                if self.global_step % save_interval == 0:
                    self.save_checkpoint(save_dir / f"step_{self.global_step}.pt")

            # End of epoch evaluation
            val_loss, val_bleu = self.validate(val_loader)
            history["val_loss"].append(val_loss)
            history["val_bleu"].append(val_bleu)
            print(f"\nEpoch {epoch + 1} complete: val_loss={val_loss:.4f}, BLEU={val_bleu:.2f}")

            # Save best model
            if val_bleu > self.best_bleu:
                self.best_bleu = val_bleu
                self.save_checkpoint(save_dir / "best.pt")
                print(f"  ★ New best BLEU: {val_bleu:.2f}")

            # Show translation samples
            self.show_translation_samples(val_loader.dataset, n_samples=3)

        # Save final model
        self.save_checkpoint(save_dir / "final.pt")

        print("\n" + "=" * 60)
        print(f"Training complete! Best BLEU: {self.best_bleu:.2f}")
        print("=" * 60)

        return history

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_bleu": self.best_bleu
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_bleu = checkpoint.get("best_bleu", 0.0)

        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from step {self.global_step}")

    @torch.no_grad()
    def translate(
        self,
        text: str,
        beam_size: int = 5,
        max_len: int = 100
    ) -> str:
        """
        Translate a single sentence.

        Args:
            text: Source text
            beam_size: Beam size for beam search (1 = greedy)
            max_len: Maximum output length

        Returns:
            Translated text
        """
        self.model.eval()

        # Encode source
        src_indices = self.vocab.encode(text, add_special_tokens=True)
        src = torch.tensor([src_indices], dtype=torch.long, device=self.device)

        # Decode
        if beam_size > 1:
            output = self.model.beam_search(
                src,
                max_len=max_len,
                sos_idx=self.vocab.sos_idx,
                eos_idx=self.vocab.eos_idx,
                beam_size=beam_size
            )
        else:
            output = self.model.greedy_decode(
                src,
                max_len=max_len,
                sos_idx=self.vocab.sos_idx,
                eos_idx=self.vocab.eos_idx
            )

        # Decode to text
        output_indices = output.squeeze(0).tolist()
        translation = self.vocab.decode(output_indices, skip_special_tokens=True)

        return translation

    def show_translation_samples(
        self,
        dataset,
        n_samples: int = 3,
        max_len: int = 100
    ):
        """
        Show random translation samples from the dataset.

        Args:
            dataset: Dataset with src_texts and tgt_texts attributes
            n_samples: Number of samples to show
            max_len: Maximum output length for decoding
        """
        self.model.eval()

        # Get random indices
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

        print("\n" + "-" * 40)
        print("Translation Samples:")
        print("-" * 40)

        for i, idx in enumerate(indices, 1):
            src_text = dataset.src_texts[idx]
            tgt_text = dataset.tgt_texts[idx]

            # Translate
            translation = self.translate(src_text, beam_size=1, max_len=max_len)

            print(f"\n[{i}] Source (De): {src_text}")
            print(f"    Target (En): {tgt_text}")
            print(f"    Model  (En): {translation}")
