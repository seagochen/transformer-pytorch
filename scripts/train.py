#!/usr/bin/env python3
"""
Training script for Transformer on IWSLT 2016 De-En translation.

Usage:
    python scripts/train.py
    python scripts/train.py --config training.yaml
    python scripts/train.py --epochs 50 --batch_size 64
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from transformer.models import Transformer
from transformer.training import Trainer
from transformer.utils.data import create_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Transformer on IWSLT 2016 De-En"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory for dataset"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=256,
        help="Maximum sequence length"
    )

    # Model arguments
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Model dimension"
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--n_encoder_layers",
        type=int,
        default=6,
        help="Number of encoder layers"
    )
    parser.add_argument(
        "--n_decoder_layers",
        type=int,
        default=6,
        help="Number of decoder layers"
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=2048,
        help="Feed-forward dimension"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Base learning rate"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4000,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor"
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value"
    )

    # Logging and saving
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Steps between logging"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1000,
        help="Steps between evaluation"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config(args, config: dict):
    """Merge config file values with command line arguments."""
    for section in ['model', 'training', 'data']:
        if section in config:
            for key, value in config[section].items():
                if hasattr(args, key):
                    # Only override if not explicitly set on command line
                    setattr(args, key, value)
    return args


def main():
    args = parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        args = merge_config(args, config)

    print("=" * 60)
    print("Transformer Training - IWSLT 2016 De-En")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model: d_model={args.d_model}, heads={args.n_heads}, "
          f"layers={args.n_encoder_layers}/{args.n_decoder_layers}")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}")
    print("=" * 60)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        data_dir=args.data_dir,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size
    )

    # Create model
    print("\nCreating model...")
    model = Transformer(
        src_vocab_size=vocab.size,
        tgt_vocab_size=vocab.size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        padding_idx=vocab.pad_idx,
        share_embedding=True  # Share embedding for BPE
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        vocab=vocab,
        device=args.device,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        gradient_clip=args.gradient_clip
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval
    )

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_loss, test_bleu = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test BLEU: {test_bleu:.2f}")

    # Save final metrics
    metrics = {
        "test_loss": test_loss,
        "test_bleu": test_bleu,
        "best_val_bleu": trainer.best_bleu
    }

    metrics_path = Path(args.save_dir) / "metrics.yaml"
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)

    print(f"\nMetrics saved to {metrics_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
