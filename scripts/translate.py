#!/usr/bin/env python3
"""
Translation/inference script for trained Transformer model.

Usage:
    python scripts/translate.py --checkpoint checkpoints/best.pt --text "Guten Tag"
    python scripts/translate.py --checkpoint checkpoints/best.pt --input input.txt --output output.txt
    python scripts/translate.py --checkpoint checkpoints/best.pt --interactive
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from transformer.models import Transformer
from transformer.training import Trainer
from transformer.utils.data import Vocabulary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate with trained Transformer model"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Single text to translate"
    )
    input_group.add_argument(
        "--input",
        type=str,
        help="Input file with one sentence per line"
    )
    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive translation mode"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for translations"
    )

    # Vocabulary
    parser.add_argument(
        "--vocab",
        type=str,
        default="./data/vocab.json",
        help="Path to vocabulary file"
    )

    # Model arguments (should match training)
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
        "--max_seq_len",
        type=int,
        default=256,
        help="Maximum sequence length"
    )

    # Decoding arguments
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for beam search (1 = greedy)"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="Maximum output length"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    return parser.parse_args()


def load_model(args, vocab: Vocabulary) -> Trainer:
    """Load model from checkpoint."""
    # Create model
    model = Transformer(
        src_vocab_size=vocab.size,
        tgt_vocab_size=vocab.size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff,
        dropout=0.0,  # No dropout during inference
        max_seq_len=args.max_seq_len,
        padding_idx=vocab.pad_idx,
        share_embedding=True
    )

    # Create trainer (for translation method)
    trainer = Trainer(
        model=model,
        vocab=vocab,
        device=args.device
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from {args.checkpoint}")
    return trainer


def translate_single(trainer: Trainer, text: str, beam_size: int, max_len: int) -> str:
    """Translate a single sentence."""
    return trainer.translate(text, beam_size=beam_size, max_len=max_len)


def translate_file(
    trainer: Trainer,
    input_path: str,
    output_path: str,
    beam_size: int,
    max_len: int
):
    """Translate a file."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else None

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Translating {len(lines)} sentences...")

    # Translate
    translations = []
    for i, line in enumerate(lines):
        translation = translate_single(trainer, line, beam_size, max_len)
        translations.append(translation)

        if (i + 1) % 10 == 0:
            print(f"  Translated {i + 1}/{len(lines)}")

    # Output
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for trans in translations:
                f.write(trans + '\n')
        print(f"Translations saved to {output_path}")
    else:
        for src, tgt in zip(lines, translations):
            print(f"DE: {src}")
            print(f"EN: {tgt}")
            print()


def interactive_mode(trainer: Trainer, beam_size: int, max_len: int):
    """Interactive translation mode."""
    print("\n" + "=" * 60)
    print("Interactive Translation Mode")
    print("Enter German text to translate (Ctrl+C to exit)")
    print("=" * 60 + "\n")

    try:
        while True:
            # Get input
            text = input("DE: ").strip()
            if not text:
                continue

            # Translate
            translation = translate_single(trainer, text, beam_size, max_len)
            print(f"EN: {translation}\n")

    except KeyboardInterrupt:
        print("\n\nExiting...")


def main():
    args = parse_args()

    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab}")
    vocab = Vocabulary.load(args.vocab)
    print(f"Vocabulary size: {vocab.size}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    trainer = load_model(args, vocab)

    # Translate
    if args.text:
        # Single text
        translation = translate_single(
            trainer, args.text, args.beam_size, args.max_len
        )
        print(f"\nDE: {args.text}")
        print(f"EN: {translation}")

    elif args.input:
        # File
        translate_file(
            trainer, args.input, args.output, args.beam_size, args.max_len
        )

    elif args.interactive:
        # Interactive
        interactive_mode(trainer, args.beam_size, args.max_len)


if __name__ == "__main__":
    main()
