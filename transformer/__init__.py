"""
Transformer for IWSLT 2016 De-En Translation

A pure PyTorch implementation of the Transformer architecture
from "Attention Is All You Need" (Vaswani et al., 2017).
"""

from .models import Transformer

__version__ = "1.0.0"
__all__ = ["Transformer"]
