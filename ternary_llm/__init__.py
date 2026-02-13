"""
TernaryLLM - Quantum-Informed Accelerated Training System
==========================================================

Ein komplettes System zum Training von ternären LLMs mit:
- Ternäre Gewichte (-1, 0, +1) für 16x Kompression
- Hebbian Learning für 3x schnelleres Training
- Fisher-optimierte Lernraten
- Entropy-basiertes Sample Filtering
- Dynamische Netzwerk-Expansion

Author: TernaryLLM Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "TernaryLLM Team"

from .model import TernaryLLM, TernaryConfig
from .trainer import TernaryTrainer, TrainingConfig
from .data import DatasetLoader, TextDataset, SimpleTokenizer, TokenizerConfig
from .export import ModelExporter
from .utils import setup_logging

__all__ = [
    "TernaryLLM",
    "TernaryConfig", 
    "TernaryTrainer",
    "TrainingConfig",
    "DatasetLoader",
    "TextDataset",
    "SimpleTokenizer",
    "TokenizerConfig",
    "ModelExporter",
    "setup_logging",
]
