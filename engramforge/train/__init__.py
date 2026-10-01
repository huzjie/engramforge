"""Training utilities."""
from .data import SyntheticCorpus
from .trainer import run_train
from .scaling import run_scaling

__all__ = ["SyntheticCorpus", "run_train", "run_scaling"]
