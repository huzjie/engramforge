"""Engram conditional-memory primitives.

- hash      : deterministic addressing (multi-hash + double hashing)
- ngram     : n-gram encoder over token sequences
- table     : static embedding MemoryBank with host-memory offload
- engram    : EngramModule (O(1) lookup + gated fusion with dynamic hidden states)
- allocator : U-shaped sparsity-allocation scaling law (MoE vs Engram)
"""
from .hash import DeterministicHasher
from .ngram import NgramEncoder
from .table import MemoryBank
from .engram import EngramModule
from .allocator import SparsityAllocator

__all__ = [
    "DeterministicHasher", "NgramEncoder", "MemoryBank",
    "EngramModule", "SparsityAllocator",
]
