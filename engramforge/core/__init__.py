"""Core primitives: dtype, device, registry, zero-dependency tensor."""
from .dtype import DType
from .registry import Registry
from .tensor import Tensor, zeros, ones, randn, matmul, add, mul, relu, gelu, silu, softmax, layernorm, rmsnorm

__all__ = [
    "DType", "Registry", "Tensor",
    "zeros", "ones", "randn", "matmul", "add", "mul",
    "relu", "gelu", "silu", "softmax", "layernorm", "rmsnorm",
]
