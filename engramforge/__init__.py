"""engramforge: conditional-memory sparse LLM training & inference framework.

A zero-dependency, fully runnable reference implementation inspired by DeepSeek's
open-sourced Engram module (Conditional Memory via Scalable Lookup, 2026-10-01):

- Engram  -- O(1) sparse lookup for static knowledge via deterministic n-gram addressing
- MoE     -- conditional computation for dynamic compositional reasoning (Top-K + SwiGLU)
- Sparsity Allocation -- U-shaped scaling law trading neural compute vs static memory
- Offload -- deterministic addressing lets the giant embedding table live in host memory

Every component runs on a deterministic, trainable mock backend with no external
dependencies, plus a real CPU backend, OpenAI/vLLM/Transformers backends, an
OpenAI-compatible stdlib HTTP server, CLI, Docker/K8s/Helm, CI and LangChain/MCP
integrations.
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
