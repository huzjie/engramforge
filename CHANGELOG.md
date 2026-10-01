# Changelog

## [1.0.0] - 2026-10-01

### Added
- Engram conditional-memory module: deterministic n-gram addressing, O(1) sparse lookup, gated fusion.
- MemoryBank with host-memory offload and simulated fetch accounting.
- MoE conditional-computation backbone (Top-K router + SwiGLU experts).
- U-shaped sparsity-allocation scaling law.
- Five backends: mock (trainable), cpu, openai, vllm, transformers.
- Four-domain benchmark suite (knowledge / reasoning / code / math).
- OpenAI-compatible stdlib HTTP server, CLI, Docker/K8s/Helm, CI.
- LangChain + MCP integrations.
- 30+ pages of docs, runnable examples, and unit tests.
