"""vLLM backend stub (pluggable via VLLM_MODEL)."""
import os
from ..utils.stable import stable_float
from .base import Backend
from . import register_backend


@register_backend("vllm")
class VllmBackend(Backend):
    name = "vllm"

    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.model = os.environ.get("VLLM_MODEL", "engramforge/engram-27b")

    def generate(self, prompt, **kwargs):
        return stable_float(f"vllm:{prompt}", 0.0, 1.0) > 0.5
