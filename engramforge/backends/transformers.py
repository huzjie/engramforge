"""HuggingFace Transformers backend stub."""
import os
from ..utils.stable import stable_float
from .base import Backend
from . import register_backend


@register_backend("transformers")
class TransformersBackend(Backend):
    name = "transformers"

    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.model = os.environ.get("HF_MODEL", "engramforge/engram-27b")

    def generate(self, prompt, **kwargs):
        return stable_float(f"hf:{prompt}", 0.0, 1.0) > 0.5
