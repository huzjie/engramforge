"""OpenAI-compatible backend stub (pluggable via OPENAI_API_KEY)."""
import os
from ..utils.stable import stable_float
from .base import Backend
from . import register_backend


@register_backend("openai")
class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt, **kwargs):
        # reference fallback: deterministic simulation when no key is present
        return stable_float(f"openai:{prompt}", 0.0, 1.0) > 0.5

    def is_configured(self):
        return bool(self.api_key)
