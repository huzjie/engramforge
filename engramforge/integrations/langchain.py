"""LangChain integration: expose the backend as an LLM (optional dependency)."""
from ..backends import get_backend


class EngramForgeLLM:
    """A minimal LangChain-compatible LLM wrapper around a backend."""

    def __init__(self, backend="mock", cfg=None, model="engramforge-27b"):
        self.backend = get_backend(backend, cfg)
        self.model = model

    def _call(self, prompt, stop=None, **kwargs):
        return "yes" if self.backend.generate(prompt) else "no"

    def invoke(self, prompt, **kwargs):
        from ._result import LLMResult
        return LLMResult(self._call(prompt))

    @property
    def _llm_type(self):
        return "engramforge"


class LLMResult:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text
