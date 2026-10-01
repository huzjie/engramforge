"""Tiny result object for the LangChain shim."""


class LLMResult:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text
