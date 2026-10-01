"""Reasoning benchmark: compositional, dynamic (skill-driven only)."""


class ReasoningBench:
    name = "reasoning"

    def run(self, cfg, backend):
        n = 100
        correct = sum(1 for i in range(n) if backend.generate(f"reason:q{i}"))
        acc = correct / n
        return {"score": acc, "detail": f"acc={acc:.2f}"}
