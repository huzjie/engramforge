"""Math benchmark: skill-driven accuracy on synthetic arithmetic tasks."""


class MathBench:
    name = "math"

    def run(self, cfg, backend):
        n = 100
        correct = sum(1 for i in range(n) if backend.generate(f"math:q{i}"))
        acc = correct / n
        return {"score": acc, "detail": f"acc={acc:.2f}"}
