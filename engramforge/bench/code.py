"""Code benchmark: skill-driven accuracy on synthetic coding tasks."""


class CodeBench:
    name = "code"

    def run(self, cfg, backend):
        n = 100
        correct = sum(1 for i in range(n) if backend.generate(f"code:q{i}"))
        acc = correct / n
        return {"score": acc, "detail": f"acc={acc:.2f}"}
