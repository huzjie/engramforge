"""Benchmark orchestration: train first, then evaluate all four domains."""
from ..backends import get_backend
from ..core.registry import Registry
from .knowledge import KnowledgeBench
from .reasoning import ReasoningBench
from .code import CodeBench
from .math import MathBench

BENCHMARKS = Registry("benchmark")
BENCHMARKS.register("knowledge", KnowledgeBench())
BENCHMARKS.register("reasoning", ReasoningBench())
BENCHMARKS.register("code", CodeBench())
BENCHMARKS.register("math", MathBench())


def run_all(cfg):
    backend = get_backend(cfg.backend, cfg)
    # train the backend first (so we see before/after behaviour in one call)
    for step in range(cfg.train.max_steps):
        backend.train_step(step)

    items = {}
    scores = []
    for name in BENCHMARKS.keys():
        bench = BENCHMARKS.get(name)
        result = bench.run(cfg, backend)
        items[name] = result
        scores.append(result["score"])

    avg = sum(scores) / len(scores) if scores else 0.0
    report = {"avg_score": avg, "items": items, "backend_skill": backend.current_skill()}

    print("[bench] === results ===", flush=True)
    for name, r in items.items():
        extra = r.get("detail", "")
        print(f"[bench] {name:<10} score={r['score']:.4f} {extra}", flush=True)
    print(f"[bench] AVG={avg:.4f} skill={backend.current_skill():.3f}", flush=True)
    return report
