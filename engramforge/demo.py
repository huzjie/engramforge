"""End-to-end demo: trainable mock backend + Engram lookup + MoE + benchmarks."""
from .config import Config


def run(cfg: Config):
    from .backends import get_backend
    from .bench.run import run_all

    backend = get_backend(cfg.backend, cfg)
    print(f"[demo] backend={cfg.backend} initial skill={backend.current_skill():.3f}", flush=True)

    for step in range(30):
        loss = backend.train_step(step)
    print(f"[demo] after 30 steps skill={backend.current_skill():.3f} loss={loss:.3f}", flush=True)

    report = run_all(cfg)
    avg = report["avg_score"]
    print(f"[demo] benchmark AVG score = {avg:.4f}", flush=True)
