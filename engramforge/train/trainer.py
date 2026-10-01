"""Training loop over the trainable mock backend."""
from ..backends import get_backend


def run_train(cfg):
    backend = get_backend(cfg.backend, cfg)
    print(f"[train] backend={cfg.backend} skill_initial={backend.current_skill():.3f}", flush=True)
    last_loss = None
    for step in range(cfg.train.max_steps):
        last_loss = backend.train_step(step)
        if step % 200 == 0:
            print(f"[train] step={step} skill={backend.current_skill():.3f} loss={last_loss:.3f}", flush=True)
    print(f"[train] skill_final={backend.current_skill():.3f} loss_final={last_loss:.3f}", flush=True)
    return backend.current_skill()
