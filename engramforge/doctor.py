"""Self-check entrypoint: validate every registered component."""
from .config import Config


def run(cfg: Config):
    from .backends import list_backends
    from .memory.engram import EngramModule
    from .memory.allocator import SparsityAllocator
    from .moe.layer import MoELayer
    from .model.transformer import EngramTransformer
    from .bench.run import BENCHMARKS

    backends = list_backends()
    print(f"[doctor] backends = {backends}", flush=True)

    engram = EngramModule(cfg.engram, cfg.model)
    out = engram([101, 202, 303, 404, 505])
    print(f"[doctor] EngramModule lookup -> fused vec len={len(out)}", flush=True)

    alloc = SparsityAllocator()
    ratio = alloc.optimal_engram_ratio(1e12, 1e12)
    print(f"[doctor] SparsityAllocator optimal engram ratio @1e12 = {ratio:.3f}", flush=True)

    moe = MoELayer(cfg.moe, cfg.model)
    topk, weights = moe.router.route("tok")
    print(f"[doctor] MoELayer route -> topk={topk} weights={[round(w, 3) for w in weights]}", flush=True)

    model = EngramTransformer(cfg.model, cfg.engram, cfg.moe)
    logits = model.forward([1, 2, 3, 4, 5, 6, 7, 8])
    print(f"[doctor] EngramTransformer forward -> logits dim={len(logits)}", flush=True)

    print(f"[doctor] benchmarks registered = {len(BENCHMARKS)}", flush=True)
    print("[doctor] OK", flush=True)
