"""Fit and report the U-shaped sparsity-allocation scaling law."""
from ..memory.allocator import SparsityAllocator


def run_scaling(cfg):
    alloc = SparsityAllocator()
    print("[scaling] U-shaped loss sweep (optimal=0.35):", flush=True)
    for r, l in alloc.sweep(n=10):
        bar = "#" * int(round((l - 0.4) * 40))
        print(f"  ratio={r:.2f} loss={l:.3f} {bar}", flush=True)
    print("[scaling] optimal engram ratio vs compute budget:", flush=True)
    for b, r in alloc.scaling_law([1e3, 1e6, 1e9, 1e12, 1e15]):
        print(f"  budget={b:.0e} -> optimal_engram_ratio={r:.3f}", flush=True)
    return alloc.scaling_law([1e3, 1e6, 1e9, 1e12, 1e15])
