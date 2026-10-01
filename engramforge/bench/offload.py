"""Host-memory embedding offload benchmark.

Engram's deterministic addressing makes the giant embedding table offloadable to
host memory with minimal inference overhead. This benchmark measures (in the
reference impl) resident cache size, host-fetch counts and effective hit rate.
"""
from ..memory.engram import EngramModule


def run_offload(cfg):
    engram = EngramModule(cfg.engram, cfg.model)
    # insert many distinct grams to force evictions (offload)
    grams = []
    for i in range(20000):
        grams.append(tuple([i % 5000 + 1, (i * 7) % 5000 + 1, (i * 13) % 5000 + 1]))
    for g in grams:
        engram.bank.insert(g)
    # perform a second pass of lookups -> host fetches
    for g in grams[:2000]:
        engram.bank.lookup(g)
    stats = engram.stats()
    print("[offload] stats:", flush=True)
    for k, v in stats.items():
        print(f"  {k} = {v}", flush=True)
    print(f"[offload] offload enabled={stats['offload']} resident={stats['resident_slots']} "
          f"host_fetches={stats['host_fetches']}", flush=True)
    return stats
