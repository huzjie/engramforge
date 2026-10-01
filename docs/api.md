# API 参考

## memory
- `NgramEncoder(order).encode(tokens)`
- `DeterministicHasher(table_size, num_hashes).slots(key)`
- `MemoryBank(...).insert(gram) / .lookup(gram) / .stats()`
- `EngramModule(engram_cfg, model_cfg).forward(tokens) / .retrieve(tokens) / .set_gate(v)`
- `SparsityAllocator().loss / .optimal_engram_ratio / .sweep`

## moe
- `TopKRouter(...).route(token)`
- `MoELayer(moe_cfg, model_cfg).forward(token)`

## model
- `EngramTransformer(model_cfg, engram_cfg, moe_cfg).forward(tokens)`

## backends
- `get_backend(name, cfg)` / `list_backends()`

## bench
- `run_all(cfg)` / `run_offload(cfg)` / `run_analysis(cfg)`
