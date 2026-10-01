"""Static embedding MemoryBank with host-memory offload.

Engram stores fixed knowledge as static embeddings in a giant table. Because
addressing is deterministic (see hash.py), the table never needs to be fully
resident on the accelerator: the vast majority can live in host memory and be
fetched on demand, which is what makes the 27B-scale table practical.
"""
from ..core.tensor import zeros
from ..utils.stable import stable_vector
from .hash import DeterministicHasher, key_of


class MemoryBank:
    """A static embedding table addressed by deterministic hashing.

    offload=True simulates host-memory residency: only a small active cache is
    kept in the "device" dict; the rest is materialized on demand and counted as
    a simulated host fetch (tiny latency, no real I/O in the reference impl).
    """

    def __init__(self, table_size=65536, embedding_dim=256, num_hashes=4, offload=True,
                 fp8_quant=False):
        self.table_size = table_size
        self.embedding_dim = embedding_dim
        self.offload = offload
        self.fp8_quant = fp8_quant
        self.hasher = DeterministicHasher(table_size, num_hashes)
        self._store = {}          # slot -> embedding list (resident or cache)
        self._inserted = set()    # keys actually written (vs uninitialized slots)
        self._host_fetches = 0
        self._cache_size = 4096   # resident cache capacity (slots)

    def _embedding_for(self, key):
        # static embedding is fully determined by the key (deterministic, learnable in real training)
        return stable_vector(f"mem:{key}", self.embedding_dim, -0.1, 0.1)

    def insert(self, gram):
        """Insert a gram's static embedding (best-effort, collision-resolving)."""
        key = key_of(gram)
        if key in self._inserted:
            return
        emb = self._embedding_for(key)
        for i in range(self.hasher.num_hashes):
            slot = self.hasher.slot(key, i)
            if slot not in self._store:
                self._store[slot] = emb
                self._inserted.add(key)
                self._evict_if_needed()
                return
        # all candidate slots occupied -> write to the first (overwrite is allowed in lookup tables)
        self._store[self.hasher.slot(key, 0)] = emb
        self._inserted.add(key)
        self._evict_if_needed()

    def _evict_if_needed(self):
        if not self.offload:
            return
        if len(self._store) > self._cache_size:
            # simulate offload: evict the least recently used half to host memory
            keys = list(self._store.keys())[: len(self._store) // 2]
            for k in keys:
                del self._store[k]
            self._host_fetches += len(keys)

    def _fetch(self, key, i):
        slot = self.hasher.slot(key, i)
        if slot in self._store:
            return self._store[slot]
        # host-memory fetch (simulated)
        if self.offload:
            self._host_fetches += 1
            self._store[slot] = self._embedding_for(key)
            self._evict_if_needed()
            return self._store[slot]
        return self._embedding_for(key)

    def lookup(self, gram):
        """Average the embeddings found at the k candidate slots for a gram."""
        key = key_of(gram)
        acc = [0.0] * self.embedding_dim
        for i in range(self.hasher.num_hashes):
            emb = self._fetch(key, i)
            for j in range(self.embedding_dim):
                acc[j] += emb[j]
        n = self.hasher.num_hashes
        return [x / n for x in acc]

    def stats(self):
        return {
            "table_size": self.table_size,
            "resident_slots": len(self._store),
            "inserted_keys": len(self._inserted),
            "host_fetches": self._host_fetches,
            "offload": self.offload,
        }
