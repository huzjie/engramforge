"""Deterministic addressing for O(1) sparse lookup.

Engram's key system property is *deterministic addressing*: the slot for a given
n-gram is computed by a pure hash function, not learned. This lets a massive
embedding table be addressed without any dynamic routing network, and lets the
table be offloaded to host memory because lookups are direct and cache-friendly.
"""
import hashlib


class DeterministicHasher:
    """A family of k deterministic hash functions over a fixed table size.

    Uses md5(key || salt_i) to derive slot indices. Collisions are resolved with
    double hashing when probing for insertion, and reads simply average the
    embeddings found at each of the k candidate slots.
    """

    def __init__(self, table_size=65536, num_hashes=4, salt="engram"):
        self.table_size = table_size
        self.num_hashes = num_hashes
        self.salt = salt

    def _h(self, key, i):
        m = hashlib.md5(f"{self.salt}:{i}:{key}".encode("utf-8")).digest()
        return int.from_bytes(m[:8], "big")

    def slot(self, key, i):
        """i-th candidate slot for a key (0 <= slot < table_size)."""
        return self._h(key, i) % self.table_size

    def slots(self, key):
        """All candidate slots for a key."""
        return [self.slot(key, i) for i in range(self.num_hashes)]

    def probe(self, key, i, step):
        """Double-hashing probe: slot(key, i) + step * slot(key, num_hashes)."""
        return (self.slot(key, i) + step * self.slot(key, self.num_hashes)) % self.table_size


def key_of(gram):
    """Canonical string key for an n-gram tuple (deterministic)."""
    if isinstance(gram, (list, tuple)):
        return ":".join(str(g) for g in gram)
    return str(gram)
