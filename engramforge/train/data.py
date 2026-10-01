"""Synthetic corpus spanning knowledge / reasoning / code / math.

Engram's evaluation covers four domains. We synthesize a deterministic corpus so
the whole pipeline (data -> train -> eval) runs with zero dependencies.
"""
from ..utils.stable import stable_ints, stable_choice


ENTITIES = [
    "DeepSeek", "北京大学", "conditional-memory", "Engram", "MoE",
    "transformer", "attention", "n-gram", "sparse-lookup", "scaling-law",
]


class SyntheticCorpus:
    def __init__(self, num_docs=1000, seed=42, ngram_order=3):
        self.num_docs = num_docs
        self.seed = seed
        self.ngram_order = ngram_order

    def docs(self):
        out = []
        for i in range(self.num_docs):
            domain = ["knowledge", "reasoning", "code", "math"][i % 4]
            tokens = self._tokens(i, domain)
            out.append({"id": i, "domain": domain, "tokens": tokens})
        return out

    def _tokens(self, i, domain):
        n = 8 + (i % 24)
        base = stable_ints(f"doc:{self.seed}:{i}", n, 1, 5000)
        if domain == "knowledge":
            ent = stable_choice(f"ent:{i}", ENTITIES)
            base = base[:n // 2] + [abs(hash(ent)) % 5000 + 1] * 3 + base[n // 2 + 3:]
        return base

    def knowledge_pairs(self, n=200):
        """(prompt, answer) pairs for the knowledge benchmark."""
        pairs = []
        for i in range(n):
            ent = ENTITIES[i % len(ENTITIES)]
            pairs.append((f"What is {ent}?", ent))
        return pairs
