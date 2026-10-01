"""N-gram encoding over token sequences.

Language modelling mixes two sub-tasks: compositional reasoning (dynamic, deep
computation) and knowledge retrieval (static, local, highly stereotyped patterns
like named entities and formulaic expressions). Classic n-gram models are very
good at the latter. Engram modernizes n-gram embeddings into a learnable static
memory addressed by a deterministic hash.
"""


class NgramEncoder:
    """Enumerate all n-grams of order 1..order for a token sequence."""

    def __init__(self, order=3):
        self.order = max(1, order)

    def encode(self, tokens):
        """Return a list of n-gram tuples (order 1..order) present in the sequence."""
        grams = []
        n = len(tokens)
        for k in range(1, self.order + 1):
            for i in range(0, n - k + 1):
                grams.append(tuple(tokens[i:i + k]))
        return grams

    def context_grams(self, tokens):
        """Only the highest-order grams ending at the last token (for next-token lookups)."""
        grams = []
        n = len(tokens)
        for k in range(1, self.order + 1):
            if n >= k:
                grams.append(tuple(tokens[n - k:n]))
        return grams
