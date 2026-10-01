"""Top-K gating router (conditional computation)."""
import math

from ..utils.stable import stable_float


class TopKRouter:
    def __init__(self, num_experts=8, top_k=2, hidden=512, key="router"):
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.key = key

    def logits(self, token_embedding):
        return [stable_float(f"{self.key}:{i}:{token_embedding}", -1, 1) for i in range(self.num_experts)]

    def route(self, token_embedding):
        logits = self.logits(token_embedding)
        topk = sorted(range(self.num_experts), key=lambda i: logits[i], reverse=True)[:self.top_k]
        mx = max(logits)
        ex = [math.exp(x - mx) for x in logits]
        s = sum(ex)
        probs = [e / s for e in ex]
        weights = [probs[i] for i in topk]
        sw = sum(weights) or 1.0
        weights = [w / sw for w in weights]
        return topk, weights

    def load_balance_loss(self, token_embeddings):
        """Auxiliary loss encouraging uniform expert usage (deterministic)."""
        counts = [0] * self.num_experts
        for t in token_embeddings:
            topk, _ = self.route(t)
            for e in topk:
                counts[e] += 1
        n = len(token_embeddings) or 1
        mean = n * self.top_k / self.num_experts
        return sum((c - mean) ** 2 for c in counts) / (self.num_experts * mean * mean + 1e-6)
