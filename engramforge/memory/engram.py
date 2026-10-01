"""EngramModule: O(1) sparse lookup fused with dynamic hidden states.

The module augments the backbone by (1) retrieving static n-gram memory and
(2) fusing it with the dynamic hidden states through a learned gate. Static
patterns (entities, formulas) are thus served by cheap table lookup instead of
being re-derived by early attention/FFN layers, preserving effective depth for
genuine compositional reasoning.
"""
from ..core.tensor import Tensor, add, mul
from ..utils.stable import stable_vector
from .ngram import NgramEncoder
from .table import MemoryBank


class EngramModule:
    """Conditional-memory module: deterministic lookup + gated fusion."""

    def __init__(self, engram_cfg, model_cfg):
        self.cfg = engram_cfg
        self.model_cfg = model_cfg
        self.encoder = NgramEncoder(engram_cfg.ngram_order)
        self.bank = MemoryBank(
            table_size=engram_cfg.table_size,
            embedding_dim=engram_cfg.embedding_dim,
            num_hashes=engram_cfg.num_hashes,
            offload=engram_cfg.offload,
            fp8_quant=engram_cfg.fp8_quant,
        )
        self.gate = engram_cfg.gate_init

    def retrieve(self, tokens):
        """Retrieve (and cache) the static memory vector for a token sequence.

        Averages the embeddings of every n-gram present in the context.
        """
        grams = self.encoder.encode(tokens)
        if not grams:
            return [0.0] * self.cfg.embedding_dim
        acc = [0.0] * self.cfg.embedding_dim
        for g in grams:
            self.bank.insert(g)
            emb = self.bank.lookup(g)
            for j in range(self.cfg.embedding_dim):
                acc[j] += emb[j]
        n = len(grams)
        return [x / n for x in acc]

    def _hidden(self, tokens):
        # dynamic hidden states (simulated; in a real model these come from the backbone)
        return stable_vector(f"hidden:{tokens}", self.cfg.embedding_dim, -0.2, 0.2)

    def forward(self, tokens, hidden=None):
        """Fuse static memory with dynamic hidden states through a learned gate.

        fused = gate * mem + (1 - gate) * hidden
        """
        mem = self.retrieve(tokens)
        if hidden is None:
            hidden = self._hidden(tokens)
        g = max(0.0, min(1.0, self.gate))
        fused = [g * m + (1.0 - g) * h for m, h in zip(mem, hidden)]
        return fused

    def __call__(self, tokens, hidden=None):
        return self.forward(tokens, hidden)

    def set_gate(self, value):
        self.gate = max(0.0, min(1.0, value))

    def stats(self):
        return self.bank.stats()
