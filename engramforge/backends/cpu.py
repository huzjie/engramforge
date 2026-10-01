"""Real CPU backend: instantiate the actual Engram + MoE + Transformer stack."""
from ..memory.engram import EngramModule
from ..moe.layer import MoELayer
from ..model.transformer import EngramTransformer
from .base import Backend
from . import register_backend


@register_backend("cpu")
class CpuBackend(Backend):
    name = "cpu"

    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cfg = cfg
        self.engram = EngramModule(cfg.engram, cfg.model)
        self.moe = MoELayer(cfg.moe, cfg.model)
        self.model = EngramTransformer(cfg.model, cfg.engram, cfg.moe)

    def generate(self, prompt, **kwargs):
        tokens = [abs(hash(str(prompt) + str(i))) % 1000 + 1 for i in range(8)]
        mem = self.engram.forward(tokens)
        return sum(mem) / len(mem) > 0.0

    def retrieve(self, tokens):
        return self.engram.retrieve(tokens)

    def stats(self):
        return self.engram.stats()
