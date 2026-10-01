"""A sparse Transformer block: early blocks fuse Engram, later blocks use MoE."""
from ..core.tensor import layernorm, rmsnorm, add, silu
from ..memory.engram import EngramModule
from ..moe.layer import MoELayer


class EngramBlock:
    """One block: (optional Engram fusion) + attention (simulated) + MoE/FFN + residuals."""

    def __init__(self, layer_idx, model_cfg, engram_cfg, moe_cfg):
        self.idx = layer_idx
        self.model_cfg = model_cfg
        self.use_engram = layer_idx < model_cfg.engram_layers
        self.use_moe = layer_idx >= model_cfg.engram_layers
        self.engram = EngramModule(engram_cfg, model_cfg) if self.use_engram else None
        self.moe = MoELayer(moe_cfg, model_cfg) if self.use_moe else None

    def forward(self, hidden, tokens=None):
        """hidden: list of floats (seq-level). Returns transformed hidden."""
        if self.use_engram and tokens is not None:
            mem = self.engram.forward(tokens)
            hidden = [h + m for h, m in zip(hidden, mem)]
        hidden = rmsnorm(Tensor_like(hidden)) if False else hidden  # placeholder no-op
        if self.use_moe:
            token_repr = str(tokens)[:64] if tokens is not None else "tok"
            out = self.moe.forward(token_repr)
            hidden = [h + o for h, o in zip(hidden, out)]
        return hidden


def Tensor_like(data):
    from ..core.tensor import Tensor
    return Tensor(data)
