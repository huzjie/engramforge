"""EngramTransformer: sparse backbone with conditional memory."""
from ..utils.stable import stable_vector
from .block import EngramBlock


class EngramTransformer:
    """A minimal but structurally faithful Engram-augmented Transformer.

    Early layers attach an EngramModule (static lookup), later layers route
    through MoE (conditional computation). Attention is simulated deterministically
    in the reference implementation; the interesting behaviour is the split of
    static knowledge (Engram) vs dynamic reasoning (MoE).
    """

    def __init__(self, model_cfg, engram_cfg, moe_cfg):
        self.cfg = model_cfg
        self.engram_cfg = engram_cfg
        self.moe_cfg = moe_cfg
        self.blocks = [
            EngramBlock(i, model_cfg, engram_cfg, moe_cfg)
            for i in range(model_cfg.num_layers)
        ]

    def forward(self, tokens):
        hidden = stable_vector(f"h0:{tokens}", self.cfg.hidden_size, -0.1, 0.1)
        for blk in self.blocks:
            hidden = blk.forward(hidden, tokens=tokens)
        # output logits: project hidden -> vocab
        return hidden[:self.cfg.vocab_size] if self.cfg.vocab_size <= len(hidden) else hidden

    def num_engram_layers(self):
        return self.cfg.engram_layers

    def num_moe_layers(self):
        return self.cfg.moe_layers
