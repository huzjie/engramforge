"""MoE layer: route -> expert compute -> weighted combine (conditional computation)."""
from .router import TopKRouter
from .expert import SwiGLUExpert


class MoELayer:
    def __init__(self, moe_cfg, model_cfg):
        self.cfg = moe_cfg
        self.model_cfg = model_cfg
        self.router = TopKRouter(moe_cfg.num_experts, moe_cfg.top_k, model_cfg.hidden_size)
        self.experts = [SwiGLUExpert(model_cfg.hidden_size, moe_cfg.expert_hidden, i)
                        for i in range(moe_cfg.num_experts)]

    def forward(self, token_embedding):
        """Weighted sum of the top-k expert outputs for one token."""
        topk, weights = self.router.route(token_embedding)
        dim = self.model_cfg.hidden_size
        acc = [0.0] * dim
        for e, w in zip(topk, weights):
            out = self.experts[e].forward(token_embedding)
            # project expert output (expert_hidden) down to hidden
            step = max(1, len(out) // dim)
            for j in range(dim):
                acc[j] += w * out[j * step]
        return acc

    def __call__(self, token_embedding):
        return self.forward(token_embedding)
