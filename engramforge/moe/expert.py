"""SwiGLU expert (feed-forward with gated activation)."""
import math

from ..core.tensor import silu


class SwiGLUExpert:
    def __init__(self, hidden=512, expert_hidden=1024, idx=0, key="expert"):
        self.hidden = hidden
        self.expert_hidden = expert_hidden
        self.idx = idx
        self.key = key

    def forward(self, x):
        """Deterministic reference forward (no real weights in mock mode)."""
        import random
        from ..utils.stable import _seed
        r = random.Random(_seed(f"{self.key}:{self.idx}:{str(x)[:64]}"))
        return [r.uniform(-0.5, 0.5) for _ in range(self.expert_hidden)]


def swiglu(x):
    """Element-wise SwiGLU(x) = x * silu(x)."""
    return [v * (v / (1.0 + math.exp(-v))) for v in x]
