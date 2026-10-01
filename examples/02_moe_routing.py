"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""2. Conditional computation: Top-K routing to SwiGLU experts."""
from engramforge.moe.layer import MoELayer

moe = MoELayer(CFG.moe, CFG.model)
for tok in ["token-a", "token-b", "token-c"]:
    topk, weights = moe.router.route(tok)
    print(f"{tok}: topk={topk} weights={[round(w, 3) for w in weights]}")
