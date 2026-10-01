"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""1. O(1) sparse lookup of static n-gram memory."""
from engramforge.memory.engram import EngramModule

em = EngramModule(CFG.engram, CFG.model)
tokens = [101, 202, 303, 404, 505, 606]
vec = em.forward(tokens)
print(f"tokens={tokens}")
print(f"fused vector: dim={len(vec)}  sum={sum(vec):.4f}")
print("bank stats:", em.stats())
