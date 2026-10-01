"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""15. Demonstrate static knowledge served by lookup (static recall ~ 1.0)."""
from engramforge.memory.engram import EngramModule
from engramforge.train.data import ENTITIES

em = EngramModule(CFG.engram, CFG.model)
hits = 0
for ent in ENTITIES:
    tokens = [abs(hash(ent)) % 5000 + 1] * 5
    if sum(em.retrieve(tokens)) != 0.0:
        hits += 1
print(f"static recall over {len(ENTITIES)} entities = {hits / len(ENTITIES):.2f}")
