"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""11. Inspect MemoryBank internals: insertion, lookup, eviction."""
from engramforge.memory.table import MemoryBank

bank = MemoryBank(table_size=1024, embedding_dim=64, num_hashes=4, offload=True)
for i in range(5000):
    bank.insert((i, i + 1, i + 2))
print("stats after 5000 inserts:", bank.stats())
print("lookup (1,2,3) dim=", len(bank.lookup((1, 2, 3))))
