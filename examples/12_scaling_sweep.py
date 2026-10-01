"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""12. Visualize the U-shaped loss sweep."""
from engramforge.memory.allocator import SparsityAllocator

a = SparsityAllocator()
print("ratio | loss | bar")
for r, l in a.sweep(n=20, optimal=0.35, curvature=2.0):
    print(f"{r:.2f}  | {l:.2f} | {'#' * int(l * 20)}")
