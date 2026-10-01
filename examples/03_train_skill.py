"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""3. Train the deterministic mock backend: skill monotonically -> 1.0."""
from engramforge.backends import get_backend

be = get_backend(CFG.backend, CFG)
print(f"initial skill={be.current_skill():.3f}")
for step in range(100):
    loss = be.train_step(step)
print(f"final skill={be.current_skill():.3f}  final loss={loss:.3f}")
print("eval:", be.evaluate())
