"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""4. Run the full knowledge/reasoning/code/math benchmark suite."""
from engramforge.bench.run import run_all

report = run_all(CFG)
print(f"AVG score = {report['avg_score']:.4f}  skill = {report['backend_skill']:.3f}")
