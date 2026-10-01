"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""10. Mechanistic analysis: how Engram relieves early layers."""
from engramforge.bench.analysis import run_analysis

run_analysis(CFG)
