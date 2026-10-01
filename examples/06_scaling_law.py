"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""6. U-shaped sparsity-allocation scaling law."""
from engramforge.train.scaling import run_scaling

run_scaling(CFG)
