"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""5. Host-memory embedding offload benchmark."""
from engramforge.bench.offload import run_offload

stats = run_offload(CFG)
