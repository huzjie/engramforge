"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""13. Full pipeline: data -> train -> bench -> analysis -> offload."""
from engramforge.train.data import SyntheticCorpus
from engramforge.backends import get_backend
from engramforge.bench.run import run_all
from engramforge.bench.analysis import run_analysis
from engramforge.bench.offload import run_offload

corpus = SyntheticCorpus(num_docs=100)
print("docs:", len(corpus.docs()), "first domain:", corpus.docs()[0]["domain"])
be = get_backend(CFG.backend, CFG)
for step in range(500):
    be.train_step(step)
print("skill:", round(be.current_skill(), 3))
run_all(CFG)
run_analysis(CFG)
run_offload(CFG)
