"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""8. Use the LangChain-style wrapper."""
from engramforge.integrations.langchain import EngramForgeLLM

llm = EngramForgeLLM(backend=CFG.backend, cfg=CFG)
print("invoke ->", llm.invoke("is the sky blue?"))
