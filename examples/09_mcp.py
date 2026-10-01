"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""9. MCP tool + resource schema."""
from engramforge.integrations.mcp import EngramForgeMCP

mcp = EngramForgeMCP(backend="cpu", cfg=CFG)
print("tools:", mcp.tools()[0]["name"])
print("call:", mcp.call_tool("engram_lookup", {"tokens": [1, 2, 3]}))
print("resources:", [r["uri"] for r in mcp.resources()])
