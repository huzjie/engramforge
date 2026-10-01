"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""14. Start the OpenAI server in a background thread and hit /health + /v1/models."""
import json
import threading
import urllib.request

from engramforge.serving.server import run_server

t = threading.Thread(target=run_server, args=(CFG,), daemon=True)
t.start()
import time
time.sleep(1.0)
for path in ("/health", "/v1/models"):
    with urllib.request.urlopen(f"http://127.0.0.1:{CFG.serving.port}{path}", timeout=5) as r:
        print(path, "->", r.read().decode("utf-8")[:120])
