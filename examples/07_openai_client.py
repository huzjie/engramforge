"""Demo script. Run directly: `python examples/<name>.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engramforge.config import load_config

CFG = load_config()

"""7. Query the OpenAI-compatible server (start it first with `python -m engramforge serve`)."""
import json
import urllib.request

req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps({
        "model": "engramforge-27b",
        "messages": [{"role": "user", "content": "What is conditional memory?"}],
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=10) as r:
    print(json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"])
