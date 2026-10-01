"""OpenAI-compatible stdlib HTTP server (no dependencies)."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..backends import get_backend
from ..config import load_config
from .openai import chat_completion


def _make_handler(cfg, backend, model_name):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/v1/models", "/v1/models/"):
                self._send(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            elif self.path == "/health":
                self._send(200, {"status": "ok", "backend": cfg.backend})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/v1/chat/completions":
                ln = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(ln) if ln else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    data = {}
                prompt = " ".join(m.get("content", "") for m in data.get("messages", []))
                resp = chat_completion(model_name, prompt, backend)
                self._send(200, resp)
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def run_server(cfg):
    backend = get_backend(cfg.backend, cfg)
    model_name = "engramforge-27b"
    handler = _make_handler(cfg, backend, model_name)
    host, port = cfg.serving.host, cfg.serving.port
    server = HTTPServer((host, port), handler)
    print(f"[serve] listening on http://{host}:{port} (backend={cfg.backend})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server(load_config())
