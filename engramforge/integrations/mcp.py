"""MCP (Model Context Protocol) integration: tool + resource schema.

Exposes the Engram lookup as an MCP-style tool description so an MCP host can
call `engram_lookup` and read `engram://stats` as a resource.
"""


class EngramForgeMCP:
    def __init__(self, backend="cpu", cfg=None):
        from ..backends import get_backend
        self.backend = get_backend(backend, cfg)

    def tools(self):
        return [{
            "name": "engram_lookup",
            "description": "Retrieve the static n-gram memory for a token sequence via O(1) sparse lookup.",
            "inputSchema": {
                "type": "object",
                "properties": {"tokens": {"type": "array", "items": {"type": "integer"}}},
                "required": ["tokens"],
            },
        }]

    def call_tool(self, name, arguments):
        if name == "engram_lookup":
            tokens = arguments.get("tokens", [])
            vec = self.backend.retrieve(tokens) if hasattr(self.backend, "retrieve") else [0.0]
            return {"content": [{"type": "text", "text": f"lookup dim={len(vec)} sum={sum(vec):.4f}"}]}
        return {"error": f"unknown tool: {name}"}

    def resources(self):
        return [{"uri": "engram://stats", "name": "Engram memory-bank statistics",
                 "mimeType": "application/json"}]
