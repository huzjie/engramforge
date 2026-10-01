"""OpenAI-compatible request/response models (pure Python)."""
import json
import time


def chat_completion(model, prompt, backend, message_id=None):
    """Build an OpenAI /v1/chat/completions response payload."""
    answer = "yes" if backend.generate(prompt) else "no"
    token = 1 if answer == "yes" else 0
    return {
        "id": f"chatcmpl-{message_id or int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 1,
                  "total_tokens": len(prompt.split()) + 1},
    }
