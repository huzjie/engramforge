"""Backend registry."""
from ..core.registry import Registry

BACKENDS = Registry("backend")


def register_backend(key, obj=None):
    if obj is None:
        def _decorator(o):
            BACKENDS.register(key, o)
            return o
        return _decorator
    BACKENDS.register(key, obj)
    return obj


def get_backend(key, cfg=None, **kwargs):
    cls = BACKENDS.get(key)
    if cls is None:
        raise ValueError(f"unknown backend: {key} (available: {list(BACKENDS.keys())})")
    return cls(cfg=cfg, **kwargs)


def list_backends():
    return BACKENDS.keys()


# import modules so their @register_backend decorators run
from . import mock, cpu, openai, vllm, transformers  # noqa: E402,F401

__all__ = ["BACKENDS", "register_backend", "get_backend", "list_backends"]
