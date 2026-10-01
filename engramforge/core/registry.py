"""A tiny named registry used across the framework (backends, benchmarks)."""


class Registry:
    def __init__(self, name):
        self.name = name
        self._items = {}

    def register(self, key, obj=None):
        if obj is None:
            def _decorator(o):
                self._items[key] = o
                return o
            return _decorator
        self._items[key] = obj
        return obj

    def get(self, key):
        return self._items.get(key)

    def keys(self):
        return list(self._items.keys())

    def __contains__(self, key):
        return key in self._items

    def __len__(self):
        return len(self._items)
