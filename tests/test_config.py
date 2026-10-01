"""Config loading tests."""
import json
import os
import tempfile
import unittest

from engramforge.config import Config, load_config


class TestConfig(unittest.TestCase):
    def test_default(self):
        c = Config()
        self.assertEqual(c.backend, "mock")
        self.assertEqual(c.engram.ngram_order, 3)

    def test_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"backend": "cpu", "engram": {"ngram_order": 4}}, f)
            path = f.name
        c = load_config(path)
        self.assertEqual(c.backend, "cpu")
        self.assertEqual(c.engram.ngram_order, 4)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
