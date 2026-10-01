"""Serving layer tests (no socket needed)."""
import unittest

from engramforge.config import Config
from engramforge.backends import get_backend
from engramforge.serving.openai import chat_completion


class TestServing(unittest.TestCase):
    def test_chat_completion(self):
        be = get_backend("mock", Config())
        resp = chat_completion("engramforge-27b", "hello", be)
        self.assertEqual(resp["object"], "chat.completion")
        self.assertEqual(len(resp["choices"]), 1)


if __name__ == "__main__":
    unittest.main()
