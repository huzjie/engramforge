"""Engram module tests."""
import unittest

from engramforge.config import Config
from engramforge.memory.engram import EngramModule


class TestEngram(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.em = EngramModule(self.cfg.engram, self.cfg.model)

    def test_forward_dim(self):
        out = self.em.forward([1, 2, 3, 4])
        self.assertEqual(len(out), self.cfg.engram.embedding_dim)

    def test_gate(self):
        self.em.set_gate(1.0)
        out1 = self.em.forward([9, 9, 9])
        mem = self.em.retrieve([9, 9, 9])
        for a, b in zip(out1, mem):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
