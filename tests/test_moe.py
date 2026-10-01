"""MoE tests."""
import unittest

from engramforge.config import Config
from engramforge.moe.layer import MoELayer


class TestMoE(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.moe = MoELayer(self.cfg.moe, self.cfg.model)

    def test_route(self):
        topk, weights = self.moe.router.route("tok")
        self.assertEqual(len(topk), self.cfg.moe.top_k)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)

    def test_forward_dim(self):
        out = self.moe.forward("tok")
        self.assertEqual(len(out), self.cfg.model.hidden_size)


if __name__ == "__main__":
    unittest.main()
