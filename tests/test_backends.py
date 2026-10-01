"""Backend registry tests."""
import unittest

from engramforge.backends import get_backend, list_backends
from engramforge.config import Config


class TestBackends(unittest.TestCase):
    def test_list(self):
        self.assertIn("mock", list_backends())
        self.assertIn("cpu", list_backends())

    def test_mock_train(self):
        be = get_backend("mock", Config())
        s0 = be.current_skill()
        for _ in range(50):
            be.train_step(0)
        self.assertGreater(be.current_skill(), s0)

    def test_cpu_retrieve(self):
        be = get_backend("cpu", Config())
        v = be.retrieve([1, 2, 3, 4])
        self.assertEqual(len(v), Config().engram.embedding_dim)


if __name__ == "__main__":
    unittest.main()
