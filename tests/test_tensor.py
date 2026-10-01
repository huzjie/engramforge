"""Tensor core tests."""
import unittest

from engramforge.core.tensor import Tensor, zeros, ones, matmul, add, softmax, layernorm


class TestTensor(unittest.TestCase):
    def test_zeros_ones(self):
        z = zeros((2, 3))
        self.assertEqual(z.shape, (2, 3))
        self.assertEqual(sum(z.data), 0.0)
        o = ones((4,))
        self.assertEqual(sum(o.data), 4.0)

    def test_matmul(self):
        a = Tensor([1, 2, 3, 4], (2, 2))
        b = Tensor([1, 0, 0, 1], (2, 2))
        c = matmul(a, b)
        self.assertEqual(c.data, [1, 2, 3, 4])

    def test_add(self):
        a = Tensor([1, 2], (2,))
        b = Tensor([10, 20], (2,))
        self.assertEqual(add(a, b).data, [11, 22])

    def test_softmax(self):
        s = softmax(Tensor([1.0, 1.0], (1, 2)))
        self.assertAlmostEqual(sum(s.data), 1.0, places=6)

    def test_layernorm(self):
        ln = layernorm(Tensor([1.0, 3.0], (2,)))
        self.assertAlmostEqual(sum(ln.data), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
