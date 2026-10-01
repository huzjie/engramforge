"""Sparsity allocator tests."""
import unittest

from engramforge.memory.allocator import SparsityAllocator


class TestAllocator(unittest.TestCase):
    def test_ushaped_min(self):
        a = SparsityAllocator()
        r, l = a.best(n=200, optimal=0.35)
        self.assertAlmostEqual(r, 0.35, places=2)

    def test_optimum_decays_with_scale(self):
        a = SparsityAllocator()
        small = a.optimal_engram_ratio(1e3, 1e3)
        large = a.optimal_engram_ratio(1e15, 1e15)
        self.assertGreater(small, large)


if __name__ == "__main__":
    unittest.main()
