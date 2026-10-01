"""N-gram encoder tests."""
import unittest

from engramforge.memory.ngram import NgramEncoder


class TestNgram(unittest.TestCase):
    def test_order2(self):
        enc = NgramEncoder(2)
        grams = enc.encode([1, 2, 3])
        self.assertIn((1,), grams)
        self.assertIn((1, 2), grams)
        self.assertIn((2, 3), grams)
        self.assertNotIn((1, 2, 3), grams)

    def test_context(self):
        enc = NgramEncoder(3)
        ctx = enc.context_grams([1, 2, 3, 4])
        self.assertIn((4,), ctx)
        self.assertIn((3, 4), ctx)
        self.assertIn((2, 3, 4), ctx)


if __name__ == "__main__":
    unittest.main()
