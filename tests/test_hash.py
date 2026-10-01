"""Deterministic addressing tests."""
import unittest

from engramforge.memory.hash import DeterministicHasher, key_of


class TestHasher(unittest.TestCase):
    def test_deterministic(self):
        h = DeterministicHasher(65536, 4)
        self.assertEqual(h.slots("a"), h.slots("a"))
        self.assertNotEqual(h.slots("a"), h.slots("b"))

    def test_range(self):
        h = DeterministicHasher(100, 4)
        for s in h.slots("xyz"):
            self.assertTrue(0 <= s < 100)

    def test_key_of(self):
        self.assertEqual(key_of((1, 2, 3)), "1:2:3")


if __name__ == "__main__":
    unittest.main()
