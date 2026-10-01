"""Memory bank tests."""
import unittest

from engramforge.memory.table import MemoryBank


class TestMemoryBank(unittest.TestCase):
    def test_insert_lookup(self):
        bank = MemoryBank(1024, 32, 4, offload=False)
        bank.insert((1, 2, 3))
        vec = bank.lookup((1, 2, 3))
        self.assertEqual(len(vec), 32)

    def test_stats(self):
        bank = MemoryBank(1024, 32, 4, offload=True)
        for i in range(100):
            bank.insert((i, i + 1))
        s = bank.stats()
        self.assertEqual(s["table_size"], 1024)


if __name__ == "__main__":
    unittest.main()
