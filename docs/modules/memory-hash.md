# memory.hash — 确定性寻址

`DeterministicHasher(table_size, num_hashes)`：k 个哈希函数，读时取 k 槽平均，写时双哈希解决碰撞。

```python
from engramforge.memory.hash import DeterministicHasher, key_of
h = DeterministicHasher(65536, 4)
h.slots((1, 2, 3))   # 4 个候选槽
h.probe((1, 2, 3), 0, 1)  # 双哈希探测
```
