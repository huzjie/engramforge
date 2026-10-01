# 稀疏分配与 U 型 scaling law

给定固定预算，损失对 Engram 占比是 U 型。最优点随规模漂移（预算越大，最优 Engram 占比越低）。

```python
from engramforge.memory.allocator import SparsityAllocator
a = SparsityAllocator()
a.sweep()           # 损失曲线
a.optimal_engram_ratio(1e12, 1e12)
```
