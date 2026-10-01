# 巨型嵌入表 host 卸载

确定性寻址使表无需常驻加速器。`MemoryBank(offload=True)` 模拟：只保留少量活跃缓存，其余按需从 host 取回并计数。

```python
from engramforge.bench.offload import run_offload
```
