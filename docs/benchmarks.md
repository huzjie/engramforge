# 四域基准

- knowledge：静态召回（Engram 查表）≈1.0 + skill 准确率
- reasoning / code / math：skill 驱动的动态准确率

```python
from engramforge.bench.run import run_all
report = run_all(cfg)   # 先训练再评估
```
