# 训练

可训练 mock 后端：skill 单调爬升到 1.0，loss 同步下降。

```python
from engramforge.backends import get_backend
be = get_backend("mock", cfg)
for step in range(2000):
    be.train_step(step)
```
