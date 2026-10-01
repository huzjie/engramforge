# 合成语料

`SyntheticCorpus` 生成四域（knowledge/reasoning/code/math）确定性语料，使全链路零依赖可跑。

```python
from engramforge.train.data import SyntheticCorpus
corpus = SyntheticCorpus(num_docs=1000)
docs = corpus.docs()
```
