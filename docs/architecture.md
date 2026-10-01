# 架构总览

engramforge 由三层构成：

## 1. 记忆层（memory/）
- `hash.py` — 确定性寻址（多哈希 + 双哈希）
- `ngram.py` — N-gram 编码器
- `table.py` — 静态嵌入 MemoryBank + offload
- `engram.py` — EngramModule（查表 + 门控融合）
- `allocator.py` — U 型稀疏分配 scaling law

## 2. 计算层（moe/ + model/）
- `moe/router.py` — Top-K 门控
- `moe/expert.py` — SwiGLU 专家
- `moe/layer.py` — MoE 层
- `model/block.py` — 稀疏块（早期挂 Engram，后期走 MoE）
- `model/transformer.py` — EngramTransformer

## 3. 工程层（train/ + backends/ + bench/ + serving/ + integrations/）
- 训练、五后端、四域基准、OpenAI 兼容服务、LangChain/MCP。

## 数据流

```
tokens -> NgramEncoder -> DeterministicHasher -> MemoryBank.lookup -> gate * mem + (1-gate) * hidden
                                                                    (early layers)
tokens -> TopKRouter -> SwiGLUExpert (top-k) -> weighted combine      (later layers)
```
