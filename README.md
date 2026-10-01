# engramforge

> 条件记忆稀疏大模型训练与推理框架 —— DeepSeek Engram 方向（Conditional Memory via Scalable Lookup）

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](./LICENSE)
[![Zero-dependency](https://img.shields.io/badge/dependencies-zero-brightgreen)]()

**一句话**：把 MoE 的「条件计算」补上一条正交的「条件记忆」轴——静态知识（实体、公式化表达）用 O(1) 稀疏查表直接取回，不再让早期注意力/FFN 层用昂贵计算去「重建一个静态查表」。

## 为什么要做这个

大模型语言建模里有两类截然不同的子任务：

| 子任务 | 性质 | 理想载体 |
|---|---|---|
| 组合推理（compositional reasoning） | 动态、深层、需计算 | MoE 条件计算 |
| 知识检索（knowledge retrieval） | 静态、局部、刻板 | Engram 条件记忆 |

Transformer 缺乏原生的知识查表原语，导致解析一个多词元实体要烧掉好几个早期注意力层+FFN——本质是「用计算在运行时重建一张静态表」。Engram 把经典 N-gram 嵌入现代化成可学习的静态记忆，用**确定性寻址**实现 O(1) 查表。

## 核心能力

- **Engram 模块**：确定性 n-gram 寻址（多哈希 + 双哈希）+ 门控融合到动态 hidden states
- **MemoryBank**：静态嵌入表 + host memory offload（确定性寻址使巨型表可卸载，推理开销极小）
- **MoE 主干**：Top-K 路由 + SwiGLU 专家 + 负载均衡（条件计算）
- **稀疏分配（Sparsity Allocation）**：U 型 scaling law，指导 MoE vs Engram 的最优容量切分
- **五后端**：mock（可训练、确定性）/ cpu / openai / vllm / transformers
- **四域基准**：知识 / 推理 / 代码 / 数学
- **零依赖**：纯 Python 标准库，OpenAI 兼容 HTTP 服务、CLI、Docker/K8s/Helm、CI、LangChain/MCP

## 快速开始

```bash
# 自检
python -m engramforge doctor

# 训练可训练后端（skill 单调爬到 1.0）
python -m engramforge train

# 跑四域基准
python -m engramforge bench

# 拟合 U 型 scaling law
python -m engramforge scaling

# 启动 OpenAI 兼容服务
python -m engramforge serve
```

## 技术拆解（干货）

详见 [`介绍.md`](./介绍.md) 和 [`docs/`](./docs/)：

- [`docs/conditional-memory.md`](./docs/conditional-memory.md) — 条件记忆的原理与设计
- [`docs/sparsity-allocation.md`](./docs/sparsity-allocation.md) — U 型 scaling law
- [`docs/offload.md`](./docs/offload.md) — 巨型嵌入表 host 卸载
- [`docs/prompts.md`](./docs/prompts.md) — 可复用提示词/工作流模板
- [`docs/recipes.md`](./docs/recipes.md) — 工程技巧与踩坑

## 目录结构

```
engramforge/
  core/        # 零依赖张量 + dtype/registry
  memory/      # Engram：hash/ngram/table/engram/allocator
  moe/         # 条件计算：router/expert/layer
  model/       # EngramTransformer + 稀疏块
  train/       # 数据 + 训练 + scaling law
  backends/    # mock/cpu/openai/vllm/transformers
  bench/       # knowledge/reasoning/code/math + offload + analysis
  serving/     # OpenAI 兼容 stdlib HTTP
  integrations/# LangChain + MCP
configs/       # default.yaml / engram-27b.json / engram-small.json
examples/      # 10 个可运行示例
tests/         # 单元测试
deploy/        # Docker / K8s / Helm
```

## License

Apache 2.0。
