# 贡献指南

1. 保持零依赖（标准库优先）。
2. 新模块补 `tests/` 单测。
3. 跑 `python -m compileall engramforge` 与 `python -m unittest discover -s tests -t .`。
