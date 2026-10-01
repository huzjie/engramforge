# memory.ngram — N-gram 编码

`NgramEncoder(order)`：`encode(tokens)` 枚举 order 1..N 全部 gram；`context_grams(tokens)` 只取以末 token 结尾的高阶 gram。
