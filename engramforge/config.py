"""Configuration loading with a zero-dependency YAML fallback."""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class EngramConfig:
    ngram_order: int = 3
    table_size: int = 65536
    embedding_dim: int = 256
    num_hashes: int = 4
    hashing: str = "deterministic"
    offload: bool = True
    fusion: str = "gated"
    gate_init: float = 0.5
    fp8_quant: bool = False


@dataclass
class MoeConfig:
    num_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 1024
    aux_loss_coef: float = 0.01
    capacity_factor: float = 1.25


@dataclass
class ModelConfig:
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    head_dim: int = 64
    vocab_size: int = 8192
    engram_layers: int = 3
    moe_layers: int = 9
    max_seq_len: int = 2048


@dataclass
class TrainConfig:
    lr: float = 3e-4
    warmup_steps: int = 100
    max_steps: int = 2000
    batch_size: int = 8
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    scheduler: str = "cosine"


@dataclass
class DataConfig:
    root: str = "data/corpus"
    num_docs: int = 1000
    ngram_order: int = 3
    dedup: bool = True
    synth_seed: int = 42


@dataclass
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    engram: EngramConfig = field(default_factory=EngramConfig)
    moe: MoeConfig = field(default_factory=MoeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    backend: str = "mock"
    output_dir: str = "outputs"
    seed: int = 42


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _defaults() -> dict:
    return {
        "engram": asdict(EngramConfig()),
        "moe": asdict(MoeConfig()),
        "model": asdict(ModelConfig()),
        "train": asdict(TrainConfig()),
        "data": asdict(DataConfig()),
        "serving": asdict(ServingConfig()),
        "backend": "mock",
        "output_dir": "outputs",
        "seed": 42,
    }


def _build(cfg: dict) -> Config:
    merged = _merge(_defaults(), cfg)
    return Config(
        engram=EngramConfig(**merged["engram"]),
        moe=MoeConfig(**merged["moe"]),
        model=ModelConfig(**merged["model"]),
        train=TrainConfig(**merged["train"]),
        data=DataConfig(**merged["data"]),
        serving=ServingConfig(**merged["serving"]),
        backend=merged.get("backend", "mock"),
        output_dir=merged.get("output_dir", "outputs"),
        seed=merged.get("seed", 42),
    )


def load_config(path: Optional[str] = None) -> Config:
    if not path or not os.path.exists(path):
        return Config()
    data = None
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except ImportError:
            from .utils.yamlish import parse
            data = parse(path)
    if data is None:
        data = {}
    return _build(data)
