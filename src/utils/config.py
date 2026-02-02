from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


import yaml

# data class schemas
@dataclass
class PathsConfig:
    project_root: str
    data_dir: str = "data"
    runs_dir: str = "runs"
    reports_dir: str = "reports"

@dataclass
class DataConfig:
    seed: int = 42
    n_paths: int = 50000
    n_steps: int = 252
    s0: float = 100.0
    k: float = 100.0
    t: float = 1.0
    r: float = 0.05
    sigma: float = 0.2
    option_type: str = "call"

@dataclass
class ModelConfig:
    hidden_dim: int = 128
    n_hidden_layers: int = 4
    activation: str = "tanh"

@dataclass
class TrainConfig:
    lr: float = 0.001
    epochs: float = 5000
    batch_size: float = 4096
    lambda_data: float = 1.0
    lambda_pde: float = 1.0
    lambda_bc: float = 1.0
    device: str = "cpu"

@dataclass
class EvalConfig:
    n_grid_s: int = 200
    n_grid_t: int = 50

@dataclass
class AppConfig:
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig

#--------------------------------------
# yaml helper
#--------------------------------------
#yaml loader/reader
def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML must be a mapping/dictionary: {path}")
    return obj

def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def find_project_root(
    start: Optional[Path] = None,
    markers: tuple[str, ...] = ("pyproject.toml", "configs", ".git"),
) -> Path:
    """
    Walk upward from `start` (default: cwd) until we find a directory that
    contains at least one of the marker files/folders.
    """
    cur = (start or Path.cwd()).resolve()

    for parent in (cur, *cur.parents):
        for m in markers:
            if (parent / m).exists():
                return parent

    # fallback: cwd (better than guessing src/)
    return cur

def load_config(
        project_root: Optional[Path] = None,
        *,
        data_yaml: str = "configs/data.yaml",
        model_yaml: str = "configs/model.yaml",
        train_yaml: str = "configs/train.yaml",
        eval_yaml: str = "configs/eval.yaml",
        override_yaml: Optional[Path] = None,
) -> AppConfig:
    """
        This will load YAML files and returns a typed AppConfig object.
    """
    pr = (project_root or find_project_root()).resolve()

    base = {
        "paths":{
            "project_root": str(pr),
            "data_dir": "data",
            "runs_dir": "runs",
            "reports_dir": "reports",
        },
        "data":{},
        "model":{},
        "train":{},
        "eval":{},
    }

    print(pr)
    cfg = _merge_dicts(base, {"data": _read_yaml(pr / data_yaml)})
    cfg = _merge_dicts(cfg, {"model": _read_yaml(pr / model_yaml)})
    cfg = _merge_dicts(cfg, {"train": _read_yaml(pr / train_yaml)})
    cfg = _merge_dicts(cfg, {"eval": _read_yaml(pr / eval_yaml)})

    if override_yaml:
        cfg = _merge_dicts(cfg, {"override_yaml", _read_yaml(pr/override_yaml)})

    return AppConfig(
        paths=PathsConfig(**cfg["paths"]),
        data=DataConfig(**cfg["data"]),
        model=ModelConfig(**cfg["model"]),
        train=TrainConfig(**cfg["train"]),
        eval=EvalConfig(**cfg["eval"]),
    )

