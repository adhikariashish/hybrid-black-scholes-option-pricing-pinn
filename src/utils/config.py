from __future__ import annotations

from dataclasses import dataclass, field
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
class OptionConfig:
    style: str = "european"   # european | american
    type: str = "call"        # call | put
    s_max: float = 300.0


@dataclass
class DirConfig:
    data: str = "data/processed/"


@dataclass
class DataConfig:
    option: OptionConfig = field(default_factory=OptionConfig)
    dir: DirConfig = field(default_factory=DirConfig)
    save_mc: bool = False
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
    in_dimension: int = 2
    out_dimension: int = 1

@dataclass
class PinnTrainConfig:
    n_interior: int = 20000
    n_terminal: int = 2000
    n_boundary: int = 2000
    resample_every: int = 100

@dataclass
class LossWeightConfig:
    pde: float = 1.0
    terminal: float = 10.0
    boundary: float = 1.0

@dataclass
class TrainConfig:
    lr: float = 0.001
    epochs: int = 5000
    device: str = "cpu"
    batch_size: int = 4096

    pinn: PinnTrainConfig = field(default_factory=PinnTrainConfig)
    weights: LossWeightConfig = field(default_factory=LossWeightConfig)

    log_every: int = 100
    save_every: int = 1000
    run_dir: str = "runs/pinn_v1"
    seed: int = 42

@dataclass
class EvalConfig:
    n_grid_s: int = 200
    n_grid_t: int = 50
    fig_dir: str = "reports/figures/"
    fig_folder_name: str = "pinn_v1"

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
        cfg = _merge_dicts(cfg, {"override", _read_yaml(pr/override_yaml)})

    # Convert nested dicts into nested dataclasses (important) ----
    data_dict = cfg["data"]

    if isinstance(data_dict.get("option"), dict):
        data_dict["option"] = OptionConfig(**data_dict["option"])

    if isinstance(data_dict.get("dir"), dict):
        data_dict["dir"] = DirConfig(**data_dict["dir"])

    train_dict = cfg["train"]
    if isinstance(train_dict.get("pinn"), dict):
        train_dict["pinn"] = PinnTrainConfig(**train_dict["pinn"])
    if isinstance(train_dict.get("weights"), dict):
        train_dict["weights"] = LossWeightConfig(**train_dict["weights"])

    return AppConfig(
        paths=PathsConfig(**cfg["paths"]),
        data=DataConfig(**data_dict),
        model=ModelConfig(**cfg["model"]),
        train=TrainConfig(**train_dict),
        eval=EvalConfig(**cfg["eval"]),
    )

