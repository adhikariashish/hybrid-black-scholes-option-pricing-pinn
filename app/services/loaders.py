# app/services/loaders.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, Tuple

import streamlit as st
import torch

from src.models.pricing_pinn import PricingPINN  # <-- adjust import to your actual location


@dataclass(frozen=True)
class RunArtifacts:
    cfg: Dict[str, Any]
    ckpt_path: Path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_paths(run_dir: str, option_type: str) -> RunArtifacts:
    """
    Resolves:
      runs/<run>/config_<opt>.json
      runs/<run>/checkpoint_<opt>.pt
    """
    run = Path(run_dir)

    cfg_path = run / f"config_{option_type}.json"
    ckpt_path = run / f"checkpoint_{option_type}.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {cfg_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    cfg = _load_json(cfg_path)
    return RunArtifacts(cfg=cfg, ckpt_path=ckpt_path)


def _model_from_cfg(cfg: Dict[str, Any]) -> PricingPINN:
    m = cfg.get("model", {})
    model = PricingPINN(
        in_dimension=int(m.get("in_dimension", 2)),
        hidden_sizes=tuple([int(m.get("hidden_dim", 128))] * int(m.get("n_hidden_layers", 4))),
        activation=str(m.get("activation", "tanh")),
        out_dimension=int(m.get("out_dimension", 1)),
    )
    return model


@st.cache_resource(show_spinner=False)
def load_model_cached(run_dir: str, option_type: str) -> Tuple[PricingPINN, Dict[str, Any]]:
    """
    Returns a (model, cfg) tuple.
    Cached by run_dir + option_type so toggling call/put is instant after first load.
    """

    artifacts = resolve_paths(run_dir, option_type)
    cfg = artifacts.cfg
    ckpt_path = artifacts.ckpt_path

    # prefer cfg device, but fallback safely
    device_str = str(cfg.get("train", {}).get("device", "cpu")).lower().strip()

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"

    device = torch.device(device_str)

    # build model
    model = _model_from_cfg(cfg).to(device)
    model.eval()

    payload = torch.load(ckpt_path, map_location=device)

    # Your training saved: {"model_state": model.state_dict(), "model_cfg":..., ...}
    if "model_state" in payload:
        model.load_state_dict(payload["model_state"])
    elif "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
    else:
        # last resort (if you saved directly state_dict)
        if isinstance(payload, dict) and all(isinstance(k, str) for k in payload.keys()):
            try:
                model.load_state_dict(payload)  # type: ignore[arg-type]
            except Exception as e:
                raise KeyError("Checkpoint payload does not contain model_state/model_state_dict and is not a raw state_dict.") from e
        raise KeyError("Checkpoint payload missing model_state/model_state_dict.")

    return model, cfg


def load_history_safe(run_dir: str, option_type: str) -> Dict[str, Any] | None:
    """
    Used by Diagnostics tab.
    """
    run = Path(run_dir)
    candidates = [
        run / f"history_{option_type}.json",
        run / "history.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None
