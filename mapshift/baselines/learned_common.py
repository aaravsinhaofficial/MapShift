"""Shared Torch training helpers for learned MapShift baselines."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_torch_device(requested: str | None = None) -> torch.device:
    """Resolve a learned-baseline device request into a concrete Torch device."""

    env_value = os.environ.get("MAPSHIFT_TORCH_DEVICE")
    raw_value = env_value if env_value and str(requested or "auto").strip().lower() in {"", "auto"} else requested or "auto"
    value = str(raw_value).strip().lower()
    if value in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "gpu":
        value = "cuda:0"
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested torch_device={raw_value!r}, but torch.cuda.is_available() is false. "
            "Use torch_device='cpu' or install a CUDA-enabled PyTorch build."
        )
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported torch_device={raw_value!r}; expected 'auto', 'cpu', 'cuda', or 'cuda:N'.")
    return device


def set_torch_seed(seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and Torch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_path(
    *,
    checkpoint_dir: str | Path,
    baseline_name: str,
    environment_id: str,
    seed: int,
    parameters: dict[str, Any],
) -> Path:
    """Return a deterministic checkpoint path for one baseline/environment pair."""

    payload = json.dumps({"baseline_name": baseline_name, "seed": seed, "parameters": parameters}, sort_keys=True)
    signature = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    root = Path(checkpoint_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{baseline_name}-{environment_id}-{signature}.pt"


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist one Torch checkpoint payload."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load one Torch checkpoint payload."""

    return torch.load(Path(path), map_location=map_location, weights_only=False)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts for one Torch model."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)
