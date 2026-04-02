"""Shared Torch training helpers for learned MapShift baselines."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


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


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load one Torch checkpoint payload."""

    return torch.load(Path(path), map_location="cpu", weights_only=False)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts for one Torch model."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)
