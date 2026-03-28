"""Utilities for deterministic benchmark seeding."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SeedState:
    """Recorded random seed state for a benchmark run."""

    python_random_seed: int
    numpy_seed: Optional[int] = None
    env_seed: Optional[int] = None


def seed_everything(seed: int) -> SeedState:
    """Seed Python and, when available, NumPy."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    numpy_seed: Optional[int] = None
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        numpy_seed = seed
    except Exception:
        numpy_seed = None

    return SeedState(python_random_seed=seed, numpy_seed=numpy_seed, env_seed=seed)
