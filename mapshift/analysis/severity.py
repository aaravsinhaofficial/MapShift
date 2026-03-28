"""Severity-response analysis scaffold."""

from __future__ import annotations

from typing import Sequence


def monotone_degradation(values: Sequence[float]) -> bool:
    """Check whether a sequence is monotonically non-increasing."""

    return all(left >= right for left, right in zip(values, values[1:]))
