"""Leakage check helpers."""

from __future__ import annotations


def disjoint_overlap(left: set[str], right: set[str]) -> set[str]:
    """Return the overlap between two sets."""

    return left & right
