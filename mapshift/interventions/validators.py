"""Validator hooks for intervention isolation checks."""

from __future__ import annotations

from typing import Iterable


def validate_intervention_invariants(issues: Iterable[str]) -> list[str]:
    """Normalize a collection of intervention validation issues."""

    return [issue for issue in issues if issue]
