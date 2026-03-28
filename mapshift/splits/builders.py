"""Split builder scaffolds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitBuildResult:
    split_name: str
    motif_count: int


def build_split_manifest(split_name: str, motifs: tuple[str, ...]) -> SplitBuildResult:
    """Return a minimal split build summary."""

    return SplitBuildResult(split_name=split_name, motif_count=len(motifs))
