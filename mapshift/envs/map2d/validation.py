"""Validation and benchmark-health diagnostics for MapShift-2D."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from mapshift.metrics.statistics import NumericSummary, proportion_true, summarize_numeric

from .observation import observe_egocentric
from .state import Map2DEnvironment


@dataclass(frozen=True)
class EnvironmentDiagnostics:
    """Structured diagnostics for a single generated 2D environment."""

    environment_id: str
    motif_tag: str
    split_name: str
    node_count: int
    edge_count: int
    connected_component_count: int
    free_cell_count: int
    map_area_cells: int
    free_space_ratio: float
    start_goal_distance: float
    all_pairs_path_summary: NumericSummary
    bottleneck_edge_count: int
    bottleneck_max_delta: float
    landmark_count: int
    semantic_token_count: int
    visible_cell_fraction_from_start: float
    visible_landmark_count_from_start: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["all_pairs_path_summary"] = self.all_pairs_path_summary.to_dict()
        return payload


def _all_pairs_path_lengths(environment: Map2DEnvironment) -> list[float]:
    node_ids = sorted(environment.nodes)
    distances: list[float] = []
    for index, left in enumerate(node_ids):
        for right in node_ids[index + 1 :]:
            distance = environment.shortest_path_length(left, right)
            if distance is not None:
                distances.append(distance)
    return distances


def _bottleneck_stats(environment: Map2DEnvironment) -> tuple[int, float]:
    removable = environment.removable_edges()
    positive_deltas = [delta for _left, _right, delta in removable if delta > 0.0]
    if not positive_deltas:
        return 0, 0.0
    return len(positive_deltas), max(positive_deltas)


def analyze_map2d_environment(environment: Map2DEnvironment) -> EnvironmentDiagnostics:
    """Compute structured benchmark-health diagnostics for one environment."""

    observation = observe_egocentric(environment)
    visible_values = [cell for row in observation.geometry_patch for cell in row if cell >= 0]
    visible_free_fraction = proportion_true(cell == 1 for cell in visible_values)
    all_pairs = _all_pairs_path_lengths(environment)
    bottleneck_edge_count, bottleneck_max_delta = _bottleneck_stats(environment)

    return EnvironmentDiagnostics(
        environment_id=environment.environment_id,
        motif_tag=environment.motif_tag,
        split_name=environment.split_name,
        node_count=environment.node_count(),
        edge_count=environment.edge_count(),
        connected_component_count=len(environment.connected_components()),
        free_cell_count=environment.free_cell_count(),
        map_area_cells=environment.width_cells * environment.height_cells,
        free_space_ratio=environment.free_cell_count() / max(1, environment.width_cells * environment.height_cells),
        start_goal_distance=environment.shortest_path_length(environment.start_node_id, environment.goal_node_id) or 0.0,
        all_pairs_path_summary=summarize_numeric(all_pairs),
        bottleneck_edge_count=bottleneck_edge_count,
        bottleneck_max_delta=bottleneck_max_delta,
        landmark_count=len(environment.landmark_by_node),
        semantic_token_count=len(environment.goal_tokens),
        visible_cell_fraction_from_start=visible_free_fraction,
        visible_landmark_count_from_start=len(observation.visible_landmarks),
    )


def summarize_environment_diagnostics(diagnostics: Iterable[EnvironmentDiagnostics]) -> dict[str, Any]:
    """Aggregate environment diagnostics for release-level reporting."""

    items = list(diagnostics)
    if not items:
        return {
            "environment_count": 0,
            "motif_counts": {},
            "split_counts": {},
            "map_area_summary": summarize_numeric([]).to_dict(),
            "free_cell_count_summary": summarize_numeric([]).to_dict(),
            "connected_component_summary": summarize_numeric([]).to_dict(),
            "path_length_summary": summarize_numeric([]).to_dict(),
            "all_pairs_path_summary": summarize_numeric([]).to_dict(),
            "free_space_ratio_summary": summarize_numeric([]).to_dict(),
            "bottleneck_edge_summary": summarize_numeric([]).to_dict(),
            "bottleneck_max_delta_summary": summarize_numeric([]).to_dict(),
            "visible_cell_fraction_summary": summarize_numeric([]).to_dict(),
            "visible_landmark_summary": summarize_numeric([]).to_dict(),
            "semantic_token_distribution": {},
            "landmark_distribution": {},
        }

    motif_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    semantic_token_distribution: dict[str, int] = {}
    landmark_distribution: dict[str, int] = {}

    for diagnostic in items:
        motif_counts[diagnostic.motif_tag] = motif_counts.get(diagnostic.motif_tag, 0) + 1
        split_counts[diagnostic.split_name] = split_counts.get(diagnostic.split_name, 0) + 1
        semantic_token_distribution[str(diagnostic.semantic_token_count)] = semantic_token_distribution.get(str(diagnostic.semantic_token_count), 0) + 1
        landmark_distribution[str(diagnostic.landmark_count)] = landmark_distribution.get(str(diagnostic.landmark_count), 0) + 1

    return {
        "environment_count": len(items),
        "motif_counts": dict(sorted(motif_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "map_area_summary": summarize_numeric([item.map_area_cells for item in items]).to_dict(),
        "free_cell_count_summary": summarize_numeric([item.free_cell_count for item in items]).to_dict(),
        "connected_component_summary": summarize_numeric([item.connected_component_count for item in items]).to_dict(),
        "path_length_summary": summarize_numeric([item.start_goal_distance for item in items]).to_dict(),
        "all_pairs_path_summary": summarize_numeric([item.all_pairs_path_summary.mean for item in items if item.all_pairs_path_summary.count > 0]).to_dict(),
        "free_space_ratio_summary": summarize_numeric([item.free_space_ratio for item in items]).to_dict(),
        "bottleneck_edge_summary": summarize_numeric([item.bottleneck_edge_count for item in items]).to_dict(),
        "bottleneck_max_delta_summary": summarize_numeric([item.bottleneck_max_delta for item in items]).to_dict(),
        "visible_cell_fraction_summary": summarize_numeric([item.visible_cell_fraction_from_start for item in items]).to_dict(),
        "visible_landmark_summary": summarize_numeric([item.visible_landmark_count_from_start for item in items]).to_dict(),
        "semantic_token_distribution": dict(sorted(semantic_token_distribution.items())),
        "landmark_distribution": dict(sorted(landmark_distribution.items())),
    }


def validate_map2d_instance(environment: Any) -> list[str]:
    """Return a list of validation issues for a generated environment."""

    if environment is None:
        return ["environment is None"]
    if not isinstance(environment, Map2DEnvironment):
        return [f"unexpected environment type: {type(environment).__name__}"]

    issues: list[str] = []
    if environment.width_cells <= 0 or environment.height_cells <= 0:
        issues.append("environment dimensions must be positive")
    if len(environment.occupancy_grid) != environment.height_cells:
        issues.append("occupancy_grid height does not match height_cells")
    if any(len(row) != environment.width_cells for row in environment.occupancy_grid):
        issues.append("occupancy_grid width does not match width_cells")
    if not environment.nodes:
        issues.append("environment has no nodes")
    if environment.edge_count() == 0:
        issues.append("environment has no edges")
    if environment.start_node_id not in environment.nodes:
        issues.append("start_node_id is missing from nodes")
    if environment.goal_node_id not in environment.nodes:
        issues.append("goal_node_id is missing from nodes")
    if environment.free_cell_count() <= 0:
        issues.append("environment has no free cells")
    for node_id, node in environment.nodes.items():
        if not environment.is_free(node.cell):
            issues.append(f"node {node_id} is not on a free cell")
    for token, node_id in environment.goal_tokens.items():
        if node_id not in environment.nodes:
            issues.append(f"goal token {token} references missing node {node_id}")
    for node_id in environment.landmark_by_node:
        if node_id not in environment.nodes:
            issues.append(f"landmark references missing node {node_id}")
    if not environment.reachable(environment.start_node_id, environment.goal_node_id):
        issues.append("start and goal are not connected")
    serialized = environment.to_dict()
    replayed = Map2DEnvironment.from_dict(serialized)
    if replayed.to_dict() != serialized:
        issues.append("environment serialization is not stable under replay")
    return issues
