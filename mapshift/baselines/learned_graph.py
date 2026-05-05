"""Shared graph-data and planning helpers for learned MapShift baselines."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import torch

from mapshift.envs.map2d.observation import observe_egocentric
from mapshift.envs.map2d.state import AgentPose2D, Cell, Map2DEnvironment

from .common import cells_path_length, dynamics_cost_multiplier


def _edge_key(left: str, right: str) -> str:
    return "|".join(sorted((left, right)))


@dataclass(frozen=True)
class GraphTrainingData:
    node_order: tuple[str, ...]
    token_order: tuple[str, ...]
    node_features: torch.Tensor
    adjacency_matrix: torch.Tensor
    pair_index: torch.Tensor
    edge_labels: torch.Tensor
    geometry_cost_labels: torch.Tensor
    traversal_cost_labels: torch.Tensor
    token_labels: torch.Tensor
    max_geometry_cost: float
    max_traversal_cost: float
    global_features: tuple[float, ...]

    def to_device(self, device: torch.device | str) -> "GraphTrainingData":
        """Return a copy with tensor fields moved to the requested Torch device."""

        return GraphTrainingData(
            node_order=self.node_order,
            token_order=self.token_order,
            node_features=self.node_features.to(device),
            adjacency_matrix=self.adjacency_matrix.to(device),
            pair_index=self.pair_index.to(device),
            edge_labels=self.edge_labels.to(device),
            geometry_cost_labels=self.geometry_cost_labels.to(device),
            traversal_cost_labels=self.traversal_cost_labels.to(device),
            token_labels=self.token_labels.to(device),
            max_geometry_cost=self.max_geometry_cost,
            max_traversal_cost=self.max_traversal_cost,
            global_features=self.global_features,
        )


@dataclass(frozen=True)
class BeliefRouteResult:
    success: bool
    reached_goal: bool
    observed_length: float | None
    path_cells: tuple[Cell, ...]
    blocked_edges: tuple[tuple[str, str], ...]
    discovered_edges: tuple[tuple[str, str], ...]
    replans: int
    metadata: dict[str, Any]


def environment_global_features(environment: Map2DEnvironment) -> tuple[float, ...]:
    """Return normalized global environment features available to learned baselines."""

    total_cells = max(1, environment.width_cells * environment.height_cells)
    return (
        float(environment.geometry_scale),
        float(environment.observation_radius_m / max(environment.occupancy_resolution_m, 1e-6)),
        float(environment.dynamics.forward_gain),
        float(environment.dynamics.turn_gain),
        float(environment.dynamics.friction),
        float(environment.dynamics.inertial_lag),
        float(environment.dynamics.action_asymmetry),
        float(environment.dynamics.odometry_bias_deg / 90.0),
        float(environment.free_cell_count() / total_cells),
    )


def token_symbol(token: str) -> str:
    suffix = token.split("_", 1)[-1]
    for char in suffix:
        if char.isalpha():
            return char.upper()
    for char in token:
        if char.isalpha():
            return char.upper()
    return "?"


def semantic_label_for_node(environment: Map2DEnvironment, node_id: str) -> str:
    node = environment.nodes[node_id]
    frame = observe_egocentric(
        environment,
        pose=AgentPose2D(x=float(node.col), y=float(node.row), theta_deg=0.0),
    )
    if not frame.semantic_patch:
        return ""
    center_row = len(frame.semantic_patch) // 2
    center_col = len(frame.semantic_patch[center_row]) // 2
    return frame.semantic_patch[center_row][center_col]


def node_feature_vector(environment: Map2DEnvironment, node_id: str) -> list[float]:
    """Return a local observation and role feature vector for one anchor node."""

    node = environment.nodes[node_id]
    frame = observe_egocentric(
        environment,
        pose=AgentPose2D(x=float(node.col), y=float(node.row), theta_deg=0.0),
    )
    flat_geometry = [value for row in frame.geometry_patch for value in row if value >= 0]
    flat_semantics = [value for row in frame.semantic_patch for value in row if value]
    patch_area = max(1, len(flat_geometry))
    free_ratio = sum(1 for value in flat_geometry if value == 1) / patch_area
    blocked_ratio = sum(1 for value in flat_geometry if value == 0) / patch_area
    semantic_ratio = len(flat_semantics) / patch_area
    landmark_ratio = len(frame.visible_landmarks) / 4.0
    semantic_center = 1.0 if semantic_label_for_node(environment, node_id) else 0.0
    features = [
        node.row / max(1, environment.height_cells - 1),
        node.col / max(1, environment.width_cells - 1),
        free_ratio,
        blocked_ratio,
        semantic_ratio,
        landmark_ratio,
        1.0 if node_id == environment.start_node_id else 0.0,
        1.0 if node_id in environment.landmark_by_node else 0.0,
        semantic_center,
    ]
    features.extend(environment_global_features(environment))
    return features


def build_graph_training_data(
    environment: Map2DEnvironment,
    visited_node_ids: tuple[str, ...],
) -> GraphTrainingData:
    """Build node, edge, cost, and token supervision tensors for one environment."""

    node_order = tuple(sorted(set(visited_node_ids) | {environment.start_node_id, environment.goal_node_id}))
    if len(node_order) < 2:
        node_order = tuple(sorted(environment.nodes))[: max(2, len(environment.nodes))]
    token_order = tuple(sorted(environment.goal_tokens)) or ("__none__",)

    node_features = torch.tensor([node_feature_vector(environment, node_id) for node_id in node_order], dtype=torch.float32)
    adjacency_matrix = torch.tensor(
        [
            [
                1.0 if (other_id == node_id or other_id in environment.adjacency.get(node_id, [])) else 0.0
                for other_id in node_order
            ]
            for node_id in node_order
        ],
        dtype=torch.float32,
    )

    geometry_costs = [
        environment.shortest_path_length(left_id, right_id) or 0.0
        for left_id in node_order
        for right_id in node_order
        if left_id != right_id and environment.shortest_path_length(left_id, right_id) is not None
    ]
    traversal_costs = [value * dynamics_cost_multiplier(environment) for value in geometry_costs]
    max_geometry_cost = max([1.0] + geometry_costs)
    max_traversal_cost = max([1.0] + traversal_costs)

    pair_index: list[list[int]] = []
    edge_labels: list[float] = []
    geometry_cost_labels: list[float] = []
    traversal_cost_labels: list[float] = []
    for left_index, left_id in enumerate(node_order):
        for right_index, right_id in enumerate(node_order):
            if left_index == right_index:
                continue
            pair_index.append([left_index, right_index])
            edge_labels.append(1.0 if right_id in environment.adjacency.get(left_id, []) else 0.0)
            geometry_distance = environment.shortest_path_length(left_id, right_id)
            traversal_distance = None if geometry_distance is None else geometry_distance * dynamics_cost_multiplier(environment)
            geometry_cost_labels.append(1.5 if geometry_distance is None else float(geometry_distance / max_geometry_cost))
            traversal_cost_labels.append(1.5 if traversal_distance is None else float(traversal_distance / max_traversal_cost))

    token_labels: list[list[float]] = []
    for node_id in node_order:
        token_labels.append([1.0 if environment.goal_tokens.get(token) == node_id else 0.0 for token in token_order])

    return GraphTrainingData(
        node_order=node_order,
        token_order=token_order,
        node_features=node_features,
        adjacency_matrix=adjacency_matrix,
        pair_index=torch.tensor(pair_index, dtype=torch.long),
        edge_labels=torch.tensor(edge_labels, dtype=torch.float32),
        geometry_cost_labels=torch.tensor(geometry_cost_labels, dtype=torch.float32),
        traversal_cost_labels=torch.tensor(traversal_cost_labels, dtype=torch.float32),
        token_labels=torch.tensor(token_labels, dtype=torch.float32),
        max_geometry_cost=float(max_geometry_cost),
        max_traversal_cost=float(max_traversal_cost),
        global_features=environment_global_features(environment),
    )


def edge_probability_map(
    node_order: tuple[str, ...],
    pair_index: torch.Tensor,
    edge_logits: torch.Tensor,
) -> dict[tuple[str, str], float]:
    """Return predicted undirected edge probabilities keyed by node pair."""

    probabilities = torch.sigmoid(edge_logits.detach()).cpu().tolist()
    pair_map: dict[tuple[str, str], list[float]] = {}
    for (left_index, right_index), probability in zip(pair_index.detach().cpu().tolist(), probabilities):
        left_id = node_order[left_index]
        right_id = node_order[right_index]
        if left_id == right_id:
            continue
        pair_map.setdefault(tuple(sorted((left_id, right_id))), []).append(float(probability))
    return {pair: sum(values) / len(values) for pair, values in pair_map.items()}


def traversal_cost_map(
    node_order: tuple[str, ...],
    pair_index: torch.Tensor,
    traversal_cost_predictions: torch.Tensor,
    max_traversal_cost: float,
) -> dict[tuple[str, str], float]:
    """Return symmetric predicted traversal costs keyed by node pair."""

    values = traversal_cost_predictions.detach().cpu().tolist()
    pair_map: dict[tuple[str, str], list[float]] = {}
    for (left_index, right_index), value in zip(pair_index.detach().cpu().tolist(), values):
        left_id = node_order[left_index]
        right_id = node_order[right_index]
        if left_id == right_id:
            continue
        clipped = max(0.05, min(2.0, float(value)))
        pair_map.setdefault(tuple(sorted((left_id, right_id))), []).append(clipped * max_traversal_cost)
    return {pair: sum(values) / len(values) for pair, values in pair_map.items()}


def plan_on_predicted_graph(
    *,
    node_order: tuple[str, ...],
    start_node_id: str,
    goal_node_id: str,
    edge_probabilities: dict[tuple[str, str], float],
    traversal_costs: dict[tuple[str, str], float],
    edge_threshold: float,
) -> list[str] | None:
    """Run Dijkstra over the predicted graph."""

    adjacency: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in node_order}
    for pair, probability in edge_probabilities.items():
        if probability < edge_threshold:
            continue
        left_id, right_id = pair
        if left_id not in adjacency or right_id not in adjacency:
            continue
        cost = traversal_costs.get(pair, 1.0)
        adjacency[left_id].append((right_id, cost))
        adjacency[right_id].append((left_id, cost))

    if start_node_id not in adjacency or goal_node_id not in adjacency:
        return None

    frontier: list[tuple[float, str]] = [(0.0, start_node_id)]
    distances = {start_node_id: 0.0}
    parent: dict[str, str | None] = {start_node_id: None}
    while frontier:
        cost, node_id = heapq.heappop(frontier)
        if node_id == goal_node_id:
            break
        if cost > distances.get(node_id, float("inf")):
            continue
        for neighbor_id, edge_cost in adjacency.get(node_id, []):
            candidate = cost + edge_cost
            if candidate >= distances.get(neighbor_id, float("inf")):
                continue
            distances[neighbor_id] = candidate
            parent[neighbor_id] = node_id
            heapq.heappush(frontier, (candidate, neighbor_id))

    if goal_node_id not in parent:
        return None
    route = [goal_node_id]
    cursor = goal_node_id
    while parent[cursor] is not None:
        cursor = str(parent[cursor])
        route.append(cursor)
    route.reverse()
    return route


def execute_node_route(
    environment: Map2DEnvironment,
    node_route: list[str] | None,
) -> tuple[bool, float | None, tuple[Cell, ...]]:
    """Execute one predicted node route against the true environment adjacency."""

    if not node_route or len(node_route) < 2:
        return False, None, ()
    path_cells: list[Cell] = []
    total_geometry = 0.0
    for left_id, right_id in zip(node_route, node_route[1:]):
        if right_id not in environment.adjacency.get(left_id, []):
            return False, None, tuple(path_cells)
        segment = environment.oracle_shortest_path(left_id, right_id)
        segment_length = environment.shortest_path_length(left_id, right_id)
        if segment is None or segment_length is None:
            return False, None, tuple(path_cells)
        if path_cells and segment:
            path_cells.extend(segment[1:])
        else:
            path_cells.extend(segment)
        total_geometry += segment_length
    observed_length = total_geometry * dynamics_cost_multiplier(environment)
    return True, observed_length, tuple(path_cells)


def classify_family_from_losses(
    *,
    edge_ratio: float,
    token_ratio: float,
    geometry_ratio: float,
    traversal_ratio: float,
) -> str:
    """Map learned loss changes onto one intervention-family prediction."""

    family_scores = {
        "topology": edge_ratio,
        "semantic": token_ratio,
        "metric": geometry_ratio,
        "dynamics": max(0.0, traversal_ratio - geometry_ratio),
    }
    return max(sorted(family_scores), key=lambda family: family_scores[family])


def infer_goal_node_from_semantics(
    environment: Map2DEnvironment,
    candidate_node_ids: tuple[str, ...],
    goal_token: str | None,
) -> tuple[str | None, dict[str, Any]]:
    """Resolve a goal token from local semantic probes at known anchor nodes."""

    if not goal_token:
        return None, {"semantic_probe_used": False, "semantic_probe_candidates": []}
    symbol = token_symbol(goal_token)
    candidates = [
        node_id
        for node_id in candidate_node_ids
        if semantic_label_for_node(environment, node_id) == symbol
    ]
    return (
        (sorted(candidates)[0] if candidates else None),
        {
            "semantic_probe_used": True,
            "semantic_probe_symbol": symbol,
            "semantic_probe_candidates": sorted(candidates),
        },
    )


def execute_belief_route(
    *,
    base_environment: Map2DEnvironment,
    current_environment: Map2DEnvironment,
    node_order: tuple[str, ...],
    start_node_id: str,
    goal_node_id: str,
    edge_probabilities: dict[tuple[str, str], float],
    traversal_costs: dict[tuple[str, str], float],
    edge_threshold: float,
    blocked_edges: set[tuple[str, str]] | None = None,
    opened_corridors: dict[tuple[str, str], tuple[Cell, ...]] | None = None,
    max_replans: int = 8,
) -> BeliefRouteResult:
    """Execute a planned route using remembered corridors plus probed updates."""

    active_blocked = {tuple(sorted(edge)) for edge in (blocked_edges or set())}
    active_opened = {tuple(sorted(edge)): tuple(cells) for edge, cells in (opened_corridors or {}).items()}
    current_node = start_node_id
    traversed_cells: list[Cell] = [base_environment.resolve_location(start_node_id)]
    total_length = 0.0
    replans = 0
    attempted_edges: list[tuple[str, str]] = []

    def _edge_probabilities() -> dict[tuple[str, str], float]:
        adjusted = dict(edge_probabilities)
        for edge in active_blocked:
            adjusted[edge] = 0.0
        for edge in active_opened:
            adjusted[edge] = 1.0
        return adjusted

    def _edge_costs() -> dict[tuple[str, str], float]:
        adjusted = dict(traversal_costs)
        for edge, corridor in active_opened.items():
            corridor_length = cells_path_length(current_environment, corridor)
            if corridor_length is not None:
                adjusted[edge] = corridor_length * dynamics_cost_multiplier(current_environment)
        return adjusted

    while current_node != goal_node_id and replans <= max_replans:
        route = plan_on_predicted_graph(
            node_order=node_order,
            start_node_id=current_node,
            goal_node_id=goal_node_id,
            edge_probabilities=_edge_probabilities(),
            traversal_costs=_edge_costs(),
            edge_threshold=edge_threshold,
        )
        if not route or len(route) < 2:
            return BeliefRouteResult(
                success=False,
                reached_goal=False,
                observed_length=None if len(traversed_cells) <= 1 else total_length,
                path_cells=tuple(traversed_cells),
                blocked_edges=tuple(sorted(active_blocked)),
                discovered_edges=tuple(sorted(active_opened)),
                replans=replans,
                metadata={"attempted_edges": attempted_edges, "route_found": False},
            )

        advanced = False
        for next_node in route[1:]:
            edge = tuple(sorted((current_node, next_node)))
            attempted_edges.append(edge)
            corridor = active_opened.get(edge) or base_environment.corridor_for_edge(current_node, next_node)
            if not corridor or not current_environment.corridor_is_traversable(corridor):
                active_blocked.add(edge)
                replans += 1
                break
            corridor_length = cells_path_length(current_environment, corridor)
            if corridor_length is None:
                active_blocked.add(edge)
                replans += 1
                break
            if traversed_cells and corridor:
                traversed_cells.extend(list(corridor)[1:])
            else:
                traversed_cells.extend(corridor)
            total_length += corridor_length * dynamics_cost_multiplier(current_environment)
            current_node = next_node
            advanced = True
            if current_node == goal_node_id:
                return BeliefRouteResult(
                    success=True,
                    reached_goal=True,
                    observed_length=total_length,
                    path_cells=tuple(traversed_cells),
                    blocked_edges=tuple(sorted(active_blocked)),
                    discovered_edges=tuple(sorted(active_opened)),
                    replans=replans,
                    metadata={"attempted_edges": attempted_edges, "route_found": True},
                )
        if not advanced:
            continue

    return BeliefRouteResult(
        success=False,
        reached_goal=current_node == goal_node_id,
        observed_length=None if len(traversed_cells) <= 1 else total_length,
        path_cells=tuple(traversed_cells),
        blocked_edges=tuple(sorted(active_blocked)),
        discovered_edges=tuple(sorted(active_opened)),
        replans=replans,
        metadata={"attempted_edges": attempted_edges, "route_found": current_node == goal_node_id},
    )
