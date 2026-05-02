"""Structured-dynamics learned baseline for MapShift-2D."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .api import register_baseline
from .learned_base import LearnedGraphBaseline
from .learned_common import count_parameters
from .learned_graph import GraphTrainingData


class _StructuredDynamicsWorldModel(nn.Module):
    """Factorized graph model with separate geometry and dynamics factors."""

    def __init__(self, input_dim: int, geometry_width: int, dynamics_width: int, token_count: int) -> None:
        super().__init__()
        self.geometry_encoder = nn.Sequential(
            nn.Linear(input_dim, geometry_width),
            nn.Tanh(),
            nn.Linear(geometry_width, geometry_width),
            nn.Tanh(),
        )
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(9, dynamics_width),
            nn.Tanh(),
            nn.Linear(dynamics_width, dynamics_width),
            nn.Tanh(),
        )
        self.edge_head = nn.Linear(geometry_width * 2, 1)
        self.geometry_head = nn.Linear(geometry_width * 2, 1)
        self.traversal_head = nn.Linear((geometry_width * 2) + dynamics_width, 1)
        self.token_head = nn.Linear(geometry_width, token_count)

    def forward(self, graph_data: GraphTrainingData) -> dict[str, torch.Tensor]:
        geometry_hidden = self.geometry_encoder(graph_data.node_features)
        global_features = torch.tensor(
            graph_data.global_features,
            dtype=graph_data.node_features.dtype,
            device=graph_data.node_features.device,
        )
        dynamics_hidden = self.dynamics_encoder(global_features).unsqueeze(0)
        left = geometry_hidden[graph_data.pair_index[:, 0]]
        right = geometry_hidden[graph_data.pair_index[:, 1]]
        pair_geometry = torch.cat([left, right], dim=-1)
        pair_dynamics = dynamics_hidden.expand(pair_geometry.shape[0], -1)
        return {
            "node_hidden": geometry_hidden,
            "edge_logits": self.edge_head(pair_geometry).squeeze(-1),
            "geometry_costs": self.geometry_head(pair_geometry).squeeze(-1),
            "traversal_costs": self.traversal_head(torch.cat([pair_geometry, pair_dynamics], dim=-1)).squeeze(-1),
            "token_logits": self.token_head(geometry_hidden),
        }


class StructuredDynamicsBaseline(LearnedGraphBaseline):
    """Learned graph baseline that factorizes geometry, traversal, and dynamics."""

    name = "structured_dynamics_world_model"
    category = "structured_dynamics"

    def __init__(self, run_config: Any) -> None:
        super().__init__(run_config)
        self.geometry_width = max(6, int(self.parameters.get("geometry_width", 10)))
        self.dynamics_width = max(4, int(self.parameters.get("dynamics_width", 6)))
        self.parameter_count = (self.geometry_width * 32) + (self.dynamics_width * 16)
        self.trainable_parameter_count = self.parameter_count

    @property
    def model_class(self) -> type[nn.Module]:
        return _StructuredDynamicsWorldModel

    def build_model(self, graph_data: GraphTrainingData) -> nn.Module:
        model = _StructuredDynamicsWorldModel(
            input_dim=int(graph_data.node_features.shape[-1]),
            geometry_width=self.geometry_width,
            dynamics_width=self.dynamics_width,
            token_count=max(1, len(graph_data.token_order)),
        )
        self.parameter_count, self.trainable_parameter_count = count_parameters(model)
        return model

    def forward_outputs(self, model: nn.Module, graph_data: GraphTrainingData) -> dict[str, torch.Tensor]:
        return model(graph_data)


register_baseline(StructuredDynamicsBaseline.name, StructuredDynamicsBaseline)
