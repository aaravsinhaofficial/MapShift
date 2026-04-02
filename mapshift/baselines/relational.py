"""Relational graph learned baseline for MapShift-2D."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .api import register_baseline
from .learned_common import count_parameters
from .learned_base import LearnedGraphBaseline
from .learned_graph import GraphTrainingData


class _RelationalGraphWorldModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, token_count: int, message_passing_steps: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_size)
        self.message = nn.Linear(hidden_size, hidden_size)
        self.edge_head = nn.Linear(hidden_size * 2, 1)
        self.geometry_head = nn.Linear(hidden_size * 2, 1)
        self.traversal_head = nn.Linear(hidden_size * 2, 1)
        self.token_head = nn.Linear(hidden_size, token_count)
        self.message_passing_steps = message_passing_steps

    def forward(self, graph_data: GraphTrainingData) -> dict[str, torch.Tensor]:
        node_hidden = torch.tanh(self.encoder(graph_data.node_features))
        degree = graph_data.adjacency_matrix.sum(dim=-1, keepdim=True).clamp(min=1.0)
        norm_adj = graph_data.adjacency_matrix / degree
        for _ in range(self.message_passing_steps):
            node_hidden = torch.tanh(self.message(node_hidden) + torch.matmul(norm_adj, node_hidden))
        left = node_hidden[graph_data.pair_index[:, 0]]
        right = node_hidden[graph_data.pair_index[:, 1]]
        pair_hidden = torch.cat([left, right], dim=-1)
        return {
            "node_hidden": node_hidden,
            "edge_logits": self.edge_head(pair_hidden).squeeze(-1),
            "geometry_costs": self.geometry_head(pair_hidden).squeeze(-1),
            "traversal_costs": self.traversal_head(pair_hidden).squeeze(-1),
            "token_logits": self.token_head(node_hidden),
        }


class RelationalGraphBaseline(LearnedGraphBaseline):
    """Relational world model trained on node observations and explored graph structure."""

    name = "relational_graph_world_model"
    category = "relational"

    def __init__(self, run_config: Any) -> None:
        super().__init__(run_config)
        self.hidden_size = max(6, int(self.parameters.get("hidden_size", 10)))
        self.message_passing_steps = max(1, int(self.parameters.get("message_passing_steps", 2)))
        self.parameter_count = (self.hidden_size * self.hidden_size * self.message_passing_steps) + (self.hidden_size * 20) + (self.hidden_size * 5)
        self.trainable_parameter_count = self.parameter_count

    @property
    def model_class(self) -> type[nn.Module]:
        return _RelationalGraphWorldModel

    def build_model(self, graph_data: GraphTrainingData) -> nn.Module:
        model = _RelationalGraphWorldModel(
            input_dim=int(graph_data.node_features.shape[-1]),
            hidden_size=self.hidden_size,
            token_count=max(1, len(graph_data.token_order)),
            message_passing_steps=self.message_passing_steps,
        )
        self.parameter_count, self.trainable_parameter_count = count_parameters(model)
        return model

    def forward_outputs(self, model: nn.Module, graph_data: GraphTrainingData) -> dict[str, torch.Tensor]:
        return model(graph_data)


register_baseline(RelationalGraphBaseline.name, RelationalGraphBaseline)
