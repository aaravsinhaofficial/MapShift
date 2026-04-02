"""Monolithic recurrent learned baseline for MapShift-2D."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .api import register_baseline
from .learned_common import count_parameters
from .learned_base import LearnedGraphBaseline
from .learned_graph import GraphTrainingData


class _RecurrentGraphWorldModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, token_count: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.edge_head = nn.Linear(hidden_size * 2, 1)
        self.geometry_head = nn.Linear(hidden_size * 2, 1)
        self.traversal_head = nn.Linear(hidden_size * 2, 1)
        self.token_head = nn.Linear(hidden_size, token_count)

    def forward(self, graph_data: GraphTrainingData) -> dict[str, torch.Tensor]:
        sequence = graph_data.node_features.unsqueeze(0)
        outputs, _hidden = self.gru(sequence)
        node_hidden = outputs.squeeze(0)
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


class MonolithicRecurrentBaseline(LearnedGraphBaseline):
    """Recurrent world model trained on exploration-time node observations."""

    name = "monolithic_recurrent_world_model"
    category = "recurrent"

    def __init__(self, run_config: Any) -> None:
        super().__init__(run_config)
        self.hidden_size = int(self.parameters.get("hidden_size", 12))
        self.parameter_count = (self.hidden_size * self.hidden_size) + (self.hidden_size * 20) + (self.hidden_size * 5)
        self.trainable_parameter_count = self.parameter_count

    @property
    def model_class(self) -> type[nn.Module]:
        return _RecurrentGraphWorldModel

    def build_model(self, graph_data: GraphTrainingData) -> nn.Module:
        model = _RecurrentGraphWorldModel(
            input_dim=int(graph_data.node_features.shape[-1]),
            hidden_size=self.hidden_size,
            token_count=max(1, len(graph_data.token_order)),
        )
        self.parameter_count, self.trainable_parameter_count = count_parameters(model)
        return model

    def forward_outputs(self, model: nn.Module, graph_data: GraphTrainingData) -> dict[str, torch.Tensor]:
        return model(graph_data)


register_baseline(MonolithicRecurrentBaseline.name, MonolithicRecurrentBaseline)
