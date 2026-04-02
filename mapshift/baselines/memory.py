"""Persistent-memory learned baseline for MapShift-2D."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .api import register_baseline
from .learned_common import count_parameters
from .learned_base import LearnedGraphBaseline
from .learned_graph import GraphTrainingData


class _MemoryGraphWorldModel(nn.Module):
    def __init__(self, input_dim: int, slot_dim: int, memory_slots: int, token_count: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, slot_dim)
        self.memory_slots = nn.Parameter(torch.randn(memory_slots, slot_dim) * 0.1)
        self.edge_head = nn.Linear(slot_dim * 2, 1)
        self.geometry_head = nn.Linear(slot_dim * 2, 1)
        self.traversal_head = nn.Linear(slot_dim * 2, 1)
        self.token_head = nn.Linear(slot_dim, token_count)

    def forward(self, graph_data: GraphTrainingData) -> dict[str, torch.Tensor]:
        encoded = torch.tanh(self.encoder(graph_data.node_features))
        scale = max(1, encoded.shape[-1]) ** 0.5
        attention = torch.softmax(torch.matmul(encoded, self.memory_slots.transpose(0, 1)) / scale, dim=-1)
        node_hidden = torch.matmul(attention, self.memory_slots)
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


class PersistentMemoryBaseline(LearnedGraphBaseline):
    """Memory-slot world model trained on exploration-time node observations."""

    name = "persistent_memory_world_model"
    category = "memory_augmented"

    def __init__(self, run_config: Any) -> None:
        super().__init__(run_config)
        self.memory_slots = max(4, int(self.parameters.get("memory_slots", 16)))
        self.readout_width = max(4, int(self.parameters.get("readout_width", 8)))
        self.parameter_count = (self.memory_slots * self.readout_width * 20) + (self.readout_width * 10)
        self.trainable_parameter_count = self.parameter_count

    @property
    def model_class(self) -> type[nn.Module]:
        return _MemoryGraphWorldModel

    def build_model(self, graph_data: GraphTrainingData) -> nn.Module:
        model = _MemoryGraphWorldModel(
            input_dim=int(graph_data.node_features.shape[-1]),
            slot_dim=self.readout_width,
            memory_slots=self.memory_slots,
            token_count=max(1, len(graph_data.token_order)),
        )
        self.parameter_count, self.trainable_parameter_count = count_parameters(model)
        return model

    def forward_outputs(self, model: nn.Module, graph_data: GraphTrainingData) -> dict[str, torch.Tensor]:
        return model(graph_data)


register_baseline(PersistentMemoryBaseline.name, PersistentMemoryBaseline)
