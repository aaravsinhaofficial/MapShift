"""Pretrained structured graph world-model baseline for MapShift-2D."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mapshift.core.schemas import load_release_bundle
from mapshift.envs.map2d.generator import Map2DGenerator

from .api import deterministic_exploration_trace, register_baseline
from .learned_base import LearnedGraphBaseline
from .learned_common import checkpoint_path, count_parameters, load_checkpoint, save_checkpoint, set_torch_seed
from .learned_graph import GraphTrainingData, build_graph_training_data


LOGGER = logging.getLogger(__name__)


class _PretrainedStructuredGraphWorldModel(nn.Module):
    """Larger graph model with message passing and factorized traversal dynamics."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        token_count: int,
        message_passing_steps: int,
        pair_width: int,
        dynamics_width: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_size)
        self.self_layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(message_passing_steps))
        self.message_layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(message_passing_steps))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(message_passing_steps))
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(9, dynamics_width),
            nn.Tanh(),
            nn.Linear(dynamics_width, dynamics_width),
            nn.Tanh(),
        )
        self.pair_trunk = nn.Sequential(
            nn.Linear(hidden_size * 2, pair_width),
            nn.Tanh(),
        )
        self.dynamics_pair_trunk = nn.Sequential(
            nn.Linear((hidden_size * 2) + dynamics_width, pair_width),
            nn.Tanh(),
        )
        self.edge_head = nn.Linear(pair_width, 1)
        self.geometry_head = nn.Linear(pair_width, 1)
        self.traversal_head = nn.Linear(pair_width, 1)
        self.token_head = nn.Sequential(
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.Tanh(),
            nn.Linear(max(16, hidden_size // 2), token_count),
        )

    def forward(self, graph_data: GraphTrainingData) -> dict[str, torch.Tensor]:
        node_hidden = torch.tanh(self.encoder(graph_data.node_features))
        degree = graph_data.adjacency_matrix.sum(dim=-1, keepdim=True).clamp(min=1.0)
        norm_adj = graph_data.adjacency_matrix / degree
        for self_layer, message_layer, norm in zip(self.self_layers, self.message_layers, self.norms):
            messages = torch.matmul(norm_adj, node_hidden)
            update = torch.tanh(self_layer(node_hidden) + message_layer(messages))
            node_hidden = norm(node_hidden + update)

        left = node_hidden[graph_data.pair_index[:, 0]]
        right = node_hidden[graph_data.pair_index[:, 1]]
        pair_hidden = torch.cat([left, right], dim=-1)
        pair_features = self.pair_trunk(pair_hidden)
        global_features = torch.tensor(
            graph_data.global_features,
            dtype=graph_data.node_features.dtype,
            device=graph_data.node_features.device,
        )
        dynamics_hidden = self.dynamics_encoder(global_features).unsqueeze(0).expand(pair_hidden.shape[0], -1)
        dynamics_pair_features = self.dynamics_pair_trunk(torch.cat([pair_hidden, dynamics_hidden], dim=-1))
        return {
            "node_hidden": node_hidden,
            "edge_logits": self.edge_head(pair_features).squeeze(-1),
            "geometry_costs": self.geometry_head(pair_features).squeeze(-1),
            "traversal_costs": self.traversal_head(dynamics_pair_features).squeeze(-1),
            "token_logits": self.token_head(node_hidden),
        }


class PretrainedStructuredGraphBaseline(LearnedGraphBaseline):
    """Higher-capacity graph world model pretrained across generated train motifs."""

    name = "pretrained_structured_graph_world_model"
    category = "pretrained_structured_graph"
    implementation_kind = "pretrained_torch_world_model"

    def __init__(self, run_config: Any) -> None:
        super().__init__(run_config)
        self.hidden_size = max(32, int(self.parameters.get("hidden_size", 128)))
        self.message_passing_steps = max(1, int(self.parameters.get("message_passing_steps", 4)))
        self.pair_width = max(16, int(self.parameters.get("pair_width", 128)))
        self.dynamics_width = max(8, int(self.parameters.get("dynamics_width", 64)))
        self.pretrain_train_environments = max(1, int(self.parameters.get("pretrain_train_environments", 1000)))
        self.pretrain_validation_environments = max(1, int(self.parameters.get("pretrain_validation_environments", 200)))
        self.pretrain_batch_size = max(1, int(self.parameters.get("pretrain_batch_size", 32)))
        self.pretrain_seed_offset = int(self.parameters.get("pretrain_seed_offset", 10000))
        self.benchmark_config = str(self.parameters.get("benchmark_config", "configs/benchmark/release_v0_1.json"))
        self._global_cache_key = f"{self.name}|seed{self.seed}"

    @property
    def model_class(self) -> type[nn.Module]:
        return _PretrainedStructuredGraphWorldModel

    def build_model(self, graph_data: GraphTrainingData) -> nn.Module:
        model = _PretrainedStructuredGraphWorldModel(
            input_dim=int(graph_data.node_features.shape[-1]),
            hidden_size=self.hidden_size,
            token_count=max(1, len(graph_data.token_order)),
            message_passing_steps=self.message_passing_steps,
            pair_width=self.pair_width,
            dynamics_width=self.dynamics_width,
        )
        self.parameter_count, self.trainable_parameter_count = count_parameters(model)
        return model

    def forward_outputs(self, model: nn.Module, graph_data: GraphTrainingData) -> dict[str, torch.Tensor]:
        return model(graph_data)

    def _train_or_load_model(
        self,
        environment_id: str,
        graph_data: GraphTrainingData,
        checkpoint: Path,
        seed: int,
    ) -> tuple[nn.Module, dict[str, Any]]:
        if environment_id in self._model_cache:
            return self._model_cache[environment_id], dict(self._training_cache[environment_id])
        if self._global_cache_key in self._model_cache:
            model = self._model_cache[self._global_cache_key]
            summary = dict(self._training_cache[self._global_cache_key])
            self._model_cache[environment_id] = model
            self._training_cache[environment_id] = summary
            return model, summary

        global_checkpoint = checkpoint_path(
            checkpoint_dir=self.checkpoint_dir,
            baseline_name=self.name,
            environment_id="global_pretrain",
            seed=seed,
            parameters=self.parameters,
        )
        model = self.build_model(graph_data).to(self.torch_device)
        total_parameters, trainable_parameters = count_parameters(model)
        self.parameter_count = total_parameters
        self.trainable_parameter_count = trainable_parameters

        if global_checkpoint.is_file():
            LOGGER.info(
                "Loading pretrained graph baseline checkpoint baseline=%s seed=%s device=%s checkpoint=%s",
                self.name,
                seed,
                self.torch_device,
                global_checkpoint,
            )
            payload = load_checkpoint(global_checkpoint, map_location=self.torch_device)
            try:
                model.load_state_dict(payload["state_dict"])
            except RuntimeError:
                LOGGER.warning("Discarding incompatible pretrained checkpoint path=%s", global_checkpoint)
                global_checkpoint.unlink(missing_ok=True)
            else:
                model.eval()
                summary = dict(payload.get("training_summary", {}))
                summary.setdefault("parameter_count", total_parameters)
                summary.setdefault("trainable_parameter_count", trainable_parameters)
                summary["torch_device_resolved"] = str(self.torch_device)
                self._cache_global_and_environment(environment_id, model, summary)
                return model, summary

        train_graphs, validation_graphs = self._build_pretraining_graphs()
        LOGGER.info(
            "Pretraining graph baseline baseline=%s seed=%s train_graphs=%d validation_graphs=%d params=%d device=%s checkpoint=%s",
            self.name,
            seed,
            len(train_graphs),
            len(validation_graphs),
            trainable_parameters,
            self.torch_device,
            global_checkpoint,
        )
        summary = self._pretrain_global_model(
            model=model,
            train_graphs=train_graphs,
            validation_graphs=validation_graphs,
            seed=seed,
            checkpoint=global_checkpoint,
        )
        self._cache_global_and_environment(environment_id, model, summary)
        return model, summary

    def _cache_global_and_environment(self, environment_id: str, model: nn.Module, summary: dict[str, Any]) -> None:
        self._model_cache[self._global_cache_key] = model
        self._training_cache[self._global_cache_key] = dict(summary)
        self._model_cache[environment_id] = model
        self._training_cache[environment_id] = dict(summary)

    def _build_pretraining_graphs(self) -> tuple[list[GraphTrainingData], list[GraphTrainingData]]:
        bundle = load_release_bundle(self.benchmark_config)
        generator = Map2DGenerator(bundle.env2d)
        train_motifs = tuple(bundle.env2d.splits.train_motifs)
        val_motifs = tuple(bundle.env2d.splits.val_motifs)
        return (
            self._generate_graphs(generator, train_motifs, self.pretrain_train_environments, self.pretrain_seed_offset),
            self._generate_graphs(
                generator,
                val_motifs,
                self.pretrain_validation_environments,
                self.pretrain_seed_offset + 1_000_000,
            ),
        )

    def _generate_graphs(
        self,
        generator: Map2DGenerator,
        motifs: tuple[str, ...],
        count: int,
        seed_offset: int,
    ) -> list[GraphTrainingData]:
        graphs: list[GraphTrainingData] = []
        context_seed = self.seed
        for index in range(count):
            motif = motifs[index % len(motifs)]
            environment = generator.generate(seed=seed_offset + index, motif_tag=motif).environment
            if environment is None:
                continue
            _cells, visited_node_ids = deterministic_exploration_trace(
                environment,
                int(self.parameters.get("pretrain_exploration_budget_steps", 800)),
                context_seed + index,
            )
            graphs.append(build_graph_training_data(environment, visited_node_ids).to_device(self.torch_device))
        return graphs

    def _pretrain_global_model(
        self,
        *,
        model: nn.Module,
        train_graphs: list[GraphTrainingData],
        validation_graphs: list[GraphTrainingData],
        seed: int,
        checkpoint: Path,
    ) -> dict[str, Any]:
        set_torch_seed(seed)
        rng = random.Random(seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        best_state = self._state_dict_on_cpu(model)
        best_val_loss = float("inf")
        best_epoch = 0
        patience = 0
        train_curve: list[float] = []
        val_curve: list[float] = []
        last_breakdown: dict[str, float] = {}

        for epoch in range(self.training_epochs):
            model.train()
            epoch_losses: list[float] = []
            train_breakdowns: list[dict[str, float]] = []
            shuffled_graphs = list(train_graphs)
            rng.shuffle(shuffled_graphs)
            for start_index in range(0, len(shuffled_graphs), self.pretrain_batch_size):
                batch_graphs = shuffled_graphs[start_index : start_index + self.pretrain_batch_size]
                optimizer.zero_grad()
                for graph in batch_graphs:
                    outputs = self.forward_outputs(model, graph)
                    train_loss, train_breakdown = self.training_loss(outputs, graph)
                    (train_loss / len(batch_graphs)).backward()
                    epoch_losses.append(float(train_loss.item()))
                    train_breakdowns.append(train_breakdown)
                optimizer.step()

            model.eval()
            val_losses: list[float] = []
            val_breakdowns: list[dict[str, float]] = []
            with torch.no_grad():
                for graph in validation_graphs:
                    outputs = self.forward_outputs(model, graph)
                    validation_loss, validation_breakdown = self.training_loss(outputs, graph)
                    val_losses.append(float(validation_loss.item()))
                    val_breakdowns.append(validation_breakdown)

            train_loss_value = sum(epoch_losses) / max(1, len(epoch_losses))
            validation_loss_value = sum(val_losses) / max(1, len(val_losses))
            train_curve.append(train_loss_value)
            val_curve.append(validation_loss_value)
            if val_breakdowns:
                last_breakdown = {
                    f"val_{key}": round(sum(item[key] for item in val_breakdowns) / len(val_breakdowns), 6)
                    for key in val_breakdowns[0]
                }
                if train_breakdowns:
                    last_breakdown.update(
                        {
                            f"train_{key}": round(sum(item[key] for item in train_breakdowns) / len(train_breakdowns), 6)
                            for key in train_breakdowns[0]
                        }
                    )

            if validation_loss_value + 1e-8 < best_val_loss:
                best_val_loss = validation_loss_value
                best_epoch = epoch + 1
                best_state = self._state_dict_on_cpu(model)
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        summary = {
            "training_scope": "pretrained_across_generated_train_motifs",
            "training_epochs": len(train_curve),
            "configured_epochs": self.training_epochs,
            "learning_rate": self.learning_rate,
            "best_epoch": best_epoch,
            "best_validation_loss": round(best_val_loss if best_val_loss != float("inf") else 0.0, 6),
            "final_train_loss": round(train_curve[-1] if train_curve else 0.0, 6),
            "train_loss_curve": tuple(round(value, 6) for value in train_curve),
            "validation_loss_curve": tuple(round(value, 6) for value in val_curve),
            "loss_breakdown": dict(last_breakdown),
            "pretrain_train_environments": self.pretrain_train_environments,
            "pretrain_validation_environments": self.pretrain_validation_environments,
            "pretrain_batch_size": self.pretrain_batch_size,
            "checkpoint_path": str(checkpoint),
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "torch_device_request": self.device_request,
            "torch_device_resolved": str(self.torch_device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(self.torch_device) if self.torch_device.type == "cuda" else "",
        }
        save_checkpoint(checkpoint, {"state_dict": self._state_dict_on_cpu(model), "training_summary": summary})
        LOGGER.info(
            "Finished pretrained graph baseline baseline=%s seed=%s epochs=%d best_validation_loss=%s",
            self.name,
            seed,
            len(train_curve),
            summary["best_validation_loss"],
        )
        return summary


register_baseline(PretrainedStructuredGraphBaseline.name, PretrainedStructuredGraphBaseline)
