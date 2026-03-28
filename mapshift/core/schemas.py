"""Zero-dependency config schemas and release bundle validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal

FamilyName = Literal["metric", "topology", "dynamics", "semantic"]
TaskClassName = Literal["planning", "inference", "adaptation"]
TierName = Literal["mapshift_2d", "mapshift_3d"]

FAMILY_NAMES: tuple[FamilyName, ...] = ("metric", "topology", "dynamics", "semantic")
TASK_CLASS_NAMES: tuple[TaskClassName, ...] = ("planning", "inference", "adaptation")
TIER_NAMES: tuple[TierName, ...] = ("mapshift_2d", "mapshift_3d")


class ConfigValidationError(ValueError):
    """Raised when a config bundle is structurally invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigValidationError(message)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigValidationError(f"Missing config file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"Invalid JSON in config file {path}: {exc}") from exc

    _require(isinstance(data, dict), f"Top-level config must be an object: {path}")
    return data


def _as_non_empty_str(value: Any, field_name: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field_name} must be a non-empty string")
    return value


def _as_bool(value: Any, field_name: str) -> bool:
    _require(isinstance(value, bool), f"{field_name} must be a boolean")
    return value


def _as_int(value: Any, field_name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field_name} must be an integer")
    return value


def _as_positive_int(value: Any, field_name: str) -> int:
    parsed = _as_int(value, field_name)
    _require(parsed > 0, f"{field_name} must be > 0")
    return parsed


def _as_float(value: Any, field_name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field_name} must be numeric")
    return float(value)


def _as_non_empty_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    _require(isinstance(value, list) and value, f"{field_name} must be a non-empty list")
    items = tuple(_as_non_empty_str(item, f"{field_name}[]") for item in value)
    return items


def _as_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{field_name} must be a list")
    return tuple(_as_non_empty_str(item, f"{field_name}[]") for item in value)


def _as_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    _require(isinstance(value, dict), f"{field_name} must be an object")
    return value


def _choice(value: str, field_name: str, allowed: Iterable[str]) -> str:
    allowed_values = tuple(allowed)
    _require(value in allowed_values, f"{field_name} must be one of {allowed_values}, got {value!r}")
    return value


@dataclass(frozen=True)
class ConfigRefs:
    env2d: str
    env3d: str
    interventions: str
    tasks: str
    baselines: str
    analysis: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigRefs":
        return cls(
            env2d=_as_non_empty_str(data.get("env2d"), "config_refs.env2d"),
            env3d=_as_non_empty_str(data.get("env3d"), "config_refs.env3d"),
            interventions=_as_non_empty_str(data.get("interventions"), "config_refs.interventions"),
            tasks=_as_non_empty_str(data.get("tasks"), "config_refs.tasks"),
            baselines=_as_non_empty_str(data.get("baselines"), "config_refs.baselines"),
            analysis=_as_non_empty_str(data.get("analysis"), "config_refs.analysis"),
        )

    def resolve(self, base_dir: Path) -> Dict[str, Path]:
        resolved = {
            "env2d": (base_dir / self.env2d).resolve(),
            "env3d": (base_dir / self.env3d).resolve(),
            "interventions": (base_dir / self.interventions).resolve(),
            "tasks": (base_dir / self.tasks).resolve(),
            "baselines": (base_dir / self.baselines).resolve(),
            "analysis": (base_dir / self.analysis).resolve(),
        }
        for name, path in resolved.items():
            _require(path.exists(), f"Config reference {name} does not exist: {path}")
        return resolved


@dataclass(frozen=True)
class FreezeConfig:
    scientific_spec: tuple[str, ...]
    protocol: tuple[str, ...]
    final_evaluation: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FreezeConfig":
        return cls(
            scientific_spec=_as_non_empty_string_list(data.get("scientific_spec"), "freeze.scientific_spec"),
            protocol=_as_non_empty_string_list(data.get("protocol"), "freeze.protocol"),
            final_evaluation=_as_non_empty_string_list(data.get("final_evaluation"), "freeze.final_evaluation"),
        )


@dataclass(frozen=True)
class BenchmarkReleaseConfig:
    schema_version: str
    benchmark_name: str
    benchmark_version: str
    release_name: str
    status: str
    primary_tier: TierName
    supported_tiers: tuple[TierName, ...]
    config_refs: ConfigRefs
    freeze: FreezeConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkReleaseConfig":
        supported_tiers = tuple(
            _choice(_as_non_empty_str(item, "supported_tiers[]"), "supported_tiers[]", TIER_NAMES)
            for item in data.get("supported_tiers", [])
        )
        _require(supported_tiers, "supported_tiers must be a non-empty list")
        primary_tier = _choice(_as_non_empty_str(data.get("primary_tier"), "primary_tier"), "primary_tier", TIER_NAMES)

        config = cls(
            schema_version=_as_non_empty_str(data.get("schema_version"), "schema_version"),
            benchmark_name=_as_non_empty_str(data.get("benchmark_name"), "benchmark_name"),
            benchmark_version=_as_non_empty_str(data.get("benchmark_version"), "benchmark_version"),
            release_name=_as_non_empty_str(data.get("release_name"), "release_name"),
            status=_as_non_empty_str(data.get("status"), "status"),
            primary_tier=primary_tier,  # type: ignore[arg-type]
            supported_tiers=supported_tiers,  # type: ignore[arg-type]
            config_refs=ConfigRefs.from_dict(_as_mapping(data.get("config_refs"), "config_refs")),
            freeze=FreezeConfig.from_dict(_as_mapping(data.get("freeze"), "freeze")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        _require(self.benchmark_name == "MapShift", "benchmark_name must be 'MapShift'")
        _require(self.primary_tier in self.supported_tiers, "primary_tier must be included in supported_tiers")


@dataclass(frozen=True)
class ActionSpaceConfig:
    name: str
    forward_step_m: float
    turn_angle_deg: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionSpaceConfig":
        config = cls(
            name=_as_non_empty_str(data.get("name"), "action_space.name"),
            forward_step_m=_as_float(data.get("forward_step_m"), "action_space.forward_step_m"),
            turn_angle_deg=_as_float(data.get("turn_angle_deg"), "action_space.turn_angle_deg"),
        )
        _require(config.forward_step_m > 0.0, "action_space.forward_step_m must be > 0")
        _require(config.turn_angle_deg > 0.0, "action_space.turn_angle_deg must be > 0")
        return config


@dataclass(frozen=True)
class Observation2DConfig:
    mode: str
    radius_m: float
    field_of_view_deg: float
    semantic_channels: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation2DConfig":
        config = cls(
            mode=_as_non_empty_str(data.get("mode"), "observation.mode"),
            radius_m=_as_float(data.get("radius_m"), "observation.radius_m"),
            field_of_view_deg=_as_float(data.get("field_of_view_deg"), "observation.field_of_view_deg"),
            semantic_channels=_as_bool(data.get("semantic_channels"), "observation.semantic_channels"),
        )
        _require(config.radius_m > 0.0, "observation.radius_m must be > 0")
        _require(0.0 < config.field_of_view_deg <= 360.0, "observation.field_of_view_deg must be in (0, 360]")
        return config


@dataclass(frozen=True)
class ExplorationConfig:
    canonical_budget_steps: int
    reset_policy: str
    start_state_policy: str
    privileged_state: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any], prefix: str) -> "ExplorationConfig":
        return cls(
            canonical_budget_steps=_as_positive_int(data.get("canonical_budget_steps"), f"{prefix}.canonical_budget_steps"),
            reset_policy=_as_non_empty_str(data.get("reset_policy"), f"{prefix}.reset_policy"),
            start_state_policy=_as_non_empty_str(data.get("start_state_policy"), f"{prefix}.start_state_policy"),
            privileged_state=_as_bool(data.get("privileged_state"), f"{prefix}.privileged_state"),
        )


@dataclass(frozen=True)
class HorizonConfig:
    planning_steps: tuple[int, ...]
    inference_steps: tuple[int, ...]
    adaptation_steps: tuple[int, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HorizonConfig":
        return cls(
            planning_steps=tuple(_as_positive_int(value, "horizons.planning_steps[]") for value in data.get("planning_steps", [])),
            inference_steps=tuple(_as_positive_int(value, "horizons.inference_steps[]") for value in data.get("inference_steps", [])),
            adaptation_steps=tuple(_as_positive_int(value, "horizons.adaptation_steps[]") for value in data.get("adaptation_steps", [])),
        )


@dataclass(frozen=True)
class SplitConfig:
    train_motifs: tuple[str, ...]
    val_motifs: tuple[str, ...]
    test_motifs: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitConfig":
        config = cls(
            train_motifs=_as_non_empty_string_list(data.get("train_motifs"), "splits.train_motifs"),
            val_motifs=_as_non_empty_string_list(data.get("val_motifs"), "splits.val_motifs"),
            test_motifs=_as_non_empty_string_list(data.get("test_motifs"), "splits.test_motifs"),
        )
        overlap = (set(config.train_motifs) & set(config.val_motifs)) | (set(config.train_motifs) & set(config.test_motifs)) | (
            set(config.val_motifs) & set(config.test_motifs)
        )
        _require(not overlap, f"split motif families must be disjoint, got overlap: {sorted(overlap)}")
        return config


@dataclass(frozen=True)
class Env2DConfig:
    schema_version: str
    tier: TierName
    generator_name: str
    map_size_cells: tuple[int, int]
    occupancy_resolution_m: float
    pose_mode: str
    action_space: ActionSpaceConfig
    observation: Observation2DConfig
    motif_families: tuple[str, ...]
    splits: SplitConfig
    exploration: ExplorationConfig
    horizons: HorizonConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Env2DConfig":
        map_size = data.get("map_size_cells")
        _require(
            isinstance(map_size, list) and len(map_size) == 2,
            "map_size_cells must be a two-element list",
        )
        config = cls(
            schema_version=_as_non_empty_str(data.get("schema_version"), "schema_version"),
            tier=_choice(_as_non_empty_str(data.get("tier"), "tier"), "tier", TIER_NAMES),  # type: ignore[arg-type]
            generator_name=_as_non_empty_str(data.get("generator_name"), "generator_name"),
            map_size_cells=(
                _as_positive_int(map_size[0], "map_size_cells[0]"),
                _as_positive_int(map_size[1], "map_size_cells[1]"),
            ),
            occupancy_resolution_m=_as_float(data.get("occupancy_resolution_m"), "occupancy_resolution_m"),
            pose_mode=_as_non_empty_str(data.get("pose_mode"), "pose_mode"),
            action_space=ActionSpaceConfig.from_dict(_as_mapping(data.get("action_space"), "action_space")),
            observation=Observation2DConfig.from_dict(_as_mapping(data.get("observation"), "observation")),
            motif_families=_as_non_empty_string_list(data.get("motif_families"), "motif_families"),
            splits=SplitConfig.from_dict(_as_mapping(data.get("splits"), "splits")),
            exploration=ExplorationConfig.from_dict(_as_mapping(data.get("exploration"), "exploration"), "exploration"),
            horizons=HorizonConfig.from_dict(_as_mapping(data.get("horizons"), "horizons")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        _require(self.tier == "mapshift_2d", "Env2DConfig tier must be 'mapshift_2d'")
        _require(self.occupancy_resolution_m > 0.0, "occupancy_resolution_m must be > 0")
        _require(set(self.splits.train_motifs).issubset(set(self.motif_families)), "train split motifs must exist in motif_families")
        _require(set(self.splits.val_motifs).issubset(set(self.motif_families)), "val split motifs must exist in motif_families")
        _require(set(self.splits.test_motifs).issubset(set(self.motif_families)), "test split motifs must exist in motif_families")


@dataclass(frozen=True)
class Observation3DConfig:
    mode: str
    field_of_view_deg: float
    frame_size: tuple[int, int]
    semantic_channels: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation3DConfig":
        frame_size = data.get("frame_size")
        _require(isinstance(frame_size, list) and len(frame_size) == 2, "observation.frame_size must be a two-element list")
        return cls(
            mode=_as_non_empty_str(data.get("mode"), "observation.mode"),
            field_of_view_deg=_as_float(data.get("field_of_view_deg"), "observation.field_of_view_deg"),
            frame_size=(
                _as_positive_int(frame_size[0], "observation.frame_size[0]"),
                _as_positive_int(frame_size[1], "observation.frame_size[1]"),
            ),
            semantic_channels=_as_bool(data.get("semantic_channels"), "observation.semantic_channels"),
        )


@dataclass(frozen=True)
class Control3DConfig:
    move_step_m: float
    rotate_step_deg: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Control3DConfig":
        return cls(
            move_step_m=_as_float(data.get("move_step_m"), "control.move_step_m"),
            rotate_step_deg=_as_float(data.get("rotate_step_deg"), "control.rotate_step_deg"),
        )


@dataclass(frozen=True)
class Env3DConfig:
    schema_version: str
    tier: TierName
    platform: str
    scene_sampler: str
    observation: Observation3DConfig
    control: Control3DConfig
    exploration: ExplorationConfig
    supported_intervention_families: tuple[FamilyName, ...]
    documented_deviations: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Env3DConfig":
        families = tuple(
            _choice(_as_non_empty_str(item, "supported_intervention_families[]"), "supported_intervention_families[]", FAMILY_NAMES)
            for item in data.get("supported_intervention_families", [])
        )
        _require(families, "supported_intervention_families must be non-empty")
        config = cls(
            schema_version=_as_non_empty_str(data.get("schema_version"), "schema_version"),
            tier=_choice(_as_non_empty_str(data.get("tier"), "tier"), "tier", TIER_NAMES),  # type: ignore[arg-type]
            platform=_as_non_empty_str(data.get("platform"), "platform"),
            scene_sampler=_as_non_empty_str(data.get("scene_sampler"), "scene_sampler"),
            observation=Observation3DConfig.from_dict(_as_mapping(data.get("observation"), "observation")),
            control=Control3DConfig.from_dict(_as_mapping(data.get("control"), "control")),
            exploration=ExplorationConfig.from_dict(_as_mapping(data.get("exploration"), "exploration"), "exploration"),
            supported_intervention_families=families,  # type: ignore[arg-type]
            documented_deviations=_as_string_list(data.get("documented_deviations", []), "documented_deviations"),
        )
        _require(config.tier == "mapshift_3d", "Env3DConfig tier must be 'mapshift_3d'")
        _require(config.platform == "ProcTHOR", "Env3DConfig platform must be 'ProcTHOR'")
        return config


@dataclass(frozen=True)
class SeverityLevelConfig:
    value: float
    interventions: tuple[str, ...]

    @classmethod
    def from_dict(cls, key: str, data: Dict[str, Any]) -> "SeverityLevelConfig":
        return cls(
            value=_as_float(data.get("value"), f"severity_levels.{key}.value"),
            interventions=_as_non_empty_string_list(data.get("interventions"), f"severity_levels.{key}.interventions"),
        )


@dataclass(frozen=True)
class FamilyInterventionConfig:
    family: FamilyName
    severity_parameter: str
    severity_levels: Dict[str, SeverityLevelConfig]
    preserve: tuple[str, ...]
    expected_failures: tuple[str, ...]
    validator_invariants: tuple[str, ...]

    @classmethod
    def from_dict(cls, family_name: str, data: Dict[str, Any]) -> "FamilyInterventionConfig":
        family = _choice(family_name, "family", FAMILY_NAMES)
        raw_levels = _as_mapping(data.get("severity_levels"), f"families.{family}.severity_levels")
        severity_levels = {key: SeverityLevelConfig.from_dict(key, _as_mapping(value, f"families.{family}.severity_levels.{key}")) for key, value in raw_levels.items()}
        _require(set(severity_levels) == {"0", "1", "2", "3"}, f"{family} severity levels must be exactly 0, 1, 2, 3")
        ordered_values = [severity_levels[key].value for key in ("0", "1", "2", "3")]
        nondecreasing = ordered_values == sorted(ordered_values)
        nonincreasing = ordered_values == sorted(ordered_values, reverse=True)
        _require(nondecreasing or nonincreasing, f"{family} severity values must be monotone")
        return cls(
            family=family,  # type: ignore[arg-type]
            severity_parameter=_as_non_empty_str(data.get("severity_parameter"), f"families.{family}.severity_parameter"),
            severity_levels=severity_levels,
            preserve=_as_non_empty_string_list(data.get("preserve"), f"families.{family}.preserve"),
            expected_failures=_as_non_empty_string_list(data.get("expected_failures"), f"families.{family}.expected_failures"),
            validator_invariants=_as_non_empty_string_list(data.get("validator_invariants"), f"families.{family}.validator_invariants"),
        )


@dataclass(frozen=True)
class InterventionConfig:
    schema_version: str
    canonical_family_order: tuple[FamilyName, ...]
    families: Dict[FamilyName, FamilyInterventionConfig]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterventionConfig":
        family_order = tuple(
            _choice(_as_non_empty_str(item, "canonical_family_order[]"), "canonical_family_order[]", FAMILY_NAMES)
            for item in data.get("canonical_family_order", [])
        )
        _require(family_order == FAMILY_NAMES, f"canonical_family_order must be {FAMILY_NAMES}")
        raw_families = _as_mapping(data.get("families"), "families")
        families: Dict[FamilyName, FamilyInterventionConfig] = {}
        for family_name in FAMILY_NAMES:
            _require(family_name in raw_families, f"Missing intervention family config: {family_name}")
            families[family_name] = FamilyInterventionConfig.from_dict(family_name, _as_mapping(raw_families[family_name], f"families.{family_name}"))
        return cls(
            schema_version=_as_non_empty_str(data.get("schema_version"), "schema_version"),
            canonical_family_order=family_order,  # type: ignore[arg-type]
            families=families,
        )


@dataclass(frozen=True)
class TaskClassConfig:
    enabled: bool
    task_types: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    canonical_horizon_steps: tuple[int, ...]

    @classmethod
    def from_dict(cls, class_name: str, data: Dict[str, Any]) -> "TaskClassConfig":
        return cls(
            enabled=_as_bool(data.get("enabled"), f"classes.{class_name}.enabled"),
            task_types=_as_non_empty_string_list(data.get("task_types"), f"classes.{class_name}.task_types"),
            primary_metrics=_as_non_empty_string_list(data.get("primary_metrics"), f"classes.{class_name}.primary_metrics"),
            canonical_horizon_steps=tuple(
                _as_positive_int(value, f"classes.{class_name}.canonical_horizon_steps[]")
                for value in data.get("canonical_horizon_steps", [])
            ),
        )


@dataclass(frozen=True)
class TaskConfig:
    schema_version: str
    classes: Dict[TaskClassName, TaskClassConfig]
    hold_fixed: tuple[str, ...]
    reject_impossible_tasks: bool
    reject_trivial_tasks: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        raw_classes = _as_mapping(data.get("classes"), "classes")
        classes: Dict[TaskClassName, TaskClassConfig] = {}
        for class_name in TASK_CLASS_NAMES:
            _require(class_name in raw_classes, f"Missing task class config: {class_name}")
            classes[class_name] = TaskClassConfig.from_dict(class_name, _as_mapping(raw_classes[class_name], f"classes.{class_name}"))

        shared_sampling = _as_mapping(data.get("shared_sampling"), "shared_sampling")
        return cls(
            schema_version=_as_non_empty_str(data.get("schema_version"), "schema_version"),
            classes=classes,
            hold_fixed=_as_non_empty_string_list(shared_sampling.get("hold_fixed"), "shared_sampling.hold_fixed"),
            reject_impossible_tasks=_as_bool(shared_sampling.get("reject_impossible_tasks"), "shared_sampling.reject_impossible_tasks"),
            reject_trivial_tasks=_as_bool(shared_sampling.get("reject_trivial_tasks"), "shared_sampling.reject_trivial_tasks"),
        )


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    enabled: bool
    category: str
    notes: str

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any]) -> "BaselineSpec":
        return cls(
            name=_as_non_empty_str(data.get("name"), f"system_families[{index}].name"),
            enabled=_as_bool(data.get("enabled"), f"system_families[{index}].enabled"),
            category=_as_non_empty_str(data.get("category"), f"system_families[{index}].category"),
            notes=_as_non_empty_str(data.get("notes"), f"system_families[{index}].notes"),
        )


@dataclass(frozen=True)
class FairnessConfig:
    match_exploration_budget: bool
    shared_validation_protocol: bool
    freeze_hparams_before_test: bool
    report_compute_and_params: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FairnessConfig":
        return cls(
            match_exploration_budget=_as_bool(data.get("match_exploration_budget"), "fairness.match_exploration_budget"),
            shared_validation_protocol=_as_bool(data.get("shared_validation_protocol"), "fairness.shared_validation_protocol"),
            freeze_hparams_before_test=_as_bool(data.get("freeze_hparams_before_test"), "fairness.freeze_hparams_before_test"),
            report_compute_and_params=_as_bool(data.get("report_compute_and_params"), "fairness.report_compute_and_params"),
        )


@dataclass(frozen=True)
class BaselineConfig:
    schema_version: str
    fairness: FairnessConfig
    system_families: tuple[BaselineSpec, ...]
    sanity_baselines: tuple[str, ...]
    seed_counts: Dict[str, int]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineConfig":
        seed_counts = _as_mapping(data.get("seed_counts"), "seed_counts")
        config = cls(
            schema_version=_as_non_empty_str(data.get("schema_version"), "schema_version"),
            fairness=FairnessConfig.from_dict(_as_mapping(data.get("fairness"), "fairness")),
            system_families=tuple(
                BaselineSpec.from_dict(index, _as_mapping(item, f"system_families[{index}]"))
                for index, item in enumerate(data.get("system_families", []))
            ),
            sanity_baselines=_as_non_empty_string_list(data.get("sanity_baselines"), "sanity_baselines"),
            seed_counts={key: _as_positive_int(value, f"seed_counts.{key}") for key, value in seed_counts.items()},
        )
        _require(config.system_families, "system_families must be non-empty")
        return config


@dataclass(frozen=True)
class BootstrapConfig:
    resamples: int
    confidence_level: float
    paired_by: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BootstrapConfig":
        config = cls(
            resamples=_as_positive_int(data.get("resamples"), "bootstrap.resamples"),
            confidence_level=_as_float(data.get("confidence_level"), "bootstrap.confidence_level"),
            paired_by=_as_non_empty_str(data.get("paired_by"), "bootstrap.paired_by"),
        )
        _require(0.0 < config.confidence_level < 1.0, "bootstrap.confidence_level must be in (0, 1)")
        return config


@dataclass(frozen=True)
class AnalysisConfig:
    schema_version: str
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    health_metrics: tuple[str, ...]
    bootstrap: BootstrapConfig
    protocol_comparisons: tuple[str, ...]
    core_figures: tuple[str, ...]
    core_tables: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisConfig":
        return cls(
            schema_version=_as_non_empty_str(data.get("schema_version"), "schema_version"),
            primary_metrics=_as_non_empty_string_list(data.get("primary_metrics"), "primary_metrics"),
            secondary_metrics=_as_non_empty_string_list(data.get("secondary_metrics"), "secondary_metrics"),
            health_metrics=_as_non_empty_string_list(data.get("health_metrics"), "health_metrics"),
            bootstrap=BootstrapConfig.from_dict(_as_mapping(data.get("bootstrap"), "bootstrap")),
            protocol_comparisons=_as_non_empty_string_list(data.get("protocol_comparisons"), "protocol_comparisons"),
            core_figures=_as_non_empty_string_list(data.get("core_figures"), "core_figures"),
            core_tables=_as_non_empty_string_list(data.get("core_tables"), "core_tables"),
        )


@dataclass(frozen=True)
class ReleaseBundle:
    root_path: Path
    root: BenchmarkReleaseConfig
    env2d: Env2DConfig
    env3d: Env3DConfig
    interventions: InterventionConfig
    tasks: TaskConfig
    baselines: BaselineConfig
    analysis: AnalysisConfig

    def summary(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.root.benchmark_name,
            "benchmark_version": self.root.benchmark_version,
            "release_name": self.root.release_name,
            "primary_tier": self.root.primary_tier,
            "supported_tiers": list(self.root.supported_tiers),
            "intervention_families": list(self.interventions.canonical_family_order),
            "task_classes": list(self.tasks.classes.keys()),
            "baseline_count": len(self.baselines.system_families),
        }


def load_release_bundle(path: str | Path) -> ReleaseBundle:
    """Load and validate the canonical MapShift release bundle."""

    root_path = Path(path).resolve()
    root = BenchmarkReleaseConfig.from_dict(_read_json(root_path))
    refs = root.config_refs.resolve(root_path.parent)

    env2d = Env2DConfig.from_dict(_read_json(refs["env2d"]))
    env3d = Env3DConfig.from_dict(_read_json(refs["env3d"]))
    interventions = InterventionConfig.from_dict(_read_json(refs["interventions"]))
    tasks = TaskConfig.from_dict(_read_json(refs["tasks"]))
    baselines = BaselineConfig.from_dict(_read_json(refs["baselines"]))
    analysis = AnalysisConfig.from_dict(_read_json(refs["analysis"]))

    _require(env2d.tier in root.supported_tiers, "env2d tier is not included in root.supported_tiers")
    _require(env3d.tier in root.supported_tiers, "env3d tier is not included in root.supported_tiers")
    _require(tuple(interventions.families.keys()) == FAMILY_NAMES, "intervention families must match canonical family order")
    _require(tuple(tasks.classes.keys()) == TASK_CLASS_NAMES, "task classes must match canonical task order")

    return ReleaseBundle(
        root_path=root_path,
        root=root,
        env2d=env2d,
        env3d=env3d,
        interventions=interventions,
        tasks=tasks,
        baselines=baselines,
        analysis=analysis,
    )
