"""Intervention family implementations for MapShift."""

from mapshift.core.schemas import FamilyInterventionConfig

from .base import BaseIntervention, InterventionResult
from .dynamics import DynamicsIntervention
from .metric import MetricIntervention
from .semantic import SemanticIntervention
from .topology import TopologyIntervention

INTERVENTION_TYPES = {
    "metric": MetricIntervention,
    "topology": TopologyIntervention,
    "dynamics": DynamicsIntervention,
    "semantic": SemanticIntervention,
}


def build_intervention(family: str, family_config: FamilyInterventionConfig) -> BaseIntervention:
    """Create an intervention instance for a configured family."""

    if family not in INTERVENTION_TYPES:
        raise KeyError(f"Unknown intervention family: {family}")
    return INTERVENTION_TYPES[family](family_config)


__all__ = [
    "BaseIntervention",
    "DynamicsIntervention",
    "INTERVENTION_TYPES",
    "InterventionResult",
    "MetricIntervention",
    "SemanticIntervention",
    "TopologyIntervention",
    "build_intervention",
]
