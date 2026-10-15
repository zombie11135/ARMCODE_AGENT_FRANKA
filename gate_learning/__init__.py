"""Multimodal Gate learning components.

This package is deliberately separate from :mod:`nesyro_gate`: importing the
legacy runtime must not pull in torch or alter the existing gate defaults.
"""

from gate_learning.input_adapter import (
    MULTIMODAL_STATE_FEATURE_NAMES,
    MultimodalGateInput,
    from_sim_policy_observation,
)

__all__ = [
    "MULTIMODAL_STATE_FEATURE_NAMES",
    "MultimodalGateInput",
    "from_sim_policy_observation",
]

__version__ = "0.1.0"
