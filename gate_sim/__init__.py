"""Recoverable, partially observable Gate pick/place simulation.

This package is simulation-only.  It deliberately has no imports from the
real-robot services in ``services/`` or ``skill_lib/``.
"""

from gate_sim.config import GateSimConfig
from gate_sim.environment import GateAction, GateOccludedPickPlaceEnv

__all__ = ["GateAction", "GateOccludedPickPlaceEnv", "GateSimConfig"]
__version__ = "0.1.0"
