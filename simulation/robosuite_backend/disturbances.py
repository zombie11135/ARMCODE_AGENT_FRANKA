"""Extension point only: no perturbations or recovery enabled by default."""
from typing import Protocol, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .environment import PandaTabletop


class DisturbanceHook(Protocol):
    def __call__(self, env: 'PandaTabletop', *, step: int, sim_time_s: float,
                 rng: np.random.Generator) -> dict | None:
        """Called before each control step; return JSON event details if applied.

        A future implementation may apply body forces through env.sim. Use rng
        for repeatability. Clear transient forces in the hook when they expire.
        Called on the simulation owner thread, never exposed to policy code.
        """
