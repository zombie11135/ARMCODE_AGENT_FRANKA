"""FR3 / Robotiq adapters. Importing this package never connects to hardware."""

from .config import HardwareConfig, load_config
from .interfaces import Pose6D, PendingTransforms

__all__ = ["HardwareConfig", "load_config", "Pose6D", "PendingTransforms"]
