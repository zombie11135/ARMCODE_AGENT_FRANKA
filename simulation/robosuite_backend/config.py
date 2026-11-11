"""Simulation-only settings. Positions are metres in the MuJoCo world frame."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class SimConfig:
    seed: int = 7
    control_freq: int = 20
    camera: str = 'agentview'
    image_width: int = 320
    image_height: int = 240
    table_height: float = 0.8
    cube_half_size: float = 0.02
    spawn_jitter: float = 0.015
    transport_height: float = 1.04
    waypoint_step: float = 0.005
    position_tolerance: float = 0.008
    goal_xy_tolerance: float = 0.035
    goal_z_tolerance: float = 0.008
    settle_speed: float = 0.015
    settle_angular_speed: float = 0.15
    stable_steps: int = 12
    phase_timeout_steps: int = 180

    def __post_init__(self):
        if not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an unsigned 32-bit integer')
        for name in ('control_freq', 'image_width', 'image_height', 'stable_steps', 'phase_timeout_steps'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not 0 <= self.spawn_jitter <= 0.025:
            raise ValueError('spawn_jitter must be between 0 and 0.025 m')
        if self.table_height != 0.8 or self.cube_half_size != 0.02:
            raise ValueError('Phase 1 supports the 0.8 m table and 0.04 m cubes')
        if not 0.001 <= self.waypoint_step <= 0.01:
            raise ValueError('waypoint_step must be between 0.001 and 0.01 m')
        if not 0.95 <= self.transport_height <= 1.1:
            raise ValueError('transport_height must be between 0.95 and 1.1 m')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        unknown = data.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown simulation settings: {sorted(unknown)}')
        return cls(**data)

    def to_dict(self):
        return asdict(self)
