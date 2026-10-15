"""One explicit hardware profile; legacy xArm settings stay in the legacy path."""
from dataclasses import dataclass, asdict, field, fields, replace
import math
from pathlib import Path
from typing import Optional

import yaml

from .interfaces import Pose6D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config/hardware/fr3.yaml"


@dataclass(frozen=True)
class ArmConfig:
    workspace: str = "/home/seur/data/wpy/armcode_agent"
    stack: str = "real"
    speed_scaling: float = 0.10
    position_tolerance_m: float = 0.003
    orientation_tolerance_rad: float = math.pi / 180
    request_timeout_s: float = 180
    singularity_screening: str = 'endpoint'
    home_pose: Optional[list] = None
    # Absolute flange Z in fr3_link0 for releasing an object; None uses vision-relative height.
    place_flange_z_m: Optional[float] = None


@dataclass(frozen=True)
class GripperConfig:
    port: Optional[str] = None
    baudrate: int = 115200
    slave_id: int = 9
    io_timeout_s: float = 0.5
    motion_timeout_s: float = 15
    poll_interval_s: float = 0.05
    keepalive_interval_s: float = 0.1
    startup_timeout_s: float = 1.0
    state_timeout_s: float = 0.5
    speed_raw: int = 64
    force_raw: int = 64
    open_raw: int = 0
    close_raw: int = 255
    # Current-based closing protection, not a calibrated fingertip force sensor.
    close_current_limit_ma: Optional[int] = None
    feedback_log_dir: Optional[str] = None


@dataclass(frozen=True)
class CameraSettings:
    # Confirmed roles: fixed overhead D435i (global), calibrated wrist D435i.
    global_serial: str = "346522072096"
    wrist_serial: str = "349622072103"
    width: int = 640
    height: int = 480
    fps: int = 6
    show_window: bool = False


@dataclass(frozen=True)
class PerceptionSettings:
    grasp_backend: str = "gsnet"
    gsnet_root: str = "/home/seur/data/wpy/conda_envs/env_armcode_agent/src/gsnet"
    sam3_model_dir: str = "/home/seur/data/wpy/models/sam3"
    # Model loading is deferred until the corresponding perception skill runs.
    grounding_dino_model: Optional[str] = None
    anygrasp_dir: Optional[str] = None
    anygrasp_ckpt: Optional[str] = None
    anygrasp_python: Optional[str] = None
    reference_config: Optional[str] = None


@dataclass(frozen=True)
class HardwareConfig:
    arm: ArmConfig = field(default_factory=ArmConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    cameras: CameraSettings = field(default_factory=CameraSettings)
    perception: PerceptionSettings = field(default_factory=PerceptionSettings)
    calibration_file: Optional[str] = None
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _section(cls, data):
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def validate_gripper_feedback(config):
    limit = config.close_current_limit_ma
    if limit is not None and (type(limit) is not int or not 10 <= limit <= 2550 or limit % 10):
        raise ValueError('gripper.close_current_limit_ma must be null or 10..2550 mA in 10 mA steps')
    if config.feedback_log_dir is not None and (not isinstance(config.feedback_log_dir, str)
            or not Path(config.feedback_log_dir).is_absolute()):
        raise ValueError('gripper.feedback_log_dir must be null or an absolute path')


def load_config(path=None) -> HardwareConfig:
    path = Path(path or DEFAULT_CONFIG).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Hardware profile not found: {path}. Copy config/hardware/fr3.example.yaml "
            "to config/hardware/fr3.yaml and configure this machine; see docs/SETUP.md."
        )
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Hardware configuration must be a mapping")
    raw = dict(raw)
    for name, cls in [("arm", ArmConfig), ("gripper", GripperConfig),
                      ("cameras", CameraSettings), ("perception", PerceptionSettings)]:
        raw[name] = _section(cls, raw.get(name, {}))
    cfg = _section(HardwareConfig, raw)
    if cfg.perception.reference_config is not None:
        reference = Path(cfg.perception.reference_config).expanduser()
        if not reference.is_absolute():
            reference = path.parent / reference
        cfg = replace(cfg, perception=replace(cfg.perception, reference_config=str(reference.resolve())))
    validate_gripper_feedback(cfg.gripper)
    if cfg.perception.grasp_backend not in ("gsnet", "anygrasp"):
        raise ValueError("perception.grasp_backend must be gsnet or anygrasp")
    if cfg.schema_version != 1 or cfg.arm.stack not in ("real", "fake"):
        raise ValueError("Expected schema_version: 1 and arm.stack: real or fake")
    if cfg.arm.singularity_screening not in ('endpoint', 'advisory'):
        raise ValueError('arm.singularity_screening must be endpoint or advisory')
    if not Path(cfg.arm.workspace).is_absolute():
        raise ValueError("arm.workspace must be an absolute path")
    for name in ("speed_scaling", "position_tolerance_m", "orientation_tolerance_rad", "request_timeout_s"):
        _positive(f"arm.{name}", getattr(cfg.arm, name))
    if cfg.arm.speed_scaling > 1 or cfg.arm.orientation_tolerance_rad > math.pi:
        raise ValueError("speed_scaling must be <= 1; orientation tolerance must be <= pi")
    if cfg.arm.home_pose is not None:
        Pose6D(*cfg.arm.home_pose)
    if cfg.arm.place_flange_z_m is not None:
        _positive('arm.place_flange_z_m', cfg.arm.place_flange_z_m)
    for name in ("io_timeout_s", "motion_timeout_s", "poll_interval_s", "keepalive_interval_s",
                 "startup_timeout_s", "state_timeout_s"):
        _positive(f"gripper.{name}", getattr(cfg.gripper, name))
    if cfg.gripper.poll_interval_s < 0.005:
        raise ValueError("Robotiq polling interval must be at least 5 ms")
    if not .005 <= cfg.gripper.keepalive_interval_s <= .5:
        raise ValueError("gripper.keepalive_interval_s must be in [0.005, 0.5]")
    if not cfg.gripper.keepalive_interval_s < cfg.gripper.state_timeout_s < 1:
        raise ValueError("gripper.state_timeout_s must exceed keepalive_interval_s and be < 1 second")
    for name in ("speed_raw", "force_raw", "open_raw", "close_raw", "slave_id", "baudrate"):
        val = getattr(cfg.gripper, name)
        low, high = (1, 247) if name == "slave_id" else ((1, 1000000) if name == "baudrate" else (0, 255))
        if type(val) is not int or not low <= val <= high:
            raise ValueError(f"Invalid gripper.{name}")
    if cfg.gripper.port is not None and (not isinstance(cfg.gripper.port, str) or not cfg.gripper.port.startswith("/dev/")):
        raise ValueError("gripper.port must be null or an explicit /dev/... path")
    for name in ("width", "height", "fps"):
        val = getattr(cfg.cameras, name)
        if type(val) is not int or val <= 0:
            raise ValueError(f"Invalid cameras.{name}")
    if type(cfg.cameras.show_window) is not bool:
        raise ValueError("cameras.show_window must be a boolean")
    for name in ("global_serial", "wrist_serial"):
        if not isinstance(getattr(cfg.cameras, name), str) or not getattr(cfg.cameras, name):
            raise ValueError(f"cameras.{name} must be a quoted serial string")
    if cfg.cameras.global_serial == cfg.cameras.wrist_serial:
        raise ValueError("Global and wrist camera serials must differ")
    if cfg.calibration_file is not None:
        if not isinstance(cfg.calibration_file, str) or not cfg.calibration_file.strip():
            raise ValueError("calibration_file must be null or a nonempty path string")
        calibration = Path(cfg.calibration_file).expanduser()
        if not calibration.is_absolute():
            calibration = path.parent / calibration
        cfg = replace(cfg, calibration_file=str(calibration.resolve()))
    return cfg
