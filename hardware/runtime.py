"""FR3/Robotiq wiring for the original agent and explicit legacy pose adapter."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import warnings
from contextlib import nullcontext
from runtime.observation_memory import memory_for

from .config import HardwareConfig, load_config
from .fr3 import FrankaMoveItArm
from .interfaces import Pose6D, Transforms
from .robotiq import Robotiq2F85
from .transforms import load_transforms


class LegacyPoseAdapter:
    """Only the two pose methods used by skills; not an xArm SDK replacement.

    Converts mm to m. Frames and tool transforms are the caller's responsibility;
    it must never be fed the unconverted camera/old xArm plans.
    """
    def __init__(self, arm, healthcheck=None, observation_memory=None):
        self.arm = arm
        self.healthcheck = healthcheck
        self.observation_memory = observation_memory
        self._warned_speed = False

    def get_position(self, is_radian=True):
        if not is_radian:
            raise ValueError("Only radians are supported")
        pose = self.arm.get_pose().values
        return 0, [v * 1000 for v in pose[:3]] + pose[3:]

    def set_position(self, *, x, y, z, roll, pitch, yaw, speed=None, mvacc=None, wait=True):
        if not wait:
            raise ValueError("FR3 adapter supports blocking PTP only (wait=True)")
        if (speed is not None or mvacc is not None) and not self._warned_speed:
            warnings.warn("Legacy speed/mvacc are not FR3 units; FR3 uses --arm-speed / arm.speed_scaling", stacklevel=2)
            self._warned_speed = True
        if self.healthcheck:
            self.healthcheck()
        guard = self.observation_memory.device_lock if self.observation_memory is not None else nullcontext()
        with guard:
            self.arm.move_pose(Pose6D(x / 1000, y / 1000, z / 1000, roll, pitch, yaw))
        if self.healthcheck:
            self.healthcheck()
        return 0  # Returned only after controller + endpoint + cleanup success.

    def disconnect(self):
        self.arm.close()


@dataclass
class Hardware:
    config: HardwareConfig
    arm: FrankaMoveItArm
    gripper: Robotiq2F85
    transforms: Transforms

    def attach(self, context, *, monitor_gripper=False):
        if monitor_gripper:
            self.gripper.start_session()
            state = self.gripper.assert_healthy(require_ready=False)
            if not state.activated:
                self.gripper.activate()
            self.gripper.assert_healthy(require_ready=True)
        context.arm = self.arm
        context.gripper = self.gripper
        context.transforms = self.transforms
        if monitor_gripper:
            self.arm.motion_guard = lambda: self.gripper.assert_healthy(require_ready=True)
        context.xarm = LegacyPoseAdapter(
            self.arm, healthcheck=(lambda: self.gripper.assert_healthy(require_ready=True)) if monitor_gripper else None,
            observation_memory=memory_for(context.camera))

    def open_gripper(self):
        result = self.gripper.open()
        return result.ready and result.object_state == 3

    def close_gripper(self):
        # Completion permits the Agent to continue; this is not proof of object contact.
        result = self.gripper.close()
        return self.gripper.close_completed(result)

    def disconnect(self):
        self.gripper.disconnect()
        self.arm.close()


def build_hardware(path=None, *, execute=False, stack=None, speed_scaling=None, agent=False):
    cfg = load_config(path)
    if agent:
        cfg = replace(cfg, arm=replace(cfg.arm, singularity_screening="advisory"))
    if speed_scaling is not None:
        cfg = replace(cfg, arm=replace(cfg.arm, speed_scaling=speed_scaling))
    transforms = load_transforms(cfg)
    gripper_type = Robotiq2F85
    if cfg.gripper.contact_close_file:
        from .robotiq_contact import AgentContactGripper
        gripper_type = AgentContactGripper
    return Hardware(cfg, FrankaMoveItArm(cfg.arm, stack=stack, execution_enabled=execute),
                    gripper_type(cfg.gripper, execution_enabled=execute), transforms)


def require_task_geometry(config):
    """Guard before camera/model/hardware initialization, including dry runs."""
    if getattr(config, "hardware_config_path", None):
        cfg = load_config(config.hardware_config_path)
        load_transforms(cfg).require_ready()
        if cfg.gripper.contact_close_file:
            from .robotiq_contact import load_contact_close_setup
            load_contact_close_setup(cfg.gripper)  # Files only; fail before camera/model/device startup.
        if cfg.perception.grasp_backend == "anygrasp":
            from services.anygrasp_fr3_service import validate_anygrasp_paths
            validate_anygrasp_paths(cfg.perception)
        speed = getattr(config, "arm_speed_scaling", None)
        if speed is not None:
            FrankaMoveItArm(cfg.arm).set_speed_scaling(speed)


def camera_service(path, capture_dir):
    from services.realsense_dual_camera_service import CameraConfig, RealSenseDualCameraService
    from services.runtime_service_cache import get_or_create_service

    profile = load_config(path)
    cfg = profile.cameras
    params = {**asdict(cfg), "capture_dir": str(capture_dir), "profile": str(path),
              "observation_cache": asdict(profile.observation_cache),
              "arm_profile": asdict(profile.arm), "perception": asdict(profile.perception),
              "calibration": Path(profile.calibration_file).read_text() if profile.calibration_file else None}

    def create():
        def camera(name, serial, depth):
            return CameraConfig(name=name, serial=serial, color_width=cfg.width,
                                color_height=cfg.height, color_fps=cfg.fps,
                                enable_depth=depth, depth_width=cfg.width,
                                depth_height=cfg.height, depth_fps=cfg.fps)
        from .agent_camera import CalibratedCamera
        camera_device = RealSenseDualCameraService(
            global_config=camera("global", cfg.global_serial, False),
            wrist_config=camera("wrist", cfg.wrist_serial, True),
            save_dir=capture_dir, show_window=cfg.show_window,
            active_cameras=('wrist',) if cfg.mode == 'wrist' else ('global', 'wrist'))
        return CalibratedCamera(camera_device, profile)

    return get_or_create_service("fr3_dual_realsense", params, create), params


def apply_cli_profile(args):
    """Hardware YAML is authoritative for the FR3 entry; legacy is explicit."""
    if args.legacy_xarm:
        args.hardware_config = None
        return
    cfg = load_config(args.hardware_config)
    args.simple_actions = True
    args.global_serial = cfg.cameras.global_serial
    args.wrist_serial = cfg.cameras.wrist_serial
    args.handeye_config = cfg.calibration_file or ""
    args.xarm_ip = ""  # Never fall back to the other robot's IP.
    args.ros_ws = ""
    for name in ("grounding_dino_model", "anygrasp_dir", "anygrasp_ckpt"):
        value = getattr(cfg.perception, name)
        if value:
            setattr(args, name, value)
    if getattr(args, "arm_speed_scaling", None) is None:
        args.arm_speed_scaling = cfg.arm.speed_scaling
    FrankaMoveItArm(cfg.arm).set_speed_scaling(args.arm_speed_scaling)
    args.gripper_open_repeat_count = args.gripper_close_repeat_count = 1
    args.ignore_grip_error = False


def run_fr3_builtin(operation, args):
    hw = build_hardware(args.hardware_config, execute=args.execute_robot,
                        speed_scaling=getattr(args, "arm_speed_scaling", None), agent=True)
    try:
        if operation == "home" and hw.config.arm.home_pose is None:
            raise RuntimeError("arm.home_pose 尚未配置；请先通过 hardware.cli 测试法兰位姿")
        if not args.execute_robot:
            print(f"[builtin] FR3/Robotiq {operation}: dry-run；需 --execute_robot 才执行")
            return
        result = hw.arm.home() if operation == "home" else getattr(hw.gripper, operation)().to_dict()
        print(result)
    finally:
        hw.disconnect()
