from dataclasses import dataclass
import math
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Pose6D:
    """Absolute flange pose: metres, fixed-axis XYZ RPY radians (Rz Ry Rx)."""

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    base_frame: str = "fr3_link0"
    target_link: str = "fr3_link8"

    def __post_init__(self):
        if not all(math.isfinite(v) for v in self.values):
            raise ValueError("Pose must contain six finite metre/radian values")
        if (self.base_frame, self.target_link) != ("fr3_link0", "fr3_link8"):
            raise ValueError("This adapter currently accepts fr3_link0 -> fr3_link8 only")

    @property
    def values(self):
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]


class Arm(Protocol):
    def get_pose(self) -> Pose6D: ...
    def plan_pose(self, pose: Pose6D) -> dict: ...
    def move_pose(self, pose: Pose6D) -> dict: ...


class Transforms(Protocol):
    """Calibrated camera/tool transforms; matrices use metres.

    Column-vector convention: p_parent = T_parent_child @ p_child.
    camera_grasp is T_camera_grasp in the calibrated camera optical frame.
    current_flange is fr3_link0 -> fr3_link8 at image acquisition time for an
    eye-in-hand camera. The provider must apply the model grasp-frame/TCP
    alignment and T_flange_tcp; a collision-envelope length is not a TCP.
    device_serial and optical_frame identify the source capture. See
    config/calibration/fr3_handeye.yaml for current calibration and pending TCP.
    """

    def require_ready(self): ...

    def camera_point_to_base(self, point, camera, current_flange, *, device_serial, optical_frame): ...

    def camera_grasp_to_flange_pose(
        self, camera_grasp: Sequence[Sequence[float]], camera: str, current_flange: Pose6D,
        *, device_serial: str, optical_frame: str
    ) -> Pose6D: ...

    def tcp_to_flange_pose(self, base_tcp: Sequence[Sequence[float]]) -> Pose6D: ...


class PendingTransforms:
    def require_ready(self):
        raise RuntimeError(
            "FR3 手眼标定、TCP 变换和旧技能的坐标迁移尚未完成；不会套用 xArm 标定。"
            "当前请用 python -m hardware.cli arm plan/move 测试基座系法兰位姿。"
            "任务图可用 --no_run_executor 生成。"
        )

    def camera_point_to_base(self, point, camera, current_flange, **kwargs):
        self.require_ready()

    def camera_grasp_to_flange_pose(self, camera_grasp, camera, current_flange, **kwargs):
        self.require_ready()

    def tcp_to_flange_pose(self, base_tcp):
        self.require_ready()
