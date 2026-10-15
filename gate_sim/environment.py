"""PyBullet implementation of GateOccludedPickPlace-v0.

The module is simulation-only and runs a single DIRECT environment.  It
implements macro actions over a physically simulated Panda.  No real-robot
service is imported or reachable from this package.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont

from gate_sim.config import GateSimConfig
from gate_sim.perception import FEATURE_SCHEMA, Estimate, PolicyMemory, perceive_rgbd
from gate_sim.textures import TEXTURE_VERSION, ensure_texture


class GateAction(str, Enum):
    EXECUTE = "EXECUTE"
    VISUAL_PROBE = "VISUAL_PROBE"
    MANIPULATION_PROBE = "MANIPULATION_PROBE"
    REPLAN = "REPLAN"


@dataclass
class Snapshot:
    state_id: int
    episode_id: str
    seed: int
    scenario_type: str
    target_key: str
    memory: PolicyMemory
    active_camera: int
    counters: Dict[str, int]
    sim_time_seconds: float
    compute_time_seconds: float
    rng_state: dict
    held_constraint_id: Optional[int]
    held_body_id: Optional[int]
    stable_success_steps: int
    decision_counter: int
    frame_count: int
    latest_observation: Optional[dict]
    controller_joint_targets: Tuple[float, ...]
    intentionally_moved: Tuple[str, ...]


@dataclass
class MacroResult:
    action: str
    available: bool
    primitive_success: Optional[bool]
    probe_effect: Optional[Dict[str, Any]]
    task_success: bool
    failure_reason: Optional[str]
    action_sim_time_seconds: float
    total_sim_time_seconds: float
    compute_time_seconds: float
    total_compute_time_seconds: float
    decision_id: str
    executor_feedback: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


class GateOccludedPickPlaceEnv:
    """Recoverable high-level environment with RGB-D policy observations."""

    ENV_ID = "GateOccludedPickPlace-v0"
    TARGET_KEY = "target_red"
    DAILY_TARGET_KEY = "target_daily_oat_box"
    ARM_JOINTS = tuple(range(7))
    LEFT_FINGER_LINK = 9
    RIGHT_FINGER_LINK = 10
    EE_LINK = 11

    def __init__(
        self,
        config: Optional[GateSimConfig] = None,
        output_dir: str = "runs/gate_sim",
        record_video: bool = False,
        video_view: str = "policy",
        video_resolution: Tuple[int, int] = (1280, 720),
    ):
        self.config = config or GateSimConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_video = bool(record_video)
        if video_view not in {"policy", "overview"}:
            raise ValueError("video_view must be 'policy' or 'overview'")
        self.video_view = video_view
        self.video_resolution = tuple(int(value) for value in video_resolution)
        self.client_id = p.connect(p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("failed to connect to PyBullet DIRECT")
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        self.robot_id: Optional[int] = None
        self.object_bodies: Dict[str, int] = {}
        self.object_metadata: Dict[str, dict] = {}
        self._object_half_heights: Dict[str, float] = {}
        self._texture_ids: Dict[str, int] = {}
        self.fixed_bodies: Dict[str, List[int]] = {}
        self.memory = PolicyMemory()
        self.latest_observation: Optional[dict] = None
        self.seed = 0
        self.scenario_type = "A"
        self.target_key = self.TARGET_KEY
        self.episode_id = ""
        self.active_camera = 0
        self.counters = {"observations": 0, "visual_probes": 0, "manipulation_probes": 0, "drops": 0, "normal_grasp_contacts": 0, "unexpected_disturbances": 0}
        self.sim_time_seconds = 0.0
        self.compute_time_seconds = 0.0
        self.decision_counter = 0
        self.stable_success_steps = 0
        self.held_constraint_id: Optional[int] = None
        self.held_body_id: Optional[int] = None
        self.rng = np.random.default_rng(0)
        self._frames: List[np.ndarray] = []
        self._frame_count = 0
        self._initial_object_poses: Dict[str, Tuple[float, float, float]] = {}
        self._arm_targets: List[float] = [0.0] * 7
        self._finger_targets: List[float] = [self.config.skill.open_width_m] * 2
        self._closed = False
        self._intentionally_moved: set = set()

    def close(self) -> None:
        if not self._closed:
            p.disconnect(physicsClientId=self.client_id)
            self._closed = True

    def __enter__(self) -> "GateOccludedPickPlaceEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ---------------------------- construction ----------------------------
    def reset(self, seed: int, scenario_type: Optional[str] = None) -> dict:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.scenario_type = scenario_type or self.config.default_scene_cycle[
            self.seed % len(self.config.default_scene_cycle)
        ]
        if self.scenario_type not in self.config.scene_types:
            raise ValueError(f"unknown scenario_type {self.scenario_type!r}")
        self.target_key = self.DAILY_TARGET_KEY if self.scenario_type == "E" else self.TARGET_KEY
        self.episode_id = f"gate-sim-{self.seed:06d}-{self.scenario_type}-{uuid.uuid4().hex[:8]}"
        self.active_camera = 0
        self.counters = {"observations": 0, "visual_probes": 0, "manipulation_probes": 0, "drops": 0, "normal_grasp_contacts": 0, "unexpected_disturbances": 0}
        self.sim_time_seconds = 0.0
        self.compute_time_seconds = 0.0
        self.decision_counter = 0
        self.stable_success_steps = 0
        self.held_constraint_id = None
        self.held_body_id = None
        self.memory = PolicyMemory()
        self.latest_observation = None
        self._frames = []
        self._frame_count = 0
        self._intentionally_moved = set()
        self.object_bodies = {}
        self.object_metadata = {}
        self._object_half_heights = {}
        self._texture_ids = {}
        self.fixed_bodies = {}

        p.resetSimulation(physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            numSolverIterations=self.config.physics.solver_iterations,
            deterministicOverlappingPairs=1,
            physicsClientId=self.client_id,
        )
        p.setTimeStep(self.config.physics.timestep_seconds, physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        self._build_table_regions_and_robot()
        self._spawn_scenario()
        self._settle(self.config.physics.settle_steps)
        stable, details = self._check_scene_stability()
        if not stable:
            raise RuntimeError(f"scene failed initial settle check: {details}")
        self._initial_object_poses = {key: self._body_position(body) for key, body in self.object_bodies.items()}
        self._move_to_viewpoint(0)
        self._observe(reason="reset")
        self._write_manifest()
        return self.get_gate_observation()

    def _build_table_regions_and_robot(self) -> None:
        table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.72, 0.58, 0.025], physicsClientId=self.client_id)
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.72, 0.58, 0.025], rgbaColor=[0.55, 0.38, 0.22, 1], physicsClientId=self.client_id)
        table = p.createMultiBody(0, table_shape, table_visual, basePosition=[0.05, 0, -0.025], physicsClientId=self.client_id)
        self.fixed_bodies["table"] = [table]

        # Known buffer: a shallow cyan pad, capacity enforced in policy state.
        bx, by, bz = self.config.workspace.buffer_center
        bhx, bhy = self.config.workspace.buffer_half_extents
        bcol = p.createCollisionShape(p.GEOM_BOX, halfExtents=[bhx, bhy, 0.004], physicsClientId=self.client_id)
        bvis = p.createVisualShape(p.GEOM_BOX, halfExtents=[bhx, bhy, 0.004], rgbaColor=[0.1, 0.75, 0.75, 1], physicsClientId=self.client_id)
        buffer_body = p.createMultiBody(0, bcol, bvis, basePosition=[bx, by, bz - 0.004], physicsClientId=self.client_id)
        self.fixed_bodies["buffer"] = [buffer_body]

        # Fixed shallow goal container: bottom plus four collision walls.
        gx, gy, gz = self.config.workspace.goal_center
        ghx, ghy = self.config.workspace.goal_inner_half_extents
        wall_t = 0.008
        wall_h = self.config.workspace.goal_wall_height
        goal_parts: List[int] = []
        bottom_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ghx, ghy, 0.004], physicsClientId=self.client_id)
        goal_bottom_color = [0.55, 0.22, 0.62, 1] if self.scenario_type == "E" else [0.65, 0.65, 0.68, 1]
        bottom_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[ghx, ghy, 0.004], rgbaColor=goal_bottom_color, physicsClientId=self.client_id)
        goal_parts.append(p.createMultiBody(0, bottom_col, bottom_vis, [gx, gy, gz - 0.004], physicsClientId=self.client_id))
        for half, pos in (
            ([ghx + wall_t, wall_t, wall_h], [gx, gy - ghy - wall_t, gz + wall_h]),
            ([ghx + wall_t, wall_t, wall_h], [gx, gy + ghy + wall_t, gz + wall_h]),
            ([wall_t, ghy, wall_h], [gx - ghx - wall_t, gy, gz + wall_h]),
            ([wall_t, ghy, wall_h], [gx + ghx + wall_t, gy, gz + wall_h]),
        ):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client_id)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.50, 0.50, 0.54, 1], physicsClientId=self.client_id)
            goal_parts.append(p.createMultiBody(0, col, vis, pos, physicsClientId=self.client_id))
        self.fixed_bodies["goal_container"] = goal_parts

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[-0.48, 0.0, 0.0],
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        home = [0.0, -0.55, 0.0, -2.30, 0.0, 1.78, 0.78]
        self._arm_targets = list(home)
        self._finger_targets = [self.config.skill.open_width_m] * 2
        for joint, value in zip(self.ARM_JOINTS, home):
            p.resetJointState(self.robot_id, joint, value, physicsClientId=self.client_id)
        for joint in (9, 10):
            p.resetJointState(self.robot_id, joint, self.config.skill.open_width_m, physicsClientId=self.client_id)
        p.changeDynamics(self.robot_id, self.LEFT_FINGER_LINK, lateralFriction=2.0, spinningFriction=0.02, physicsClientId=self.client_id)
        p.changeDynamics(self.robot_id, self.RIGHT_FINGER_LINK, lateralFriction=2.0, spinningFriction=0.02, physicsClientId=self.client_id)

    def _spawn_box(self, key: str, position: Tuple[float, float, float], half_extents: Tuple[float, float, float], yaw: float = 0.0) -> None:
        rgb = self.config.object_colors_rgb[key]
        color = [channel / 255.0 for channel in rgb] + [1.0]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=self.client_id)
        body = p.createMultiBody(
            baseMass=0.12,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
            physicsClientId=self.client_id,
        )
        p.changeDynamics(body, -1, lateralFriction=0.85, rollingFriction=0.003, spinningFriction=0.003, restitution=0.0, physicsClientId=self.client_id)
        self.object_bodies[key] = body
        self._object_half_heights[key] = float(half_extents[2])
        self.object_metadata[key] = {
            "semantic_category": "target_carton" if key == self.target_key else "plain_blocker_carton",
            "shape": "box",
            "half_extents_m": list(half_extents),
            "texture": None,
        }

    def _spawn_textured_daily_object(self, key: str, spec: Dict[str, Any]) -> None:
        """Spawn a dynamic primitive with metric collision and a local texture."""

        shape = str(spec["shape"])
        position = np.asarray(spec["position_m"], dtype=float)
        position[:2] += self.rng.uniform(-0.004, 0.004, size=2)
        yaw = float(spec.get("yaw_rad", 0.0))
        rgb = self.config.object_colors_rgb[key]
        rgba = [channel / 255.0 for channel in rgb] + [1.0]
        if shape == "box":
            half_extents = tuple(float(value) for value in spec["half_extents_m"])
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id
            )
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba,
                physicsClientId=self.client_id,
            )
            half_height = half_extents[2]
        elif shape == "cylinder":
            radius = float(spec["radius_m"])
            half_height = float(spec["half_height_m"])
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=radius, height=2.0 * half_height,
                physicsClientId=self.client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER, radius=radius, length=2.0 * half_height,
                rgbaColor=rgba, physicsClientId=self.client_id,
            )
        else:
            raise ValueError("unsupported daily object shape {!r}".format(shape))
        body = p.createMultiBody(
            baseMass=float(spec["mass_kg"]),
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position.tolist(),
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            physicsClientId=self.client_id,
        )
        texture_name = str(spec["texture"])
        texture_path = ensure_texture(self.output_dir / "_procedural_textures", texture_name)
        texture_id = p.loadTexture(str(texture_path), physicsClientId=self.client_id)
        p.changeVisualShape(
            body, -1, textureUniqueId=texture_id, rgbaColor=[1.0, 1.0, 1.0, 1.0],
            physicsClientId=self.client_id,
        )
        p.changeDynamics(
            body, -1, lateralFriction=0.90, rollingFriction=0.008,
            spinningFriction=0.008, restitution=0.0, physicsClientId=self.client_id,
        )
        self.object_bodies[key] = body
        self._object_half_heights[key] = half_height
        self._texture_ids[key] = texture_id
        self.object_metadata[key] = {
            "semantic_category": spec["semantic_category"],
            "shape": shape,
            "half_height_m": half_height,
            "texture": texture_name,
            "texture_path": str(texture_path),
            "texture_version": TEXTURE_VERSION,
            "asset_source": "locally_generated_trademark_free_texture_and_pybullet_primitive",
        }

    def _spawn_daily_clutter(self, excluded: Iterable[str] = ()) -> None:
        if not self.config.textured_daily_clutter_enabled:
            return
        excluded = set(excluded)
        for key, spec in self.config.daily_object_catalog.items():
            if key in excluded:
                continue
            self._spawn_textured_daily_object(key, spec)

    def _spawn_scenario(self) -> None:
        target_h = self.config.workspace.target_half_extents
        blocker_h = self.config.workspace.distractor_half_extents
        jitter = self.rng.uniform(-0.006, 0.006, size=2)
        if self.scenario_type == "A":
            self._spawn_box("target_red", (0.12 + jitter[0], 0.01 + jitter[1], target_h[2]), target_h, yaw=0.08)
            self._spawn_box("distractor_blue", (0.20, -0.15, blocker_h[2]), blocker_h, yaw=-0.15)
        elif self.scenario_type == "B":
            self._spawn_box("target_red", (0.14 + jitter[0], 0.075 + jitter[1], target_h[2]), target_h)
            self._spawn_box("distractor_blue", (0.14, -0.012, blocker_h[2]), blocker_h, yaw=0.03)
            self._spawn_box("distractor_green", (0.27, -0.11, blocker_h[2]), blocker_h, yaw=0.20)
        elif self.scenario_type == "C":
            # Stable physical stack: moving the blue box exposes the red target.
            self._spawn_box("target_red", (0.14 + jitter[0], 0.025 + jitter[1], target_h[2]), target_h)
            top_z = 2.0 * target_h[2] + blocker_h[2] + 0.001
            self._spawn_box("distractor_blue", (0.14 + jitter[0], 0.025 + jitter[1], top_z), blocker_h, yaw=0.02)
            self._spawn_box("distractor_green", (0.27, -0.12, blocker_h[2]), blocker_h, yaw=-0.15)
        elif self.scenario_type == "D":
            self._spawn_box("target_red", (0.12 + jitter[0], 0.00 + jitter[1], target_h[2]), target_h, yaw=0.04)
            self._spawn_box("distractor_blue", (0.16, -0.120, blocker_h[2]), blocker_h, yaw=-0.05)
            self._spawn_box("distractor_green", (0.27, -0.14, blocker_h[2]), blocker_h, yaw=0.18)
            self._spawn_box("distractor_yellow", (0.02, 0.15, blocker_h[2]), blocker_h, yaw=-0.20)
        elif self.scenario_type == "E":
            daily_target_half = (0.042, 0.030, 0.026)
            daily_target_spec = {
                "semantic_category": "target_food_carton",
                "shape": "box",
                "half_extents_m": daily_target_half,
                "position_m": (0.14 + jitter[0], 0.025 + jitter[1], daily_target_half[2]),
                "yaw_rad": 0.01,
                "mass_kg": 0.11,
                "texture": "oat_box",
            }
            self._spawn_textured_daily_object(self.target_key, daily_target_spec)
            target_position = self._body_position(self.object_bodies[self.target_key])
            blocker_z = 2.0 * daily_target_half[2] + blocker_h[2] + 0.001
            self._spawn_box(
                "distractor_blue",
                (target_position[0], target_position[1], blocker_z),
                blocker_h,
                yaw=0.02,
            )
        self._spawn_daily_clutter(excluded={"daily_oat_box"} if self.scenario_type == "E" else ())
        self._assert_no_initial_penetration()

    def _assert_no_initial_penetration(self) -> None:
        items = list(self.object_bodies.items())
        for index, (name_a, body_a) in enumerate(items):
            for name_b, body_b in items[index + 1 :]:
                contacts = p.getClosestPoints(body_a, body_b, distance=0.0, physicsClientId=self.client_id)
                worst = min((float(point[8]) for point in contacts), default=0.0)
                # Scenario C is intentionally touching/stacked, but not deeply penetrating.
                if worst < -0.0015:
                    raise RuntimeError(f"initial penetration {name_a}/{name_b}: {worst:.6f} m")

    # ------------------------------- sensing ------------------------------
    def _camera_matrices(self) -> Tuple[list, list]:
        eye = self.config.camera.viewpoints[self.active_camera]
        view = p.computeViewMatrix(eye, self.config.camera.look_at, [0, 0, 1])
        projection = p.computeProjectionMatrixFOV(
            self.config.camera.fov_degrees,
            self.config.camera.width / self.config.camera.height,
            self.config.camera.near,
            self.config.camera.far,
        )
        return view, projection

    def _render(self) -> dict:
        view, projection = self._camera_matrices()
        width, height, rgba, depth, segmentation = p.getCameraImage(
            self.config.camera.width,
            self.config.camera.height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.client_id,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape((height, width, 4))[..., :3]
        depth_buffer = np.asarray(depth, dtype=np.float32).reshape((height, width))
        near, far = self.config.camera.near, self.config.camera.far
        depth_m = far * near / (far - (far - near) * depth_buffer)
        del segmentation
        return {"rgb": rgb, "depth_m": depth_m, "depth_buffer": depth_buffer, "view_matrix": view, "projection_matrix": projection}

    def render_overview(self, width: int = 1280, height: int = 720) -> np.ndarray:
        """Render a presentation view without changing policy observation state."""

        if self.robot_id is None or not self.object_bodies:
            raise RuntimeError("reset must be called before rendering an overview")
        if width < 320 or height < 240:
            raise ValueError("overview resolution must be at least 320x240")
        view = p.computeViewMatrix(
            cameraEyePosition=[0.88, -1.18, 0.78],
            cameraTargetPosition=[0.02, 0.0, 0.16],
            cameraUpVector=[0.0, 0.0, 1.0],
        )
        projection = p.computeProjectionMatrixFOV(
            fov=52.0,
            aspect=float(width) / float(height),
            nearVal=0.05,
            farVal=3.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.client_id,
        )
        return np.asarray(rgba, dtype=np.uint8).reshape((height, width, 4))[..., :3]

    def _observe(self, reason: str) -> dict:
        rendered = self._render()
        self.counters["observations"] += 1
        index = self.counters["observations"]
        estimates = perceive_rgbd(
            rendered["rgb"], rendered["depth_buffer"], rendered["view_matrix"], rendered["projection_matrix"],
            self.active_camera, index, self._object_half_heights,
        )
        # Current camera provides new evidence only for actually detected objects.
        # Absent objects are not fabricated; an old estimate stays stale.
        self.memory.objects.update(estimates)
        if self.active_camera not in self.memory.visited_viewpoints:
            self.memory.visited_viewpoints.append(self.active_camera)
        refs = self._save_observation_artifacts(index, rendered["rgb"], rendered["depth_m"])
        summary = {
            "observation_index": index,
            "camera_index": self.active_camera,
            "reason": reason,
            "detected_objects": sorted(estimates),
            "artifact_refs": refs,
        }
        self.memory.observations.append(summary)
        self.latest_observation = {**rendered, **summary, "estimates": estimates}
        if self.record_video:
            self._frames.append(self._video_frame(rendered["rgb"]))
        return self.latest_observation

    def _save_observation_artifacts(self, index: int, rgb: np.ndarray, depth_m: np.ndarray) -> Dict[str, str]:
        episode_dir = self.output_dir / self.episode_id / "observations"
        episode_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = episode_dir / f"obs_{index:03d}_cam_{self.active_camera}.png"
        depth_path = episode_dir / f"obs_{index:03d}_depth.npy"
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth_m)
        return {"rgb": str(rgb_path), "depth": str(depth_path)}

    def get_gate_observation(self) -> dict:
        if self.latest_observation is None:
            raise RuntimeError("reset must be called first")
        return {
            "policy_observation": {
                "rgb": self.latest_observation["rgb"].copy(),
                "depth_m": self.latest_observation["depth_m"].copy(),
                "camera_index": self.active_camera,
                "robot_state": self._robot_state(),
                "policy_counters": {
                    "observations": int(self.counters["observations"]),
                    "visual_probes": int(self.counters["visual_probes"]),
                    "manipulation_probes": int(self.counters["manipulation_probes"]),
                },
                "task_state": {
                    "node": self.memory.task_node,
                    "task_progress": {
                        "grasp_target": 0.0,
                        "place_target": 0.5,
                        "verify_task": 1.0,
                    }.get(self.memory.task_node),
                    "sim_time_seconds": float(self.sim_time_seconds),
                    "remaining_sim_time_fraction": max(
                        0.0,
                        1.0
                        - float(self.sim_time_seconds)
                        / float(self.config.physics.task_time_limit_seconds),
                    ),
                },
                "known_regions": {
                    "goal_container": asdict(self.config.workspace)["goal_center"],
                    "goal_for_object": self.target_key,
                    "buffer": asdict(self.config.workspace)["buffer_center"],
                    "search_region": {
                        "center": self.config.workspace.search_region_center,
                        "half_extents": self.config.workspace.search_region_half_extents,
                    },
                },
                "current_estimates": {key: value.to_dict() for key, value in self.latest_observation["estimates"].items()},
                "memory": self.memory.to_dict(),
                "feature_schema": deepcopy(FEATURE_SCHEMA),
                "artifact_refs": deepcopy(self.latest_observation["artifact_refs"]),
            },
            "action_mask": self.action_mask(),
        }

    # ---------------------------- high-level API --------------------------
    def action_mask(self) -> Dict[str, bool]:
        if (
            self.sim_time_seconds >= self.config.physics.task_time_limit_seconds
            or self.compute_time_seconds >= self.config.physics.compute_time_limit_seconds
        ):
            return {action.value: False for action in GateAction}
        if self.memory.task_node == "verify_task":
            return {action.value: False for action in GateAction}
        target = self.memory.objects.get(self.target_key)
        target_ready = bool(
            target
            and not target.stale_pose
            and target.field_validity.get("position_m", False)
            and target.graspable
            and target.visibility >= 0.90
        )
        execute = bool(self.memory.gripper_holding == self.target_key or (self.memory.task_node == "grasp_target" and target_ready))
        visual = bool(
            self.counters["visual_probes"] < self.config.skill.max_visual_probes
            and len(self.memory.visited_viewpoints) < len(self.config.camera.viewpoints)
            and self.memory.gripper_holding is None
        )
        manipulation = bool(
            self.counters["manipulation_probes"] < self.config.skill.max_manipulation_probes
            and not self.memory.buffer_occupied
            and self.memory.gripper_holding is None
            and self._select_manipulation_candidate() is not None
        )
        return {
            GateAction.EXECUTE.value: execute,
            GateAction.VISUAL_PROBE.value: visual,
            GateAction.MANIPULATION_PROBE.value: manipulation,
            GateAction.REPLAN.value: False,
        }

    def step_gate(self, action: Any) -> dict:
        action = GateAction(action)
        start = time.perf_counter()
        sim_start = self.sim_time_seconds
        self.decision_counter += 1
        decision_id = f"decision-{self.episode_id}-{self.decision_counter:04d}"
        mask = self.action_mask()
        before_memory = self.memory.to_dict()
        before_obs = deepcopy(self.latest_observation["artifact_refs"]) if self.latest_observation else None
        selected_candidate = self._select_manipulation_candidate() if action == GateAction.MANIPULATION_PROBE else None
        snapshot_reference = self._persist_physics_snapshot(decision_id)
        self._append_log({
            "event": "gate_decision",
            "decision_id": decision_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "scene_type": self.scenario_type,
            "target_object_id": self.target_key,
            "scene_config_version": self.config.scene_config_version,
            "current_step": self.memory.task_node,
            "policy_observation_before": before_obs,
            "memory_before": before_memory,
            "candidate_objects": self._candidate_summaries(),
            "candidate_viewpoints": self._candidate_viewpoints(),
            "action_mask": mask,
            "selected_action": action.value,
            "selected_candidate": selected_candidate,
            "snapshot_reference": snapshot_reference,
            "gate_input_features": self.build_gate_input_dict(),
        })

        if not mask[action.value]:
            reason = self._unavailable_reason(action)
            primitive_success: Optional[bool] = None
            probe_effect = None
            feedback = {"executed": False}
            available = False
        else:
            available = True
            before_target = self._target_observation_metrics()
            if action == GateAction.EXECUTE:
                primitive_success, reason, feedback = self._execute_task_node()
            elif action == GateAction.VISUAL_PROBE:
                primitive_success, reason, feedback = self._visual_probe()
            elif action == GateAction.MANIPULATION_PROBE:
                primitive_success, reason, feedback = self._manipulation_probe(selected_candidate)
            else:
                primitive_success, reason, feedback = None, "replan_not_implemented", {"executed": False}
            after_target = self._target_observation_metrics()
            probe_effect = self._probe_effect(action, before_target, after_target, primitive_success)
            self.memory.last_action_result = 1.0 if primitive_success else -1.0

        task_success = self._update_task_success()
        self._update_unexpected_disturbances()
        compute_time = time.perf_counter() - start
        self.compute_time_seconds += compute_time
        result = MacroResult(
            action=action.value,
            available=available,
            primitive_success=primitive_success,
            probe_effect=probe_effect,
            task_success=task_success,
            failure_reason=reason,
            action_sim_time_seconds=self.sim_time_seconds - sim_start,
            total_sim_time_seconds=self.sim_time_seconds,
            compute_time_seconds=compute_time,
            total_compute_time_seconds=self.compute_time_seconds,
            decision_id=decision_id,
            executor_feedback=feedback,
        )
        self._append_log({
            "event": "gate_outcome",
            "decision_id": decision_id,
            "episode_id": self.episode_id,
            "target_object_id": self.target_key,
            "primitive_success": primitive_success,
            "probe_effect": probe_effect,
            "task_success": task_success,
            "execution_failure_type": reason,
            "executor_version": self.config.executor_version,
            "action_compute_time_seconds": compute_time,
            "action_sim_time_seconds": self.sim_time_seconds - sim_start,
            "total_sim_time_seconds": self.sim_time_seconds,
            "total_compute_time_seconds": self.compute_time_seconds,
            "counts": deepcopy(self.counters),
            "policy_observation_after": deepcopy(self.latest_observation["artifact_refs"]) if self.latest_observation else None,
            "memory_after": self.memory.to_dict(),
            "evaluator_metrics": self.get_evaluator_state(),
            "executor_feedback": feedback,
        })
        return result.to_dict()

    def _unavailable_reason(self, action: GateAction) -> str:
        if (
            self.sim_time_seconds >= self.config.physics.task_time_limit_seconds
            or self.compute_time_seconds >= self.config.physics.compute_time_limit_seconds
        ):
            return "task_budget_exhausted"
        if action == GateAction.REPLAN:
            return "replan_not_implemented"
        if self.memory.gripper_holding is not None and action != GateAction.EXECUTE:
            return "gripper_holding_object"
        if action == GateAction.VISUAL_PROBE:
            return "no_unvisited_viewpoint_or_visual_budget_exhausted"
        if action == GateAction.MANIPULATION_PROBE:
            if self.memory.buffer_occupied:
                return "buffer_occupied"
            return "no_visible_graspable_candidate_or_budget_exhausted"
        return "insufficient_target_information"

    def _execute_task_node(self) -> Tuple[bool, Optional[str], dict]:
        if self.memory.task_node == "grasp_target":
            estimate = self.memory.objects.get(self.target_key)
            if not estimate or estimate.position_m is None or estimate.stale_pose:
                return False, "insufficient_target_information", {"executed": False}
            success, feedback = self._physical_grasp(estimate.position_m, self.target_key)
            if success:
                self.memory.gripper_holding = self.target_key
                self.memory.task_node = "place_target"
            return success, None if success else feedback.get("failure_reason", "grasp_failed"), feedback
        if self.memory.task_node == "place_target" and self.memory.gripper_holding == self.target_key:
            success, feedback = self._physical_place(self.config.workspace.goal_center, destination="goal")
            if success:
                self.memory.gripper_holding = None
                self.memory.task_node = "verify_task"
            return success, None if success else feedback.get("failure_reason", "place_failed"), feedback
        return False, "task_node_not_executable", {"executed": False, "task_node": self.memory.task_node}

    def _visual_probe(self) -> Tuple[bool, Optional[str], dict]:
        unvisited = [index for index in range(len(self.config.camera.viewpoints)) if index not in self.memory.visited_viewpoints]
        if not unvisited:
            return False, "no_unvisited_viewpoint", {"executed": False}
        target_index = unvisited[0]
        moved, motion = self._move_to_viewpoint(target_index)
        if not moved:
            return False, "viewpoint_unreachable", motion
        self.counters["visual_probes"] += 1
        self.memory.invalidate_after_change(pose=True, occlusion=True)
        self.active_camera = target_index
        observation = self._observe(reason="visual_probe")
        return True, None, {"executed": True, "viewpoint": target_index, "motion": motion, "observation": observation["artifact_refs"]}

    def _manipulation_probe(self, candidate: Optional[str]) -> Tuple[bool, Optional[str], dict]:
        if candidate is None:
            return False, "no_visible_graspable_candidate", {"executed": False}
        estimate = self.memory.objects[candidate]
        if estimate.position_m is None or estimate.stale_pose:
            return False, "candidate_pose_stale", {"executed": False, "candidate": candidate}
        self.counters["manipulation_probes"] += 1
        grasped, grasp_feedback = self._physical_grasp(estimate.position_m, candidate)
        if not grasped:
            return False, grasp_feedback.get("failure_reason", "probe_grasp_failed"), {"candidate": candidate, "grasp": grasp_feedback}
        self.memory.gripper_holding = candidate
        placed, place_feedback = self._physical_place(self.config.workspace.buffer_center, destination="buffer")
        if not placed:
            return False, place_feedback.get("failure_reason", "probe_place_failed"), {"candidate": candidate, "grasp": grasp_feedback, "place": place_feedback}
        self.memory.gripper_holding = None
        self.memory.buffer_occupied = True
        self._intentionally_moved.add(candidate)
        self.memory.invalidate_after_change(pose=True, occlusion=True)
        moved, motion = self._move_to_viewpoint(self.active_camera)
        observation = self._observe(reason="manipulation_probe_reobserve")
        return True, None, {
            "executed": True,
            "candidate": candidate,
            "grasp": grasp_feedback,
            "place": place_feedback,
            "return_to_view": motion,
            "return_motion_success": moved,
            "observation": observation["artifact_refs"],
        }

    def _select_manipulation_candidate(self) -> Optional[str]:
        """Shared deterministic selector using policy memory only."""
        cx, cy = self.config.workspace.search_region_center
        ranked = []
        for key, estimate in self.memory.objects.items():
            if key == self.target_key or estimate.position_m is None or estimate.stale_pose or not estimate.graspable:
                continue
            x, y, _ = estimate.position_m
            distance = math.hypot(x - cx, y - cy)
            ranked.append((distance, -estimate.visibility, key))
        return min(ranked)[2] if ranked else None

    # -------------------------- physical skills ---------------------------
    def _move_to_viewpoint(self, index: int) -> Tuple[bool, dict]:
        requested = self.config.camera.viewpoints[index]
        home_motion = self._move_arm_home()
        success, motion = self._move_ee(requested, p.getQuaternionFromEuler([math.pi, 0, 0]))
        actual = p.getLinkState(self.robot_id, self.EE_LINK, computeForwardKinematics=True, physicsClientId=self.client_id)[4]
        error = float(np.linalg.norm(np.asarray(actual) - np.asarray(requested)))
        motion.update({"requested_position_m": list(requested), "actual_position_m": list(actual), "position_error_m": error, "home_transit": home_motion})
        # View sensing tolerates a slightly larger Cartesian error than grasp.
        return bool(error < 0.045), motion

    def _move_arm_home(self) -> dict:
        """Collision-safe joint-space transit used between distant work zones."""
        home = np.asarray([0.0, -0.55, 0.0, -2.30, 0.0, 1.78, 0.78], dtype=float)
        start = np.asarray([p.getJointState(self.robot_id, joint, physicsClientId=self.client_id)[0] for joint in self.ARM_JOINTS])
        for step in range(self.config.skill.motion_steps):
            alpha = (step + 1) / self.config.skill.motion_steps
            target = (start * (1.0 - alpha) + home * alpha).tolist()
            p.setJointMotorControlArray(
                self.robot_id, self.ARM_JOINTS, p.POSITION_CONTROL,
                targetPositions=target, forces=[120] * 7,
                positionGains=[0.08] * 7, velocityGains=[1.0] * 7,
                physicsClientId=self.client_id,
            )
            self._arm_targets = list(target)
            self._step_physics(capture=(step % 24 == 0))
        actual = np.asarray([p.getJointState(self.robot_id, joint, physicsClientId=self.client_id)[0] for joint in self.ARM_JOINTS])
        return {"joint_error_l2_rad": float(np.linalg.norm(actual - home))}

    def _physical_grasp(self, estimated_center: Iterable[float], requested_object: str) -> Tuple[bool, dict]:
        center = np.asarray(tuple(estimated_center), dtype=float)
        self._command_fingers(self.config.skill.open_width_m)
        orientation = p.getQuaternionFromEuler([math.pi, 0, 0])
        approach = center.copy(); approach[2] += self.config.skill.approach_height_m
        grasp = center.copy(); grasp[2] += self.config.skill.grasp_clearance_m
        ok_approach, a = self._move_ee(approach, orientation)
        ok_descend, d = self._move_ee(grasp, orientation)
        if not (ok_approach and ok_descend):
            return False, {"failure_reason": "ik_or_motion_failed", "approach": a, "descend": d, "requested_object": requested_object}
        self._command_fingers(self.config.skill.closed_width_m)
        left = {point[2] for point in p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.LEFT_FINGER_LINK, physicsClientId=self.client_id) if point[2] in self.object_bodies.values()}
        right = {point[2] for point in p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.RIGHT_FINGER_LINK, physicsClientId=self.client_id) if point[2] in self.object_bodies.values()}
        common = sorted(left & right)
        bilateral = bool(common)
        if bilateral:
            self.counters["normal_grasp_contacts"] += 1
        if self.config.skill.bilateral_contact_required and not bilateral:
            self._move_ee(approach, orientation)
            return False, {
                "failure_reason": "bilateral_finger_contact_not_observed",
                "left_contact_count": len(left),
                "right_contact_count": len(right),
                "bilateral_contact": False,
                "requested_object": requested_object,
            }
        contacted_body = common[0] if common else next(iter(left | right), None)
        if contacted_body is None:
            return False, {"failure_reason": "no_object_contact", "requested_object": requested_object}
        if self.config.skill.contact_latch_enabled:
            self._create_contact_latch(contacted_body)
        lift = center.copy(); lift[2] += self.config.skill.lift_height_m
        ok_lift, lift_feedback = self._move_ee(lift, orientation)
        body_z = self._body_position(contacted_body)[2]
        lifted = bool(body_z > 0.10)
        if not (ok_lift and lifted):
            self._release_latch()
            self.counters["drops"] += 1
            return False, {
                "failure_reason": "lift_verification_failed",
                "bilateral_contact": bilateral,
                "lifted_body_z_m": body_z,
                "lift_motion": lift_feedback,
                "requested_object": requested_object,
            }
        self.held_body_id = contacted_body
        return True, {
            "bilateral_contact": bilateral,
            "contact_latch": self.config.skill.contact_latch_enabled,
            "lifted_body_z_m": body_z,
            "requested_object": requested_object,
            "contacted_body_is_requested": contacted_body == self.object_bodies.get(requested_object),
            "approach": a,
            "descend": d,
            "lift": lift_feedback,
        }

    def _physical_place(self, destination_center: Iterable[float], destination: str) -> Tuple[bool, dict]:
        if self.held_body_id is None:
            return False, {"failure_reason": "gripper_not_holding_physical_body"}
        center = np.asarray(tuple(destination_center), dtype=float)
        orientation = p.getQuaternionFromEuler([math.pi, 0, 0])
        above = center.copy(); above[2] = 0.24
        release = center.copy(); release[2] = 0.13
        ok_above, a = self._move_ee(above, orientation)
        ok_release, r = self._move_ee(release, orientation)
        body = self.held_body_id
        self._release_latch()
        self._command_fingers(self.config.skill.open_width_m)
        self._move_ee(above, orientation)
        self._settle(int(0.8 / self.config.physics.timestep_seconds))
        position = self._body_position(body)
        if destination == "goal":
            success = self._body_inside_goal(body) and self._body_speed(body)[0] < self.config.physics.settle_linear_velocity_mps
        else:
            bx, by, _ = self.config.workspace.buffer_center
            hx, hy = self.config.workspace.buffer_half_extents
            success = abs(position[0] - bx) <= hx and abs(position[1] - by) <= hy and position[2] > 0.0
        return bool(ok_above and ok_release and success), {
            "destination": destination,
            "released": True,
            "final_position_m": list(position),
            "inside_destination": bool(success),
            "above_motion": a,
            "release_motion": r,
        }

    def _create_contact_latch(self, body: int) -> None:
        link_state = p.getLinkState(self.robot_id, self.EE_LINK, computeForwardKinematics=True, physicsClientId=self.client_id)
        link_pos, link_orn = link_state[4], link_state[5]
        body_pos, body_orn = p.getBasePositionAndOrientation(body, physicsClientId=self.client_id)
        inv_pos, inv_orn = p.invertTransform(link_pos, link_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, body_pos, body_orn)
        self.held_constraint_id = p.createConstraint(
            self.robot_id, self.EE_LINK, body, -1, p.JOINT_FIXED, [0, 0, 0],
            parentFramePosition=rel_pos, childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_orn, childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.client_id,
        )
        p.changeConstraint(self.held_constraint_id, maxForce=220, physicsClientId=self.client_id)

    def _release_latch(self) -> None:
        if self.held_constraint_id is not None:
            try:
                p.removeConstraint(self.held_constraint_id, physicsClientId=self.client_id)
            except Exception:
                pass
        self.held_constraint_id = None
        self.held_body_id = None

    def _move_ee(self, target_position: Iterable[float], target_orientation: Iterable[float]) -> Tuple[bool, dict]:
        target_position = tuple(float(value) for value in target_position)
        target_orientation = tuple(float(value) for value in target_orientation)
        start = np.asarray(p.getLinkState(self.robot_id, self.EE_LINK, computeForwardKinematics=True, physicsClientId=self.client_id)[4])
        target = np.asarray(target_position)
        for step in range(self.config.skill.motion_steps):
            alpha = (step + 1) / self.config.skill.motion_steps
            waypoint = tuple((start * (1.0 - alpha) + target * alpha).tolist())
            joints = p.calculateInverseKinematics(
                self.robot_id, self.EE_LINK, waypoint, target_orientation,
                lowerLimits=[-2.9, -1.76, -2.9, -3.07, -2.9, -0.02, -2.9],
                upperLimits=[2.9, 1.76, 2.9, -0.07, 2.9, 3.75, 2.9],
                jointRanges=[5.8, 3.52, 5.8, 3.0, 5.8, 3.77, 5.8],
                restPoses=[0, -0.55, 0, -2.30, 0, 1.78, 0.78],
                maxNumIterations=100,
                residualThreshold=1e-5,
                physicsClientId=self.client_id,
            )
            p.setJointMotorControlArray(
                self.robot_id, self.ARM_JOINTS, p.POSITION_CONTROL,
                targetPositions=joints[:7], forces=[120] * 7,
                positionGains=[0.08] * 7, velocityGains=[1.0] * 7,
                physicsClientId=self.client_id,
            )
            self._arm_targets = list(joints[:7])
            self._step_physics(capture=(step % 24 == 0))
        actual = np.asarray(p.getLinkState(self.robot_id, self.EE_LINK, computeForwardKinematics=True, physicsClientId=self.client_id)[4])
        error = float(np.linalg.norm(actual - target))
        return error < 0.045, {"target_position_m": list(target_position), "actual_position_m": actual.tolist(), "position_error_m": error}

    def _command_fingers(self, width: float) -> None:
        for _ in range(self.config.skill.finger_steps):
            p.setJointMotorControlArray(
                self.robot_id, [9, 10], p.POSITION_CONTROL,
                targetPositions=[width, width], forces=[60, 60], positionGains=[0.35, 0.35],
                physicsClientId=self.client_id,
            )
            self._finger_targets = [float(width), float(width)]
            self._step_physics(capture=False)

    def _step_physics(self, capture: bool = False) -> None:
        p.stepSimulation(physicsClientId=self.client_id)
        self.sim_time_seconds += self.config.physics.timestep_seconds
        if capture and self.record_video:
            self._frames.append(self._video_frame())
            self._frame_count += 1

    def _video_frame(self, policy_rgb: Optional[np.ndarray] = None) -> np.ndarray:
        if self.video_view == "overview":
            width, height = self.video_resolution
            return self.render_overview(width, height)
        if policy_rgb is None:
            policy_rgb = self._render()["rgb"]
        return policy_rgb.copy()

    def _settle(self, steps: int) -> None:
        for _ in range(int(steps)):
            self._step_physics(capture=False)

    # ------------------------- evaluation channel -------------------------
    def get_evaluator_state(self) -> dict:
        """Privileged truth for metrics/labels only; never used by selectors."""
        target = self.object_bodies.get(self.target_key)
        object_states = {}
        for key, body in self.object_bodies.items():
            position, orientation = p.getBasePositionAndOrientation(body, physicsClientId=self.client_id)
            linear, angular = p.getBaseVelocity(body, physicsClientId=self.client_id)
            object_states[key] = {
                "position_m": list(position), "orientation_xyzw": list(orientation),
                "linear_velocity_mps": list(linear), "angular_velocity_radps": list(angular),
                "in_goal": self._body_inside_goal(body),
                "object_metadata": deepcopy(self.object_metadata.get(key, {})),
            }
        return {
            "channel": "evaluator_state_privileged",
            "target_object_id": self.target_key,
            "object_states": object_states,
            "target_in_goal": bool(target is not None and self._body_inside_goal(target)),
            "target_released": bool(target is not None and self.held_body_id != target),
            "stable_success_steps": self.stable_success_steps,
            "scene_type": self.scenario_type,
        }

    def _update_task_success(self) -> bool:
        target = self.object_bodies[self.target_key]
        linear, angular = self._body_speed(target)
        stable_now = (
            self._body_inside_goal(target)
            and self.held_body_id != target
            and linear < self.config.physics.settle_linear_velocity_mps
            and angular < self.config.physics.settle_angular_velocity_radps
        )
        if stable_now:
            self.stable_success_steps += int(0.8 / self.config.physics.timestep_seconds)
        else:
            self.stable_success_steps = 0
        required = int(self.config.physics.stable_duration_seconds / self.config.physics.timestep_seconds)
        return self.stable_success_steps >= required

    def _body_inside_goal(self, body: int) -> bool:
        x, y, z = self._body_position(body)
        gx, gy, _ = self.config.workspace.goal_center
        hx, hy = self.config.workspace.goal_inner_half_extents
        return abs(x - gx) <= hx and abs(y - gy) <= hy and 0.0 < z < 0.16

    def _check_scene_stability(self) -> Tuple[bool, dict]:
        velocities = {key: self._body_speed(body) for key, body in self.object_bodies.items()}
        positions = {key: self._body_position(body) for key, body in self.object_bodies.items()}
        stable = all(v[0] <= self.config.physics.settle_linear_velocity_mps and v[1] <= self.config.physics.settle_angular_velocity_radps and positions[key][2] > 0 for key, v in velocities.items())
        return stable, {"velocities": velocities, "positions": positions}

    def _update_unexpected_disturbances(self) -> None:
        disturbed = 0
        for key, body in self.object_bodies.items():
            if key == self.target_key or key in self._intentionally_moved:
                continue
            initial = np.asarray(self._initial_object_poses.get(key, self._body_position(body)))
            current = np.asarray(self._body_position(body))
            if float(np.linalg.norm(current - initial)) > 0.035:
                disturbed += 1
        self.counters["unexpected_disturbances"] = disturbed

    def _body_position(self, body: int) -> Tuple[float, float, float]:
        return tuple(float(v) for v in p.getBasePositionAndOrientation(body, physicsClientId=self.client_id)[0])

    def _body_speed(self, body: int) -> Tuple[float, float]:
        linear, angular = p.getBaseVelocity(body, physicsClientId=self.client_id)
        return float(np.linalg.norm(linear)), float(np.linalg.norm(angular))

    # ------------------------------ snapshots -----------------------------
    def save_snapshot(self) -> Snapshot:
        return Snapshot(
            state_id=p.saveState(physicsClientId=self.client_id),
            episode_id=self.episode_id,
            seed=self.seed,
            scenario_type=self.scenario_type,
            target_key=self.target_key,
            memory=self.memory.clone(),
            active_camera=self.active_camera,
            counters=deepcopy(self.counters),
            sim_time_seconds=self.sim_time_seconds,
            compute_time_seconds=self.compute_time_seconds,
            rng_state=deepcopy(self.rng.bit_generator.state),
            held_constraint_id=self.held_constraint_id,
            held_body_id=self.held_body_id,
            stable_success_steps=self.stable_success_steps,
            decision_counter=self.decision_counter,
            frame_count=len(self._frames),
            latest_observation=deepcopy(self.latest_observation),
            controller_joint_targets=tuple(
                list(self._arm_targets) + list(self._finger_targets)
            ),
            intentionally_moved=tuple(sorted(self._intentionally_moved)),
        )

    def restore_snapshot(self, snapshot: Snapshot, branch_id: Optional[str] = None) -> dict:
        if snapshot.seed != self.seed or snapshot.scenario_type != self.scenario_type:
            raise ValueError("snapshot belongs to a different episode scene")
        p.restoreState(stateId=snapshot.state_id, physicsClientId=self.client_id)
        self.episode_id = snapshot.episode_id if branch_id is None else f"{snapshot.episode_id}-{branch_id}"
        self.memory = snapshot.memory.clone()
        self.target_key = snapshot.target_key
        self.active_camera = snapshot.active_camera
        self.counters = deepcopy(snapshot.counters)
        self.sim_time_seconds = snapshot.sim_time_seconds
        self.compute_time_seconds = snapshot.compute_time_seconds
        self.rng.bit_generator.state = deepcopy(snapshot.rng_state)
        self.held_constraint_id = snapshot.held_constraint_id
        self.held_body_id = snapshot.held_body_id
        self.stable_success_steps = snapshot.stable_success_steps
        self.decision_counter = snapshot.decision_counter
        self._frames = self._frames[: snapshot.frame_count]
        self.latest_observation = deepcopy(snapshot.latest_observation)
        self._arm_targets = list(snapshot.controller_joint_targets[:7])
        self._finger_targets = list(snapshot.controller_joint_targets[7:])
        self._intentionally_moved = set(snapshot.intentionally_moved)
        p.setJointMotorControlArray(
            self.robot_id, self.ARM_JOINTS, p.POSITION_CONTROL,
            targetPositions=list(snapshot.controller_joint_targets[:7]),
            forces=[120] * 7, positionGains=[0.08] * 7, velocityGains=[1.0] * 7,
            physicsClientId=self.client_id,
        )
        p.setJointMotorControlArray(
            self.robot_id, [9, 10], p.POSITION_CONTROL,
            targetPositions=list(snapshot.controller_joint_targets[7:]),
            forces=[60, 60], positionGains=[0.35, 0.35],
            physicsClientId=self.client_id,
        )
        return self.get_gate_observation()

    def physical_state_vector(self) -> np.ndarray:
        values: List[float] = []
        for key in sorted(self.object_bodies):
            body = self.object_bodies[key]
            position, orientation = p.getBasePositionAndOrientation(body, physicsClientId=self.client_id)
            linear, angular = p.getBaseVelocity(body, physicsClientId=self.client_id)
            values.extend(position); values.extend(orientation); values.extend(linear); values.extend(angular)
        for joint in self.ARM_JOINTS + (9, 10):
            state = p.getJointState(self.robot_id, joint, physicsClientId=self.client_id)
            values.extend([state[0], state[1]])
        return np.asarray(values, dtype=np.float64)

    # ------------------------------- records ------------------------------
    def build_gate_input_dict(self) -> dict:
        """23-feature schema adapter matching nesyro_gate.learnable_gate order."""
        # Local import keeps this simulator usable for isolated debugging and
        # proves the order comes from the existing project schema.
        from nesyro_gate.learnable_gate import GATE_FEATURE_NAMES

        target = self.memory.objects.get(self.target_key)
        blocker_key = self._select_manipulation_candidate()
        blocker = self.memory.objects.get(blocker_key) if blocker_key else None
        target_valid = bool(target and not target.stale_pose)
        blocker_valid = bool(blocker and not blocker.stale_pose)
        relation = self._observed_blocker_relation(blocker, target)
        skill_name = "place" if self.memory.task_node == "place_target" else "grasp"
        raw: Dict[str, Any] = {
            "target_confidence": target.identity_confidence if target_valid else None,
            "target_visibility": target.visibility if target and not target.stale_occlusion else None,
            "occlusion_score": (1.0 - target.visibility) if target and not target.stale_occlusion else None,
            "pose_confidence": target.pose_confidence if target_valid else None,
            "blocker_confidence": blocker.identity_confidence if blocker_valid else None,
            "blocker_graspability": float(blocker.graspable) if blocker_valid and blocker.graspable is not None else None,
            "blocker_target_relation": relation,
            "prerequisite_lc": 1.0 if self.action_mask()[GateAction.EXECUTE.value] else 0.0,
            "previous_action_result": self.memory.last_action_result,
            "visual_probe_count": self.counters["visual_probes"],
            "manipulation_probe_count": self.counters["manipulation_probes"],
            "task_progress": {"grasp_target": 0.0, "place_target": 0.5, "verify_task": 0.9}.get(self.memory.task_node),
            "detector_confidence": target.identity_confidence if target_valid else None,
            "ocr_match_score": None,
            "vlm_match_score": None,
            "reachable": float(target.graspable) if target_valid and target.graspable is not None else None,
            "graspable": float(target.graspable) if target_valid and target.graspable is not None else None,
            "in_gripper": float(self.memory.gripper_holding == self.target_key),
            "free_space": 1.0,
            "placeable": 1.0,
            "is_observe": float(skill_name == "observe"),
            "is_grasp": float(skill_name == "grasp"),
            "is_place": float(skill_name == "place"),
        }
        if list(raw) != list(GATE_FEATURE_NAMES):
            raise RuntimeError("simulator feature adapter no longer matches existing 23-D schema")
        return {
            "schema": "existing_nesyro_gate_23d",
            "feature_names": list(GATE_FEATURE_NAMES),
            "raw_values": [raw[name] for name in GATE_FEATURE_NAMES],
            "validity": [raw[name] is not None for name in GATE_FEATURE_NAMES],
            "provenance": {name: FEATURE_SCHEMA.get(name, {"source": "policy memory/task state", "unit": "dimensionless"}) for name in GATE_FEATURE_NAMES},
        }

    def _observed_blocker_relation(self, blocker: Optional[Estimate], target: Optional[Estimate]) -> Optional[float]:
        if not blocker or blocker.stale_pose or blocker.position_m is None:
            return None
        if target and not target.stale_pose and target.position_m is not None:
            distance = math.hypot(blocker.position_m[0] - target.position_m[0], blocker.position_m[1] - target.position_m[1])
        else:
            cx, cy = self.config.workspace.search_region_center
            distance = math.hypot(blocker.position_m[0] - cx, blocker.position_m[1] - cy)
        return float(np.clip(1.0 - distance / 0.20, 0.0, 1.0))

    def _target_observation_metrics(self) -> dict:
        target = self.memory.objects.get(self.target_key)
        if target is None or target.stale_occlusion:
            return {"available": False, "visibility": None, "identity_confidence": None, "graspable": None, "observation_index": None}
        return {
            "available": True,
            "visibility": target.visibility,
            "identity_confidence": target.identity_confidence,
            "graspable": target.graspable,
            "observation_index": target.observation_index,
        }

    @staticmethod
    def _probe_effect(action: GateAction, before: dict, after: dict, primitive_success: Optional[bool]) -> Optional[dict]:
        if action not in (GateAction.VISUAL_PROBE, GateAction.MANIPULATION_PROBE):
            return None
        if not primitive_success:
            return {"available": True, "new_observation": False, "improved_target_confirmation": False, "improved_operability": False}
        visibility_delta = None
        if before["visibility"] is not None and after["visibility"] is not None:
            visibility_delta = after["visibility"] - before["visibility"]
        return {
            "available": True,
            "new_observation": after["observation_index"] != before["observation_index"],
            "visibility_delta": visibility_delta,
            "improved_target_confirmation": bool(after["identity_confidence"] is not None and (before["identity_confidence"] is None or after["identity_confidence"] > before["identity_confidence"] + 0.02)),
            "improved_operability": bool(after["graspable"] and not before["graspable"]),
            "true_information_gain": None,
            "true_information_gain_valid": False,
        }

    def _candidate_summaries(self) -> list:
        return [estimate.to_dict() for key, estimate in sorted(self.memory.objects.items()) if key != self.target_key and not estimate.stale_pose]

    def _candidate_viewpoints(self) -> list:
        return [
            {"index": index, "position_m": list(position), "visited": index in self.memory.visited_viewpoints}
            for index, position in enumerate(self.config.camera.viewpoints)
        ]

    def _robot_state(self) -> dict:
        joints = [p.getJointState(self.robot_id, joint, physicsClientId=self.client_id)[:2] for joint in self.ARM_JOINTS + (9, 10)]
        ee = p.getLinkState(self.robot_id, self.EE_LINK, computeForwardKinematics=True, physicsClientId=self.client_id)
        return {"joint_position_velocity": [list(value) for value in joints], "ee_position_m": list(ee[4]), "ee_orientation_xyzw": list(ee[5])}

    def _append_log(self, record: dict) -> None:
        path = self.output_dir / self.episode_id / "gate_experience.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=self._json_default) + "\n")

    def _persist_physics_snapshot(self, decision_id: str) -> dict:
        """Persist a replay reference; complete branch state remains in-memory."""
        directory = self.output_dir / self.episode_id / "snapshots"
        directory.mkdir(parents=True, exist_ok=True)
        bullet_path = directory / f"{decision_id}.bullet"
        metadata_path = directory / f"{decision_id}.json"
        p.saveBullet(str(bullet_path), physicsClientId=self.client_id)
        metadata = {
            "decision_id": decision_id,
            "seed": self.seed,
            "scene_type": self.scenario_type,
            "target_object_id": self.target_key,
            "active_camera": self.active_camera,
            "counters": self.counters,
            "sim_time_seconds": self.sim_time_seconds,
            "compute_time_seconds": self.compute_time_seconds,
            "remaining_budget": {
                "sim_time_seconds": max(0.0, self.config.physics.task_time_limit_seconds - self.sim_time_seconds),
                "compute_time_seconds": max(0.0, self.config.physics.compute_time_limit_seconds - self.compute_time_seconds),
            },
            "memory": self.memory.to_dict(),
            "complete_restore_scope": "in_process Snapshot also includes controller/RNG/frame state",
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=self._json_default), encoding="utf-8")
        return {"physics": str(bullet_path), "metadata": str(metadata_path)}

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    def _write_manifest(self) -> None:
        path = self.output_dir / self.episode_id / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "env_id": self.ENV_ID,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "scene_type": self.scenario_type,
            "target_object_id": self.target_key,
            "config": self.config.to_dict(),
            "policy_evaluator_boundary": "policy receives RGB-D estimates only; PyBullet body truth is evaluator-only",
            "oracle_debug": False,
            "real_robot_enabled": False,
        }
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def begin_video_segment(self) -> None:
        """Clear earlier frames and start a segment at the current physical state."""

        if not self.record_video:
            raise RuntimeError("record_video must be enabled")
        self._frames = [self._video_frame()]
        self._frame_count = 1

    def save_video(
        self,
        name: str,
        overlay_lines: Iterable[str] = (),
        minimum_duration_seconds: float = 0.0,
    ) -> Optional[str]:
        if not self._frames:
            return None
        path = self.output_dir / self.episode_id / f"{name}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = 8
        frames = self._frames
        lines = [str(value) for value in overlay_lines]
        if lines:
            annotated = []
            for frame in frames:
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", max(16, image.height // 32))
                except OSError:
                    font = ImageFont.load_default()
                line_height = max(22, image.height // 28)
                banner_height = 12 + line_height * len(lines)
                draw.rectangle((0, 0, image.width, banner_height), fill=(20, 24, 32))
                for index, line in enumerate(lines):
                    draw.text((16, 6 + index * line_height), line, font=font, fill=(245, 245, 245))
                annotated.append(np.asarray(image))
            frames = annotated
        minimum_frames = int(math.ceil(float(minimum_duration_seconds) * fps))
        if len(frames) < minimum_frames:
            frames = list(frames) + [frames[-1]] * (minimum_frames - len(frames))
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return str(path)
