"""A real RoboCasa Kitchen subclass with deterministic, physically settled objects."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET

from robocasa.environments.kitchen.kitchen import Kitchen
from robosuite.models.objects import MujocoXMLObject
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.utils.mjcf_utils import array_to_string

from .cameras import ARM_BASE_BODY, LEFT_CAMERA, LeftCameraDeployment, camera_quaternion


SCENARIOS = ("clear", "label_occluded", "fully_occluded")


class OcclusionCounter(Kitchen):
    """Scene construction only. Clearing and grasping policies are not implemented here."""

    def __init__(self, assets, scenario="label_occluded", scene_seed=0,
                 left_camera_deployment=None, **kwargs):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self.scene_assets = assets
        self.carton_size = tuple(assets["cartons"][0]["size_m"])
        self.blocker_sizes = {name: tuple(asset["size_m"]) for name, asset in assets["blockers"].items()}
        self.scenario = scenario
        self.scene_seed = scene_seed
        self.left_camera_deployment = left_camera_deployment or LeftCameraDeployment()
        self.left_camera_setup = None
        self.label_order = np.random.default_rng(scene_seed).permutation(3).tolist()
        self.target_key = f"carton_{self.label_order.index(0)}"
        self.hard_reset = False
        super().__init__(seed=scene_seed, use_object_obs=False, randomize_cameras=False,
                         robot_spawn_deviation_pos_x=0, robot_spawn_deviation_pos_y=0,
                         robot_spawn_deviation_rot=0, use_distractors=False,
                         generative_textures=None, clutter_mode=0, **kwargs)
        self.hard_reset = False

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        # Choose the widest plain counter, deterministically, leaving room for buffers.
        counters = [f for f in self.fixtures.values()
                    if f.__class__.__name__ == "Counter" and f.size[0] > 1.1
                    and f.size[1] > 0.55 and not f.has_opening]
        if not counters:
            raise RuntimeError("This kitchen has no plain counter wider than 1.1 m")
        self.counter = sorted(counters, key=lambda f: (-f.size[0], f.name))[0]
        self.fixture_refs["work_counter"] = (self.counter, False)
        self.init_robot_base_ref = self.counter
        self.counter_rotation = Rotation.from_euler("z", self.counter.rot).as_matrix()
        self.surface_origin = np.asarray(self.counter.pos, float) + [0, 0, self.counter.size[2] / 2]

    def world(self, xyz):
        return self.surface_origin + self.counter_rotation @ np.asarray(xyz, float)

    def _create_objects(self):
        self.objects = {}
        self.object_cfgs = []
        for slot, label_index in enumerate(self.label_order):
            name = f"carton_{slot}"
            self.objects[name] = MujocoXMLObject(
                self.scene_assets["cartons"][label_index]["path"], name=name,
                joints=[dict(type="free", damping="0.0005")],
                obj_type="all", duplicate_collision_geoms=False)
        for kind in self.blocker_sizes:
            name = f"blocker_{kind}"
            self.objects[name] = MujocoXMLObject(
                self.scene_assets["blockers"][kind]["path"], name=name,
                joints=[dict(type="free", damping="0.0005")],
                obj_type="all", duplicate_collision_geoms=False)
        quat = Rotation.from_matrix(self.counter_rotation).as_quat()[[3, 0, 1, 2]]
        for name, xyz in self._placements(self.scenario).items():
            self.objects[name].get_obj().set("pos", array_to_string(self.world(xyz)))
            self.objects[name].get_obj().set("quat", array_to_string(quat))
        self.model.merge_objects(list(self.objects.values()))
        self._add_workcell_visuals()

    def _add_workcell_visuals(self):
        quat = Rotation.from_matrix(self.counter_rotation).as_quat()[[3, 0, 1, 2]]
        for name, x, color in [("buffer", -0.43, "0.2 0.6 0.65 1"),
                               ("goal", 0.43, "0.55 0.35 0.65 1")]:
            ET.SubElement(self.model.worldbody, "geom", name=f"{name}_pad", type="box",
                          size="0.13 0.24 0.0003", pos=array_to_string(self.world([x, 0, 0.0004])),
                          quat=array_to_string(quat), rgba=color, group="1", contype="0", conaffinity="0")
        # Named cameras are fixed in the world; wrist camera comes from PandaOmron.
        target = self.world([0, -0.015, 0.025])
        for name, eye_local, fovy in [
            ("inspection", [0, -0.20, 0.90], 51),
            ("overhead", [0, -0.01, 1.10], 55),
            ("overview", [1.5, -2.1, 1.55], 65),
        ]:
            eye = self.world(eye_local)
            quat = camera_quaternion(eye, target)
            if name == "overhead":
                quat = Rotation.from_matrix(self.counter_rotation).as_quat()[[3, 0, 1, 2]]
            ET.SubElement(self.model.worldbody, "camera", name=name,
                          pos=array_to_string(eye), quat=array_to_string(quat),
                          fovy=str(fovy))
        # The robot's physical base pose is only available after Kitchen.reset().
        # A world camera is repositioned then, and never follows subsequent motion.
        ET.SubElement(self.model.worldbody, "camera", name=LEFT_CAMERA,
                      pos=array_to_string(self.world([0, 0, 1])),
                      fovy=str(self.left_camera_deployment.fovy_deg))

    def arm_base_pose(self):
        # robot0_base is a displaced XML root in PandaOmron, not the arm mount.
        body = self.sim.model.body_name2id(ARM_BASE_BODY)
        transform = np.eye(4)
        transform[:3, :3] = self.sim.data.body_xmat[body].reshape(3, 3)
        transform[:3, 3] = self.sim.data.body_xpos[body]
        return transform

    def _place_left_camera(self):
        self.sim.forward()
        setup = self.left_camera_deployment.resolve(
            self.arm_base_pose(), self.world([0, -0.015, 0.025]))
        camera = self.sim.model.camera_name2id(LEFT_CAMERA)
        self.sim.model.cam_pos[camera] = setup["position_world_m"]
        self.sim.model.cam_quat[camera] = setup["quaternion_world_wxyz"]
        self.left_camera_setup = setup
        self.sim.forward()

    def _setup_observables(self):
        # Kitchen unconditionally adds object ground-truth poses, even when
        # use_object_obs=False. Bypass that layer for policy-visible observations.
        return ManipulationEnv._setup_observables(self)

    def _reset_internal(self):
        super()._reset_internal()
        self.set_scenario(self.scenario, settle=False)
        self._place_left_camera()

    def _placements(self, scenario):
        slots = [(-0.19, 0.055), (0.0, -0.07), (0.19, 0.055)]
        placements = {f"carton_{i}": (x, y, self.carton_size[2] / 2 + 0.002)
                      for i, (x, y) in enumerate(slots)}
        placements["blocker_small"] = (-0.43, -0.13, self.blocker_sizes["small"][2] / 2 + 0.002)
        placements["blocker_large"] = (-0.43, 0.11, self.blocker_sizes["large"][2] / 2 + 0.002)
        if scenario != "clear":
            kind = "small" if scenario == "label_occluded" else "large"
            x, y = slots[int(self.target_key.rsplit("_", 1)[1])]
            placements[f"blocker_{kind}"] = (x, y, self.carton_size[2] + self.blocker_sizes[kind][2] / 2 + 0.003)
        return placements

    def set_scenario(self, scenario, settle=True):
        """Initialize an experimental condition; this is not an executed robot action."""
        if scenario not in SCENARIOS:
            raise ValueError(scenario)
        self.scenario = scenario
        placements = self._placements(scenario)
        quat = Rotation.from_matrix(self.counter_rotation).as_quat()[[3, 0, 1, 2]]
        for name, xyz in placements.items():
            self.sim.data.set_joint_qpos(self.objects[name].joints[0], np.r_[self.world(xyz), quat])
            self.sim.data.set_joint_qvel(self.objects[name].joints[0], np.zeros(6))
        self.sim.forward()
        if settle:
            self.settle()

    def settle(self, steps=30):
        action = np.zeros(self.action_spec[0].shape)
        for _ in range(steps):
            self.step(action)

    def _check_success(self):
        # No task success claim is made by this scene-construction prototype.
        return False

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "找到布洛芬，清除遮挡后将其放到紫色目标区域。"
        meta["scene_only"] = True
        return meta
