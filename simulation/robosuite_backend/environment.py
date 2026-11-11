"""Two dynamic cubes, a Panda, and two non-colliding table goal markers."""
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_site

OBJECTS = ('red_cube', 'blue_cube')
GOALS = {'red_goal': (0.13, -0.12), 'blue_goal': (0.13, 0.12)}
TASK = tuple(zip(OBJECTS, GOALS))


class PandaTabletop(ManipulationEnv):
    def __init__(self, config):
        self.config = config
        controller = load_composite_controller_config(robot='Panda')
        controller['body_parts']['right'].update(input_type='absolute', input_ref_frame='world')
        super().__init__(
            robots='Panda', controller_configs=controller, initialization_noise=None,
            use_camera_obs=False, has_renderer=False, has_offscreen_renderer=True,
            camera_names=config.camera, control_freq=config.control_freq,
            horizon=10000, ignore_done=True, hard_reset=False, seed=config.seed,
        )

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(
            self.robots[0].robot_model.base_xpos_offset['table'](0.8))
        arena = TableArena(table_full_size=(0.8, 0.8, 0.05),
                           table_friction=(1.0, 0.005, 0.0001),
                           table_offset=(0, 0, self.config.table_height))
        colors = ((0.9, 0.1, 0.1, 1), (0.1, 0.25, 0.9, 1))
        self.objects = {
            name: BoxObject(name=name, size=[self.config.cube_half_size] * 3,
                            rgba=color, friction=[1, 0.005, 0.0001], density=500)
            for name, color in zip(OBJECTS, colors)
        }
        for (name, xy), color in zip(GOALS.items(), colors):
            arena.worldbody.append(new_site(name=name, type='box',
                pos=(*xy, self.config.table_height + 0.001), size=(0.055, 0.055, 0.001),
                rgba=(*color[:3], 0.25)))
        self.model = ManipulationTask(arena, [r.robot_model for r in self.robots], list(self.objects.values()))

    def _setup_references(self):
        super()._setup_references()
        self.body_ids = {name: self.sim.model.body_name2id(obj.root_body) for name, obj in self.objects.items()}

    def _reset_internal(self):
        super()._reset_internal()
        for name, y in zip(OBJECTS, (-0.12, 0.12)):
            xy = np.array([-0.07, y]) + self.rng.uniform(-self.config.spawn_jitter, self.config.spawn_jitter, 2)
            position = [*xy, self.config.table_height + self.config.cube_half_size + 0.002]
            self.sim.data.set_joint_qpos(self.objects[name].joints[0], [*position, 1, 0, 0, 0])

    def reward(self, action=None):
        return 0.0  # Completion is evaluated by the backend after release and settling.

    def _check_success(self):
        return False
