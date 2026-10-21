"""Opt-in integration checks: needs the scene environment and official assets.

Run with .venv-robocasa/bin/python -m pytest -q tests/test_robocasa_occlusion.py.
Not included in the device/model-free core test suite.
"""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("robocasa")

from robocasa_occlusion.assets import generate_assets
from robocasa_occlusion.cli import make_env, camera_capture, label_visibility
from robocasa_occlusion.cameras import LEFT_CAMERA


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    import robocasa
    if not (Path(robocasa.__file__).parent / "models/assets/generative_textures").is_dir():
        pytest.skip("Official kitchen assets not installed")
    assets = generate_assets(tmp_path_factory.mktemp("occlusion_assets"))
    env = make_env(Namespace(scenario="clear", seed=0, layout=11, style=14), assets)
    env.reset()
    env.settle()
    yield env
    env.close()


def test_observations_do_not_expose_object_ground_truth(scene):
    obs, _, _, _ = scene.step(np.zeros(scene.action_spec[0].shape))
    assert not any(k.startswith(("carton_", "blocker_", "object-state")) for k in obs)
    assert any("joint_pos" in k for k in obs)


def test_physical_occlusion_and_reset_repeatability(scene):
    origin = scene.sim.get_state()
    seen = {}
    for case in ["clear", "label_occluded", "fully_occluded", "clear"]:
        scene.sim.set_state(origin)
        scene.set_scenario(case)
        rgb, depth, seg, k, pose = camera_capture(scene, "inspection", 640, 480)
        metric = label_visibility(scene, seg, k, pose)
        assert np.isfinite(depth).all() and (depth > 0).all()
        for obj in scene.objects.values():
            velocity = scene.sim.data.get_joint_qvel(obj.joints[0])
            assert np.linalg.norm(velocity[:3]) < .01
            assert np.linalg.norm(velocity[3:]) < .1
        if case == "clear":
            assert metric["identity_rectangle_visible_fraction"] > .95
            if "clear_rgb" in seen:
                np.testing.assert_array_equal(rgb, seen["clear_rgb"])
                np.testing.assert_allclose(depth, seen["clear_depth"], atol=1e-6, rtol=0)
            seen["clear_rgb"] = rgb
            seen["clear_depth"] = depth
            # All three cartons must be visible in the clear condition, including distractors.
            for name in ("carton_0", "carton_1", "carton_2"):
                label_id = scene.sim.model.geom_name2id(f"{name}_label")
                assert np.count_nonzero(seg[:, :, 1] == label_id) > 100
        elif case == "label_occluded":
            assert metric["identity_rectangle_visible_fraction"] < .02
            assert metric["target_visible_pixels"] > 100
        else:
            assert metric["target_visible_pixels"] == 0


def test_left_camera_mount_position_viewing_axis_and_fixed_world_pose(scene):
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix

    scene.sim.forward()
    camera_id = scene.sim.model.camera_name2id(LEFT_CAMERA)
    pose = get_camera_extrinsic_matrix(scene.sim, LEFT_CAMERA)
    # The physical arm mount is at ~0.7 m; the MJCF robot root is parked at (10,10,0).
    # This catches confusing a robot XML root with the user's base coordinate frame.
    relative = np.linalg.inv(scene.arm_base_pose()) @ pose
    np.testing.assert_allclose(relative[:3, 3], [-.25, .45, .89], atol=1e-4)
    assert scene.sim.model.cam_bodyid[camera_id] == 0  # world-fixed external camera
    target = scene.world([0, -.015, .025])
    direction = target - pose[:3, 3]
    np.testing.assert_allclose(pose[:3, 2], direction / np.linalg.norm(direction), atol=1e-8)
    rgb, depth, _, k, captured_pose = camera_capture(scene, LEFT_CAMERA, 640, 480)
    assert rgb.shape == (480, 640, 3) and depth.shape == (480, 640)
    assert np.isfinite(depth).all() and (depth > 0).all() and rgb.std() > 10
    projected = k @ (np.linalg.inv(captured_pose) @ np.r_[target, 1])[:3]
    np.testing.assert_allclose(projected[:2] / projected[2], [320, 240], atol=1e-6)
    scene.set_scenario("label_occluded")
    np.testing.assert_allclose(get_camera_extrinsic_matrix(scene.sim, LEFT_CAMERA), pose, atol=1e-10)


def test_scene_collision_widths_fit_configured_panda_opening_with_margin(scene):
    model = scene.sim.model
    finger1 = model.joint_name2id("gripper0_right_finger_joint1")
    finger2 = model.joint_name2id("gripper0_right_finger_joint2")
    nominal_opening = float(model.jnt_range[finger1, 1] - model.jnt_range[finger2, 0])
    for name in scene.objects:
        geom = model.geom_name2id(f"{name}_collision")
        dimensions = 2 * model.geom_size[geom]
        expected = scene.carton_size if name.startswith("carton_") else scene.blocker_sizes[name.removeprefix("blocker_")]
        np.testing.assert_allclose(dimensions, expected)
        assert min(dimensions[:2]) + .02 <= nominal_opening + 1e-9
