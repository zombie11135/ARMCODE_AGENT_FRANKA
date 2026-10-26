"""Parallel-jaw equivalent poses using synthetic calibration and fake devices."""
from copy import deepcopy
import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from core.context import RuntimeContext
from hardware.agent_geometry import grasp_plan, validate_pose_selection, GraspSelectionError, execute_grasp
from hardware.interfaces import Pose6D
from hardware.transforms import CalibratedTransforms, pose_matrix


@pytest.fixture
def scene(tmp_path):
    data = dict(schema_version=1, status='geometry_preview_ready', translation_unit='m',
                matrix_convention='p_parent = T_parent_child @ p_child',
                cameras={'wrist': dict(serial='fixture', mounting='eye_in_hand', image_stream='color',
                    depth_aligned_to='color', extrinsic=dict(parent_frame='fr3_link8',
                    child_frame='realsense_fixture_color_optical', matrix=np.eye(4).tolist()))},
                tool=dict(flange_frame='fr3_link8', T_flange_tcp=pose_matrix(
                    Pose6D(0., 0., .15, 0., 0., 0.)).tolist()),
                grasp_model=dict(T_grasp_tcp=[[0.,0.,1.,0.],[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]]))
    config = tmp_path/'calibration.yaml'
    config.write_text(yaml.safe_dump(data))
    transforms = CalibratedTransforms(config)
    capture = Pose6D(.4, 0., .6, math.pi, 0., -1.57)
    source = dict(capture_geometry=dict(before=capture.values, after=capture.values, serial='fixture',
        optical_frame='realsense_fixture_color_optical', calibration=transforms.summary()),
        color_path='unused.png', depth_intrinsics={'fx':100.,'fy':100.,'cx':40.,'cy':30.})
    workcell = dict(tool_max_width_m=.11, tool_length_m=.17, tool_padding_m=.005,
                    table_z_m=0., table_clearance_m=.005, grasp_table_clearance_check=True)
    def candidate(index=0, yaw=1.57, score=.9, tilt=0., pitch=0., z=.35):
        flange = pose_matrix(Pose6D(.3, .02, z, math.pi+tilt, pitch, yaw))
        matrix = (np.linalg.inv(transforms.base_T_camera('wrist', capture)) @ flange @
                  transforms.flange_T_tcp @ np.linalg.inv(transforms.grasp_T_tcp))
        return dict(index=index, score=score, width=.04, height=.02, depth=.02,
                    rotation_matrix=matrix[:3,:3].tolist(), translation=matrix[:3,3].tolist())
    def plan(candidates, **options):
        return grasp_plan(source, dict(candidate_grasps_camera=candidates), transforms, workcell, **options)
    return SimpleNamespace(transforms=transforms, source=source, capture=capture, workcell=workcell,
                           candidate=candidate, plan=plan, root=tmp_path)


def test_half_turn_chooses_negative_yaw_without_changing_contact_geometry(scene):
    candidate = scene.candidate()
    original = deepcopy(candidate)
    plan = scene.plan([candidate])
    assert candidate == original
    assert plan['selected']['jaw_symmetry_flipped']
    assert abs(plan['best_grasp_base_mrad'][5]+1.57) < .002
    np.testing.assert_allclose(plan['best_grasp_base_mrad'][:3], [.3,.02,.35], atol=1e-10)
    old, new = np.asarray(candidate['rotation_matrix']), np.asarray(plan['selected']['rotation_matrix'])
    np.testing.assert_allclose(new[:,0], old[:,0], atol=1e-10)
    np.testing.assert_allclose(new[:,1], -old[:,1], atol=1e-10)
    np.testing.assert_allclose(plan['selected']['translation_m'], candidate['translation'])
    np.testing.assert_allclose(np.asarray(plan['selected']['T_input_grasp'])[:3,:3], new)


@pytest.mark.parametrize('yaw', [-1.57, -.7])
def test_already_preferred_side_is_preserved_without_forcing_exact_yaw(scene, yaw):
    plan = scene.plan([scene.candidate(yaw=yaw)])
    assert not plan['selected']['jaw_symmetry_flipped']
    assert plan['best_grasp_base_mrad'][5] == pytest.approx(yaw)


def test_tilted_grasp_keeps_approach_and_recomputes_flange_with_lateral_tcp_offset(scene):
    scene.transforms.flange_T_tcp[0,3] = .012
    scene.source['capture_geometry']['calibration'] = scene.transforms.summary()
    candidate = scene.candidate(tilt=.13, pitch=-.17)
    plan = scene.plan([candidate])
    old = scene.transforms.base_T_camera('wrist', scene.capture) @ np.asarray(
        [[*candidate['rotation_matrix'][i], candidate['translation'][i]] for i in range(3)] + [[0,0,0,1]])
    new = np.asarray(plan['geometry']['base_T_grasp'])
    np.testing.assert_allclose(new[:3,3], old[:3,3], atol=1e-10)
    np.testing.assert_allclose(new[:3,0], old[:3,0], atol=1e-10)
    np.testing.assert_allclose(new[:3,1], -old[:3,1], atol=1e-10)
    flange = pose_matrix(Pose6D(*plan['best_grasp_base_mrad']))
    np.testing.assert_allclose(flange @ scene.transforms.flange_T_tcp,
                               new @ scene.transforms.grasp_T_tcp, atol=1e-10)


def test_disabling_pose_fix_preserves_original_candidate(scene):
    candidate = scene.candidate()
    plan = scene.plan([candidate], enable_pose_fix=False, lam_rz=0.)
    assert plan['best_grasp_base_mrad'][5] == pytest.approx(1.57)
    np.testing.assert_allclose(plan['selected']['rotation_matrix'], candidate['rotation_matrix'])


def test_yaw_preference_ranks_different_valid_grasps(scene):
    plan = scene.plan([scene.candidate(0, yaw=-.4, score=.9), scene.candidate(1, yaw=-1.57, score=.85)])
    assert plan['best_grasp_camera']['index'] == 1
    assert plan['pose_selection']['options']['target_rz'] == -1.57


def test_pose_and_table_constraints_apply_to_both_equivalent_poses(scene):
    with pytest.raises(GraspSelectionError) as caught:
        scene.plan([scene.candidate(tilt=math.radians(80))])
    assert not any(row['accepted'] for row in caught.value.report['candidates'][0]['pose_variants'])
    with pytest.raises(GraspSelectionError):
        scene.plan([scene.candidate(z=.05)])


def test_explicit_positive_target_selects_the_other_equivalent_pose(scene):
    plan = scene.plan([scene.candidate(yaw=-1.57)], target_rz=1.57)
    assert plan['selected']['jaw_symmetry_flipped']
    assert abs(plan['best_grasp_base_mrad'][5]-1.57) < .002


@pytest.mark.parametrize('options', [{'target_rz':float('nan')}, {'lam_rz':-.1},
                                     {'lam_rz':float('inf')}, {'enable_pose_fix':'false'}])
def test_invalid_yaw_options_fail_before_inference(scene, options):
    from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill
    service = Mock()
    result = PlanGraspWithAnyGraspSkill(RuntimeContext(anygrasp=service, transforms=scene.transforms)).run(
        grasp_input=scene.source, **options)
    assert not result.success
    service.plan_grasp.assert_not_called()
    with pytest.raises(ValueError):
        validate_pose_selection(**options)


def test_saved_preview_and_mock_execution_use_the_selected_equivalent_pose(scene):
    from services.gsnet_service import GSNetService
    from hardware.gsnet_o3d import load_scene
    plan = scene.plan([scene.candidate()])
    root = scene.root/'run'
    (root/'capture').mkdir(parents=True)
    (root/'anygrasp').mkdir()
    (root/'capture/capture.json').write_text(json.dumps(dict(
        device={'serial_number':'fixture'},frame='realsense_fixture_color_optical')))
    (root/'capture_pose.json').write_text(json.dumps(dict(
        before={'pose':scene.capture.values},calibration=scene.transforms.summary())))
    plan.update(run_dir=str(root), result_json_path=str(root/'result.json'), workcell=scene.workcell)
    service = GSNetService.__new__(GSNetService)
    service.backend, service.transforms = 'anygrasp', scene.transforms
    service.save_geometry(plan)
    saved = np.load(root/'anygrasp/grasps.npy')
    np.testing.assert_allclose(saved[0,4:13].reshape(3,3),plan['selected']['rotation_matrix'])
    preview = load_scene(root)
    np.testing.assert_allclose(preview['base_grasp'],plan['geometry']['base_T_grasp'])
    arm, gripper = Mock(), Mock()
    gripper.assert_healthy = Mock(return_value=SimpleNamespace(object_detected_closing=True))
    arm.get_pose.return_value = Pose6D(*plan['best_grasp_base_mrad'])
    context = RuntimeContext(arm=arm, gripper=gripper)
    assert execute_grasp(context,plan,lambda:True,lambda:True)['success']
    assert arm.move_pose.call_args_list[1].args[0].values == pytest.approx(plan['best_grasp_base_mrad'])


@pytest.mark.parametrize('disable', [False, True])
def test_cli_yaw_preferences_reach_the_original_agent_robot_config(monkeypatch, disable):
    import sys
    import hardware.runtime
    from app import cli
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    def profile(args):
        args.simple_actions = True
        args.hardware_config = None
    monkeypatch.setattr(hardware.runtime, 'apply_cli_profile', profile)
    captured = []
    def run(args, goal, executor_args):
        config = cli.build_config(args, goal, executor_args)
        captured.append(SimpleExecutionRuntime(config)._build_robot(RuntimeContext()).config.grasp_pose_options)
        return True
    monkeypatch.setattr(cli,'run_goal',run)
    argv=['agent','--goal','offline','--grasp-target-rz','-1.5','--grasp-lam-rz','.4']
    if disable:
        argv.append('--no-grasp-pose-fix')
    monkeypatch.setattr(sys,'argv',argv)
    cli.main()
    assert captured[0]['target_rz'] == -1.5
    assert captured[0]['lam_rz'] == .4
    assert captured[0]['enable_pose_fix'] is not disable
