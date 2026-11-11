from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from core.context import RuntimeContext
from hardware.config import ArmConfig
from hardware.fr3 import MotionError
from hardware.gentle_place import execute_place, prepare_place_plan
from hardware.robotiq import GripperError
from hardware.stack_surface import estimate_support


def rig():
    arm = SimpleNamespace(config=ArmConfig(speed_scaling=.2, gentle_place_enabled=True),
                          plan_pose=Mock(), move_pose=Mock(return_value={}))
    def speed(value):
        arm.config = replace(arm.config, speed_scaling=value)
    arm.set_speed_scaling = Mock(side_effect=speed)
    ctx = RuntimeContext(arm=arm, gripper=SimpleNamespace(assert_healthy=Mock()))
    plan = dict(place_pose_mmrad=[500, 0, 300, 3.14, 0, 0],
                approach_pose_mmrad=[500, 0, 420, 3.14, 0, 0],
                lift_pose_mmrad=[500, 0, 420, 3.14, 0, 0])
    return ctx, plan


def test_measured_bottom_height_and_transit_are_not_tcp_confused():
    ctx, plan = rig()
    cfg = replace(ctx.arm.config, place_support_z_m=.04, place_object_bottom_offset_m=.19)
    adjusted = prepare_place_plan(plan, cfg)
    assert adjusted['place_pose_mmrad'][2] == pytest.approx(240)
    assert adjusted['pre_release_pose_mmrad'][2] == pytest.approx(265)
    assert adjusted['approach_pose_mmrad'][2] == 420
    assert plan['place_pose_mmrad'][2] == 300  # Cached original remains immutable.


def test_stack_height_requires_geometry_and_uses_fresh_estimate():
    ctx, plan = rig()
    cfg = replace(ctx.arm.config, place_height_mode='stack_rgbd', place_object_bottom_offset_m=.18)
    with pytest.raises(ValueError, match='fresh RGB-D'):
        prepare_place_plan(plan, cfg)
    plan['stack_support'] = dict(support_z_m=.08)
    assert prepare_place_plan(plan, cfg)['place_pose_mmrad'][2] == pytest.approx(270)
    plan['stack_support']['support_z_m'] = .12
    assert prepare_place_plan(plan, cfg)['place_pose_mmrad'][2] == pytest.approx(310)


@pytest.mark.parametrize('failed_plan', range(4))
def test_all_endpoints_prechecked_before_any_actuation(failed_plan):
    ctx, plan = rig()
    ctx.arm.plan_pose.side_effect = [None]*failed_plan + [MotionError({'error': 'collision'})]
    opened = Mock()
    with pytest.raises(MotionError):
        execute_place(ctx, prepare_place_plan(plan, ctx.arm.config), opened)
    opened.assert_not_called()
    ctx.arm.move_pose.assert_not_called()


@pytest.mark.parametrize('failure', ['approach', 'pre_release', 'release', 'open', 'retreat', None])
def test_phase_order_stop_and_restore_speed(monkeypatch, failure):
    monkeypatch.setattr('hardware.gentle_place.time.sleep', lambda _: None)
    ctx, plan = rig()
    events, speeds = [], []
    def move(pose):
        label = ['approach', 'pre_release', 'release', 'retreat'][len(speeds)]
        events.append(label);speeds.append(ctx.arm.config.speed_scaling)
        if label == failure:
            raise MotionError({'error': 'controller failed'})
        return {}
    def opened():
        events.append('open')
        return failure != 'open'
    ctx.arm.move_pose.side_effect = move
    if failure:
        with pytest.raises((MotionError, GripperError)):
            execute_place(ctx, prepare_place_plan(plan, ctx.arm.config), opened)
    else:
        assert execute_place(ctx, prepare_place_plan(plan, ctx.arm.config), opened)['released']
    sequence = ['approach', 'pre_release', 'release', 'open', 'retreat']
    assert events == (sequence if not failure else sequence[:sequence.index(failure)+1])
    assert speeds == [.2, .03, .03, .05][:len(speeds)]
    assert ctx.arm.config.speed_scaling == .2
    assert ctx.manipulation_progress['released'] is (failure in ('retreat', None))


def test_placement_never_increases_user_speed(monkeypatch):
    monkeypatch.setattr('hardware.gentle_place.time.sleep', lambda _: None)
    ctx, plan = rig()
    ctx.arm.config = replace(ctx.arm.config, speed_scaling=.01)
    execute_place(ctx, prepare_place_plan(plan, ctx.arm.config), Mock(return_value=True))
    assert all(call.args[0] == .01 for call in ctx.arm.set_speed_scaling.call_args_list)


def surface(depth):
    return estimate_support(depth, dict(fx=500, fy=500, cx=100, cy=100),
                            np.eye(4), [0, 0, 200, 200], [0, 0])


def test_support_surface_height_and_rejects_stacked_edges_or_missing_depth():
    depth = np.full((200, 200), .5)
    assert surface(depth)['support_z_m'] == pytest.approx(.5)
    depth[:, 100:] = .55
    with pytest.raises(ValueError, match='not flat'):
        surface(depth)
    depth[:] = .5
    depth[:, 85:] = 0
    with pytest.raises(ValueError, match='coverage'):
        surface(depth)


def test_adjusted_plan_remains_bound_to_task_and_rejects_tampering(monkeypatch):
    from skill_lib.actions.task_containers import remember_container
    from skill_lib.simple_actions import SimpleRobotActions, SimpleRobotConfig
    monkeypatch.setattr('hardware.gentle_place.time.sleep', lambda _: None)
    ctx, plan = rig()
    robot = SimpleRobotActions(ctx, SimpleRobotConfig(execute=True), gripper_open=lambda: True)
    remember_container(robot, 'blue tray', plan)
    adjusted = prepare_place_plan(plan, ctx.arm.config)
    assert robot._is_task_container_plan(adjusted)
    assert robot._execute_cached_place(adjusted)['success']
    adjusted['place_pose_mmrad'][2] -= 20
    assert not robot._is_task_container_plan(adjusted)


def test_tool_cannot_descend_into_stack_even_if_object_offset_is_wrong(tmp_path):
    import json
    ctx, plan = rig()
    (tmp_path/'config').mkdir()
    (tmp_path/'config/workcell.json').write_text(json.dumps(dict(tool_length_m=.17,
        tool_padding_m=.005, tool_max_width_m=.10)))
    ctx.arm.config = replace(ctx.arm.config, workspace=str(tmp_path), place_height_mode='stack_rgbd',
                             place_object_bottom_offset_m=.12)
    plan['stack_support'] = dict(support_z_m=.08)
    opened = Mock()
    with pytest.raises(MotionError, match='Tool envelope'):
        execute_place(ctx, prepare_place_plan(plan, ctx.arm.config), opened)
    opened.assert_not_called();ctx.arm.move_pose.assert_not_called()
