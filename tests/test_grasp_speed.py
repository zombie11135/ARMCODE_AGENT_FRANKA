"""Stage speed policy and exception restoration, without robot connections."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from core.context import RuntimeContext
from hardware.agent_geometry import execute_grasp
from hardware.config import ArmConfig, load_config
from hardware.fr3 import FrankaMoveItArm, MotionError
from hardware.interfaces import Pose6D


def rig(speed=.5, fail_at=None):
    arm = FrankaMoveItArm(ArmConfig(speed_scaling=speed, grasp_contact_speed_scaling=.2))
    calls = []
    def request(operation, pose=None):
        calls.append((operation, arm.config.speed_scaling))
        if fail_at == len(calls):
            raise MotionError({'error': 'simulated failure'})
        return {'ok': True}
    arm._request = request
    arm.get_pose = Mock(return_value=Pose6D(.3, 0, .35, 3.14, 0, -1.57))
    grip = SimpleNamespace(assert_healthy=Mock(return_value=SimpleNamespace(object_detected_closing=True)))
    context = RuntimeContext(arm=arm, gripper=grip)
    plan = dict(approach_pose_mmrad=[300, 0, 430, 3.14, 0, -1.57],
                best_grasp_xarm_mmrad=[300, 0, 350, 3.14, 0, -1.57])
    return context, plan, calls


def test_transit_fast_contact_and_initial_lift_capped():
    context, plan, calls = rig()
    assert execute_grasp(context, plan, lambda: True, lambda: True)['success']
    assert calls == [('plan', .5), ('plan', .2), ('move', .5), ('move', .2), ('move', .2)]
    assert context.arm.config.speed_scaling == .5
    assert context.arm.motion_guard is None


@pytest.mark.parametrize('fail_at', [2, 4, 5])
def test_speed_and_health_guard_restored_on_contact_failure(fail_at):
    context, plan, calls = rig(fail_at=fail_at)
    with pytest.raises(MotionError, match='simulated failure'):
        execute_grasp(context, plan, lambda: True, lambda: True)
    assert len(calls) == fail_at
    assert context.arm.config.speed_scaling == .5
    assert context.arm.motion_guard is None


def test_slower_operator_override_is_never_raised():
    context, plan, calls = rig(speed=.05)
    execute_grasp(context, plan, lambda: True, lambda: True)
    assert all(speed == .05 for _, speed in calls)
    assert context.arm.config.speed_scaling == .05


@pytest.mark.parametrize('value', ['0', '-.1', '1.1', '.nan', '.inf', 'true'])
def test_invalid_contact_speed_rejected_before_device_start(tmp_path, value):
    config = tmp_path / 'invalid.yaml'
    config.write_text('arm:\n  grasp_contact_speed_scaling: '+value+'\n')
    with pytest.raises(ValueError):
        load_config(config)
