"""Session home command/task consistency; all FK and motion calls are fakes."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.context import RuntimeContext
from hardware.config import ArmConfig, HardwareConfig
from hardware.interfaces import Pose6D
from runtime.config import SimpleAgentSystemConfig
from runtime.home_pose import HomePoseSession


HOME = [.4, .02, .57, -3.14, .01, -1.57]
OTHER = [.6, -.2, .3, 3.14, 0., 1.57]


@pytest.fixture
def builtin(monkeypatch):
    import hardware.runtime as module
    from app.cli import run_builtin_command
    cfg = HardwareConfig()
    monkeypatch.setattr(module, 'load_config', lambda path: cfg)
    factory = Mock()
    factory.return_value.get_pose.return_value = Pose6D(*HOME)
    factory.return_value.move_pose.return_value = dict(ok=True)
    monkeypatch.setattr(module, 'FrankaMoveItArm', factory)
    monkeypatch.setattr(module, 'build_hardware', Mock(side_effect=AssertionError('Home must not construct a gripper/model/transform stack')))
    args = SimpleNamespace(hardware_config='offline-profile.yaml', execute_robot=True, arm_speed_scaling=.12)
    return SimpleNamespace(module=module, args=args, factory=factory, command=run_builtin_command)


@pytest.mark.parametrize('alias', ['home', '/home', '回到初始位置'])
def test_first_unconfigured_home_records_actual_pose_then_returns_to_same_pose(builtin, alias, capsys):
    b = builtin
    assert b.command(alias, b.args)
    arm = b.factory.return_value
    arm.get_pose.assert_called_once_with()
    arm.move_pose.assert_not_called()
    assert '本次无需移动' in capsys.readouterr().out
    arm.get_pose.return_value = Pose6D(*OTHER)
    assert b.command(alias, b.args)
    arm.get_pose.assert_called_once_with()  # Do not redefine home at the displaced pose.
    assert arm.move_pose.call_args.args[0].values == HOME
    assert arm.close.call_count == 2
    assert b.factory.call_args.args[0].speed_scaling == .12
    assert b.factory.call_args.args[0].singularity_screening == 'advisory'


def test_configured_home_moves_without_reading_or_registering_current_pose(builtin, monkeypatch):
    b = builtin
    cfg = HardwareConfig(arm=ArmConfig(home_pose=HOME))
    monkeypatch.setattr(b.module, 'load_config', lambda path: cfg)
    assert b.command('home', b.args)
    arm = b.factory.return_value
    arm.get_pose.assert_not_called()
    assert arm.move_pose.call_args.args[0].values == HOME
    assert b.args.home_pose_session.source == 'hardware_yaml'


@pytest.mark.parametrize('known', [False, True])
def test_dry_run_home_does_not_query_pose_construct_hardware_or_move(builtin, known, capsys):
    b = builtin
    b.args.execute_robot = False
    if known:
        session = HomePoseSession()
        session.resolve(b.args.hardware_config, HOME, lambda: pytest.fail('configured home needs no FK'))
        b.args.home_pose_session = session
    assert b.command('home', b.args)
    b.factory.assert_not_called()
    assert 'dry-run' in capsys.readouterr().out


def test_task_records_home_for_builtin_and_later_tasks_keep_it_after_failure(builtin, monkeypatch):
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    import hardware.config
    b = builtin
    session = HomePoseSession()
    b.args.home_pose_session = session
    monkeypatch.setattr(hardware.config, 'load_config', lambda path: HardwareConfig())
    arm = Mock()
    arm.get_pose.return_value = Pose6D(*HOME)
    config = SimpleAgentSystemConfig(goal='task one', hardware_config_path=b.args.hardware_config,
        home_pose_session=session)
    first = SimpleExecutionRuntime(config)._build_robot(RuntimeContext(arm=arm))
    assert first.config.home_pose_mmrad == pytest.approx([400., 20., 570., *HOME[3:]])
    assert b.command('home', b.args)
    assert b.factory.return_value.move_pose.call_args.args[0].values == HOME
    b.factory.return_value.get_pose.assert_not_called()
    # Starting another goal at a failed grasp pose must not silently redefine home.
    arm.get_pose.return_value = Pose6D(*OTHER)
    second_config = SimpleAgentSystemConfig(goal='task two', hardware_config_path=b.args.hardware_config,
                                           home_pose_session=session)
    second = SimpleExecutionRuntime(second_config)._build_robot(RuntimeContext(arm=arm))
    assert second.config.home_pose_mmrad == first.config.home_pose_mmrad
    arm.get_pose.assert_called_once_with()
    arm.move_pose.assert_not_called()


def test_session_home_does_not_replace_actual_buffer_startup_orientation(builtin, monkeypatch):
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    import hardware.config
    b = builtin
    session = HomePoseSession()
    session.resolve(b.args.hardware_config, HOME, lambda: None)
    monkeypatch.setattr(hardware.config, 'load_config', lambda path: HardwareConfig())
    arm = Mock()
    arm.get_pose.return_value = Pose6D(*OTHER)
    config = SimpleAgentSystemConfig(goal='buffer task', hardware_config_path=b.args.hardware_config,
        home_pose_session=session, geometry_occlusion_buffer_positions_mm=[[450,-420,300]])
    robot = SimpleExecutionRuntime(config)._build_robot(RuntimeContext(arm=arm))
    assert robot.config.home_pose_mmrad == pytest.approx([400.,20.,570.,*HOME[3:]])
    assert robot.config.geometry_occlusion_buffer_poses_mmrad == [[450.,-420.,300.,*OTHER[3:]]]
    arm.get_pose.assert_called_once_with()


@pytest.mark.parametrize('failure', ['read', 'move'])
def test_failed_read_or_motion_is_reported_once_and_arm_closed(builtin, failure, capsys):
    b = builtin
    arm = b.factory.return_value
    if failure == 'read':
        arm.get_pose.side_effect = RuntimeError('FK unavailable')
    else:
        b.args.home_pose_session = HomePoseSession()
        b.args.home_pose_session.resolve(b.args.hardware_config, HOME, lambda: None)
        arm.move_pose.side_effect = RuntimeError('motion failed')
    assert b.command('home', b.args)
    output = capsys.readouterr().out
    assert '[builtin] stopped:' in output and 'home completed' not in output
    assert arm.get_pose.call_count <= 1 and arm.move_pose.call_count <= 1
    arm.close.assert_called_once_with()
    if failure == 'read':
        assert b.args.home_pose_session.known_pose(b.args.hardware_config) is None


def test_home_sessions_are_process_local_and_isolated_by_profile():
    session = HomePoseSession()
    pose, _ = session.resolve('robot-a.yaml', None, lambda: HOME)
    pose[0] = 999  # Returned lists cannot rewrite the recorded home.
    assert session.known_pose('robot-a.yaml') == HOME
    assert session.known_pose('robot-b.yaml') is None
    assert HomePoseSession().known_pose('robot-a.yaml') is None
    target, recorded = session.resolve('robot-b.yaml', None, lambda: OTHER)
    assert recorded and target == OTHER


def test_cli_passes_same_home_session_to_each_task_without_device_access(monkeypatch):
    import sys
    from app import cli
    import hardware.runtime
    monkeypatch.setattr(hardware.runtime, 'apply_cli_profile', lambda args: setattr(args, 'simple_actions', True))
    seen = []
    def run(args, goal, executor_args):
        first = cli.build_config(args, goal, executor_args)
        second = cli.build_config(args, 'another task', executor_args)
        assert first.home_pose_session is second.home_pose_session is args.home_pose_session
        assert first.home_pose_session.pose_mrad is None
        seen.append(first)
        return True
    monkeypatch.setattr(cli, 'run_goal', run)
    monkeypatch.setattr(sys, 'argv', ['app', '--goal', 'offline home test'])
    cli.main()
    assert len(seen) == 1
