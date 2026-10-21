"""Agent contact-close adapter, validated on a simulated Modbus device only."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from hardware.config import DEFAULT_CONFIG, GripperConfig, HardwareConfig, load_config
from hardware.gripper_contact import ContactSettings, EmptyBaseline, identity
from hardware.robotiq import GripperCurrentLimitError, GripperError, GripperStatus, Robotiq2F85
from hardware.robotiq_contact import AgentContactGripper, load_contact_close_setup
from hardware.runtime import Hardware, build_hardware, require_task_geometry
from test_gripper_contact import FakeSerial


@pytest.fixture(autouse=True)
def no_hardware():
    with patch('serial.Serial', side_effect=AssertionError('No USB in tests')), \
         patch('subprocess.Popen', side_effect=AssertionError('No robot processes in tests')):
        yield


@pytest.fixture
def setup(tmp_path):
    cfg = GripperConfig(port='/dev/fake-gripper', poll_interval_s=.05, keepalive_interval_s=.1,
                        io_timeout_s=.05, motion_timeout_s=3, speed_raw=32, force_raw=0,
                        contact_close_file=str(tmp_path/'profile.json'))
    options = ContactSettings(100, extra_close_raw=3)
    base = EmptyBaseline(identity(cfg, options), 3, list(range(8, 225, 8)), [20]*28, 'saved-empty.json')
    (tmp_path/'baseline.json').write_text(json.dumps(asdict(base)))
    data = dict(schema_version=1, baseline_file='baseline.json', settings=asdict(options),
                output_dir='logs', poll_interval_s=.01, keepalive_interval_s=.05)
    (tmp_path/'profile.json').write_text(json.dumps(data))
    return cfg, data


@pytest.fixture
def device(setup):
    cfg, _ = setup
    serial = FakeSerial(current_fn=lambda p: 60 if p >= 80 else 20)
    factory = Mock(return_value=serial)
    driver = AgentContactGripper(cfg, execution_enabled=True, serial_factory=factory)
    yield driver, serial, factory
    driver.disconnect()


def test_constructor_is_passive_and_uses_experiment_timing(device):
    driver, serial, factory = device
    factory.assert_not_called()
    assert not serial.packets
    assert driver.contact_setup.settings.extra_close_raw == 3
    assert driver.contact_setup.settings.speed_raw == 0
    assert driver.config.speed_raw == 32  # Existing open speed is independent.
    assert driver.config.poll_interval_s == .01
    assert driver.config.keepalive_interval_s == .05


def test_real_close_method_stops_at_contact_plus_three_and_returns_true_status(device):
    driver, serial, _ = device
    result = driver.close()
    report = driver.last_contact_close
    assert result.ready and not result.go_to
    assert driver.close_completed(result)
    assert driver.may_be_holding_object()
    assert report['target_raw'] == report['contact']['position_raw']+3
    assert not report['grasp_verified']
    moves = [c for c in serial.commands if c[0] == 9]
    assert moves[0] == (9, 225, 0, 0)
    assert all(c[1] <= report['target_raw'] for c in moves[1:])
    assert serial.commands[-1][0] == 1
    assert not any(c[1] == 255 for c in moves)
    assert driver.last_motion_feedback['completed']
    assert driver.last_motion_feedback['extra_close_raw'] == 3
    assert Path(report['log_path']).is_file()
    assert not Path(report['log_path']).with_suffix('.png').exists()  # No plot latency in Agent.


def test_open_invalidates_hold_hint_and_next_close_still_uses_baseline(device):
    driver, serial, _ = device
    closed = driver.close()
    driver.open()
    assert not driver.close_completed(closed)
    assert not driver.may_be_holding_object()
    assert serial.commands[-1] == (9, 0, 32, 0)
    # Simulator jumps straight to open; the real hardware returns near raw=3.
    assert driver.close_completed(driver.close())


def test_current_ceiling_propagates_failure_without_full_close_or_retry(device):
    driver, serial, _ = device
    serial.current_fn = lambda p: 100 if p >= 70 else 20
    with pytest.raises(GripperCurrentLimitError):
        driver.close()
    assert driver.last_contact_close['outcome'] == 'current_limit'
    assert driver.last_contact_close['stop_confirmed']
    assert not driver.close_completed(driver.assert_healthy())
    assert not driver.last_motion_feedback['completed']
    assert serial.commands == [(9, 225, 0, 0), (1, 225, 0, 0)]


def test_no_contact_completion_preserves_existing_agent_semantics(device):
    driver, serial, _ = device
    serial.current_fn = lambda p: 20
    state = driver.close()
    assert driver.close_completed(state)
    assert driver.last_contact_close['outcome'] == 'travel_limit_no_contact'
    assert not driver.may_be_holding_object()
    assert not driver.last_contact_close['grasp_verified']


def test_native_contact_stops_without_forcing_three_more_counts(device):
    driver, serial, _ = device
    serial.native_at = 70
    assert driver.close_completed(driver.close())
    assert driver.last_contact_close['outcome'] == 'native_contact'
    assert serial.commands == [(9, 225, 0, 0), (1, 225, 0, 0)]


def test_readonly_and_speed_override_refused_before_connection(device):
    driver, serial, factory = device
    for kwargs in (dict(speed_raw=32), dict(force_raw=10), dict(speed_raw=False)):
        with pytest.raises(ValueError):
            driver.close(**kwargs)
    driver.execution_enabled = False
    with pytest.raises(GripperError):
        driver.close()
    factory.assert_not_called()
    assert not serial.packets


def test_wrong_starting_opening_is_not_silently_opened_or_recalibrated(device):
    driver, serial, _ = device
    serial.position = 50
    with pytest.raises(ValueError, match='open first'):
        driver.close()
    assert not serial.commands


@pytest.mark.parametrize('change', [dict(extra_close_raw=-1), dict(speed_raw=32), dict(force_raw=1),
                                  dict(current_limit_ma=15), dict(max_close_raw=255)])
def test_bad_settings_or_mismatched_baseline_rejected_offline(setup, change):
    cfg, data = setup
    data['settings'].update(change)
    Path(cfg.contact_close_file).write_text(json.dumps(data))
    with pytest.raises(ValueError):
        AgentContactGripper(cfg)


def test_missing_baseline_never_falls_back_to_old_close(setup):
    cfg, _ = setup
    (Path(cfg.contact_close_file).parent/'baseline.json').unlink()
    with pytest.raises(FileNotFoundError):
        AgentContactGripper(cfg)


def test_profile_paths_resolve_relative_and_constructor_dispatch_is_offline(setup):
    cfg, _ = setup
    resolved = load_contact_close_setup(cfg)
    assert resolved.output_dir == Path(cfg.contact_close_file).parent/'logs'
    settings = replace(load_config(DEFAULT_CONFIG), gripper=cfg)
    with patch('hardware.runtime.load_config', return_value=settings):
        hardware = build_hardware(agent=True)
        assert isinstance(hardware.gripper, AgentContactGripper)
        assert hardware.gripper._serial is None
        hardware.disconnect()
        require_task_geometry(SimpleNamespace(hardware_config_path='mocked.yaml'))
    legacy = replace(settings, gripper=replace(cfg, contact_close_file=None))
    with patch('hardware.runtime.load_config', return_value=legacy):
        hardware = build_hardware(agent=True)
        assert type(hardware.gripper) is Robotiq2F85
        hardware.disconnect()


def test_hardware_close_callback_accepts_deliberately_stopped_driver(device):
    driver, _, _ = device
    hardware = Hardware(HardwareConfig(gripper=driver.config), Mock(), driver, Mock())
    assert hardware.close_gripper()
    assert not driver.assert_healthy().go_to


def test_agent_callback_closes_once_and_allows_lift_after_contact_stop(device):
    from agents.simple_agent_system import SimpleAgentSystemConfig
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    from core.context import RuntimeContext
    from hardware.agent_geometry import execute_grasp
    from hardware.interfaces import Pose6D
    driver, serial, _ = device
    arm = Mock()
    arm.motion_guard = None
    arm.get_pose.return_value = Pose6D(.3, 0, .35, 3.14, 0, -1.57)
    context = RuntimeContext(arm=arm, gripper=driver)
    config = SimpleAgentSystemConfig(goal='offline contact close', execute_robot=True,
                                    hardware_config_path=str(DEFAULT_CONFIG))
    robot = SimpleExecutionRuntime(config)._build_robot(context)
    plan = dict(approach_pose_mmrad=[300, 0, 430, 3.14, 0, -1.57],
                best_grasp_xarm_mmrad=[300, 0, 350, 3.14, 0, -1.57], lift_distance_m=.05)
    with patch.object(driver, 'close', wraps=driver.close) as close:
        result = execute_grasp(context, plan, robot.gripper_open, robot.gripper_close)
        close.assert_called_once()
    assert result['success'] and result['gripper_close_completed']
    assert arm.move_pose.call_count == 3
    assert arm.move_pose.call_args.args[0].z == pytest.approx(.4)
    assert not driver.assert_healthy().go_to
    arm.motion_guard()  # Stopped close is healthy for subsequent transport.


def test_agent_does_not_lift_after_current_cutoff(device):
    from agents.simple_agent_system import SimpleAgentSystemConfig
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    from core.context import RuntimeContext
    from hardware.agent_geometry import execute_grasp
    from hardware.interfaces import Pose6D
    driver, serial, _ = device
    serial.current_fn = lambda p: 100 if p >= 70 else 20
    arm = Mock()
    arm.motion_guard = None
    arm.get_pose.return_value = Pose6D(.3, 0, .35, 3.14, 0, -1.57)
    context = RuntimeContext(arm=arm, gripper=driver)
    config = SimpleAgentSystemConfig(goal='offline cutoff', execute_robot=True,
                                    hardware_config_path=str(DEFAULT_CONFIG))
    robot = SimpleExecutionRuntime(config)._build_robot(context)
    arm.get_pose.reset_mock()
    plan = dict(approach_pose_mmrad=[300, 0, 430, 3.14, 0, -1.57],
                best_grasp_xarm_mmrad=[300, 0, 350, 3.14, 0, -1.57])
    with pytest.raises(GripperCurrentLimitError):
        execute_grasp(context, plan, robot.gripper_open, robot.gripper_close)
    assert arm.move_pose.call_count == 2  # Approach + grasp; no lift.
    arm.get_pose.assert_not_called()


def test_legacy_completion_still_checks_true_hardware_flags():
    driver = Robotiq2F85(GripperConfig())
    normal = GripperStatus(True, True, 3, 3, 0, 255, 240, 0)
    assert driver.close_completed(normal)
    assert not driver.close_completed(replace(normal, go_to=False))
    assert not driver.close_completed(replace(normal, object_state=0))
    assert not driver.close_completed(replace(normal, fault=9))


def test_yaml_resolves_contact_profile_relative_to_itself(tmp_path):
    data = load_config(DEFAULT_CONFIG).to_dict()
    data['gripper']['contact_close_file'] = 'profiles/contact.json'
    import yaml
    path = tmp_path/'hardware.yaml'
    path.write_text(yaml.safe_dump(data))
    assert load_config(path).gripper.contact_close_file == str(tmp_path/'profiles/contact.json')
