"""Spawned gripper sessions over an in-memory Modbus transport; no USB/robot."""
import ctypes
from dataclasses import asdict, dataclass, replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
from unittest.mock import Mock, patch

import pytest

from hardware.config import GripperConfig
from hardware.gripper_contact import ContactSettings, EmptyBaseline, identity
from hardware.gripper_process import ProcessGripper
from hardware.robotiq import GripperError, Robotiq2F85
from test_robotiq_session import SimulatedSerial


class JournalSerial:
    def __init__(self, root, *, contact=False, motion_reads=4):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root/'connections').open('a') as file:
            file.write(str(os.getpid())+'\n')
        if contact:
            from test_gripper_contact import FakeSerial
            self.device = FakeSerial(current_fn=lambda position: 100 if position >= 70 else 20)
        else:
            self.device = SimulatedSerial(motion_reads=motion_reads)

    def __getattr__(self, name):
        return getattr(self.device, name)

    def write(self, packet):
        if packet[1] == 16:
            with (self.root/'commands').open('a') as file:
                file.write(json.dumps(list(packet[7:13]))+'\n')
        return self.device.write(packet)

    def read(self, count):
        if (self.root/'fail').exists():
            self.device.rx = b''
            raise OSError('simulated cable failure')
        if (self.root/'delay').exists():
            (self.root/'delay').unlink()
            time.sleep(.7)
        return self.device.read(count)

    def close(self):
        self.device.close()
        (self.root/'closed').touch()


@dataclass
class SerialFactory:
    root: str
    contact: bool = False
    motion_reads: int = 4

    def __call__(self, **kwargs):
        assert kwargs['port'] == '/dev/SIMULATED-GRIPPER-ONLY'
        return JournalSerial(self.root, contact=self.contact, motion_reads=self.motion_reads)


@pytest.fixture
def config():
    return GripperConfig(port='/dev/SIMULATED-GRIPPER-ONLY', keepalive_interval_s=.03,
        state_timeout_s=.5, poll_interval_s=.01, io_timeout_s=.05,
        startup_timeout_s=.1, motion_timeout_s=3, speed_raw=32, force_raw=0)


def until(predicate, timeout=4.):
    deadline = time.monotonic()+timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Timed out waiting for simulated worker')
        time.sleep(.01)


def gil_block(seconds=.75):
    # PyDLL intentionally holds Python's GIL while the native function waits.
    library = ctypes.PyDLL(None)
    if not hasattr(library, 'usleep'):
        pytest.skip('Native GIL-holding wait requires libc usleep')
    library.usleep.argtypes = [ctypes.c_uint]
    library.usleep.restype = ctypes.c_int
    assert library.usleep(int(seconds*1_000_000)) == 0


def test_construction_and_readonly_commands_are_passive(config, tmp_path):
    driver = ProcessGripper(config, serial_factory=SerialFactory(str(tmp_path)))
    assert driver._manager is None and not driver.session_info()['active']
    for command in (driver.open, driver.close, driver.activate, driver.stop):
        with pytest.raises(GripperError, match='execution_enabled'):
            command()
    driver.disconnect()
    assert not (tmp_path/'connections').exists()


def test_worker_keeps_polling_during_parent_gil_block_that_breaks_thread_driver(config, tmp_path):
    original = Robotiq2F85(config, serial_factory=SerialFactory(str(tmp_path/'thread')))
    worker = ProcessGripper(config, serial_factory=SerialFactory(str(tmp_path/'process')))
    try:
        original.start_session()
        gil_block()
        original.status()
        with pytest.raises(GripperError, match='Status update gap exceeded'):
            original.assert_healthy()
        fault = original.session_info()['first_fault']
        assert fault['status_gap_s'] > config.state_timeout_s
        assert fault['status_transaction_s'] < config.state_timeout_s
        assert fault['status']['ready']  # A normal later reply cannot erase the failure.
        worker.start_session()
        before = worker.session_info()
        gil_block()
        assert worker.assert_healthy(require_ready=True).ready
        after = worker.session_info()
        assert after['worker_pid'] != os.getpid()
        assert after['successful_status_reads'] > before['successful_status_reads']+5
        assert after['max_status_gap_s'] < config.state_timeout_s
        assert not (tmp_path/'process'/'commands').exists()
    finally:
        original.disconnect()
        worker.disconnect()


def test_device_failure_is_latched_and_never_reconnects_or_retries(config, tmp_path):
    driver = ProcessGripper(config, execution_enabled=True, serial_factory=SerialFactory(str(tmp_path)))
    try:
        driver.start_session()
        pid = driver.session_info()['worker_pid']
        (tmp_path/'fail').touch()
        until(lambda: driver.session_info().get('error'))
        (tmp_path/'fail').unlink()
        assert driver.status().ready
        with pytest.raises(GripperError, match='polling failed'):
            driver.open()
        info = driver.session_info()
        assert info['worker_pid'] == pid and info['first_fault']['message']
        assert len((tmp_path/'connections').read_text().splitlines()) == 1
        assert not (tmp_path/'commands').exists()
        driver.disconnect()
        assert driver.start_session().ready
        assert driver.session_info()['worker_pid'] != pid
        assert len((tmp_path/'connections').read_text().splitlines()) == 2
        assert not (tmp_path/'commands').exists()
    finally:
        driver.disconnect()


def test_worker_timing_fault_keeps_original_half_second_limit(config, tmp_path):
    driver = ProcessGripper(config, execution_enabled=True, serial_factory=SerialFactory(str(tmp_path)))
    try:
        driver.start_session()
        (tmp_path/'delay').touch()
        until(lambda: driver.session_info().get('error'))
        with pytest.raises(GripperError, match='Status update gap exceeded'):
            driver.close()
        fault = driver.session_info()['first_fault']
        assert fault['state_timeout_s'] == .5 and fault['status_gap_s'] >= .7
        assert not (tmp_path/'commands').exists()
    finally:
        driver.disconnect()


def test_worker_exit_blocks_commands_until_explicit_reconnect(config, tmp_path):
    driver = ProcessGripper(config, execution_enabled=True, serial_factory=SerialFactory(str(tmp_path)))
    try:
        driver.start_session()
        process = driver._manager._process
        process.terminate()
        process.join(timeout=2)
        with pytest.raises(GripperError, match='worker exited'):
            driver.open()
        assert driver._manager._process.pid == process.pid
        assert not driver.session_info()['active'] and not (tmp_path/'commands').exists()
        driver.disconnect()
        assert driver.start_session().ready
        assert driver.session_info()['worker_pid'] != process.pid
    finally:
        driver.disconnect()


def test_rpc_timeout_does_not_resubmit_a_command(config, tmp_path):
    driver = ProcessGripper(config, serial_factory=SerialFactory(str(tmp_path)), rpc_timeout_s=.15)
    try:
        driver.start_session()
        (tmp_path/'delay').touch()
        with pytest.raises(GripperError, match='outcome unknown'):
            driver.status()
        with pytest.raises(GripperError, match='no action retried'):
            driver.assert_healthy()
        assert not (tmp_path/'commands').exists()
    finally:
        driver.disconnect()


def test_stop_interrupts_worker_close_from_a_separate_connection(config, tmp_path):
    driver = ProcessGripper(config, execution_enabled=True,
        serial_factory=SerialFactory(str(tmp_path), motion_reads=1000))
    failures = []
    def close():
        try:
            driver.close()
        except GripperError as error:
            failures.append(str(error))
    thread = threading.Thread(target=close)
    try:
        driver.start_session()
        thread.start()
        until(lambda: (tmp_path/'commands').exists())
        assert not driver.stop().go_to
        thread.join(timeout=3)
        assert not thread.is_alive() and failures
        commands = [json.loads(line) for line in (tmp_path/'commands').read_text().splitlines()]
        assert sum(command[0] == 9 for command in commands) == 1
    finally:
        if thread.is_alive():
            driver.stop()
            thread.join(timeout=3)
        driver.disconnect()


def contact_config(config, tmp_path):
    settings = ContactSettings(100, extra_close_raw=2)
    baseline = EmptyBaseline(identity(config, settings), 3, list(range(8,225,8)), [20]*28, 'offline-empty')
    (tmp_path/'baseline.json').write_text(json.dumps(asdict(baseline)))
    (tmp_path/'profile.json').write_text(json.dumps(dict(schema_version=1, baseline_file='baseline.json',
        output_dir='contact_logs', settings=asdict(settings), poll_interval_s=.01, keepalive_interval_s=.05)))
    return replace(config, contact_close_file=str(tmp_path/'profile.json'))


def test_contact_policy_and_confirmed_current_stop_run_inside_worker(config, tmp_path):
    config = contact_config(config, tmp_path)
    driver = ProcessGripper(config, execution_enabled=True, serial_factory=SerialFactory(str(tmp_path), contact=True))
    try:
        assert driver.config.keepalive_interval_s == .05
        state = driver.close()
        assert state.ready and not state.go_to and driver.close_completed(state)
        assert driver.last_motion_feedback['outcome'] == 'current_limit'
        assert driver.last_motion_feedback['completed'] and not driver.last_motion_feedback['grasp_verified']
        assert driver.last_contact_close['settings']['extra_close_raw'] == 2
        assert driver.may_be_holding_object()
        assert driver.session_info()['driver_type'] == 'AgentContactGripper'
        driver.open()
        assert not driver.may_be_holding_object()
        commands = [json.loads(line) for line in (tmp_path/'commands').read_text().splitlines()]
        assert [command[3] for command in commands if command[0] == 9] == [225,0]
    finally:
        driver.disconnect()


def test_original_agent_callbacks_and_lift_use_the_process_contact_driver(config, tmp_path):
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    from core.context import RuntimeContext
    from hardware.agent_geometry import execute_grasp
    from hardware.config import HardwareConfig
    from hardware.interfaces import Pose6D
    from runtime.config import SimpleAgentSystemConfig
    config = contact_config(config, tmp_path)
    driver = ProcessGripper(config, execution_enabled=True, serial_factory=SerialFactory(str(tmp_path), contact=True))
    arm = Mock()
    arm.motion_guard = None
    arm.get_pose.return_value = Pose6D(.3,0,.35,3.14,0,-1.57)
    def move(pose):
        if arm.move_pose.call_count == 1:
            gil_block()
        arm.motion_guard()
        return {'success': True}
    arm.move_pose.side_effect = move
    context = RuntimeContext(arm=arm, gripper=driver)
    settings = SimpleAgentSystemConfig(goal='offline worker grasp', execute_robot=True,
        hardware_config_path='simulated-config-not-a-file')
    try:
        with patch('hardware.config.load_config', return_value=HardwareConfig(gripper=config)):
            robot = SimpleExecutionRuntime(settings)._build_robot(context)
        result = execute_grasp(context, dict(approach_pose_mmrad=[300,0,430,3.14,0,-1.57],
            best_grasp_xarm_mmrad=[300,0,350,3.14,0,-1.57]), robot.gripper_open, robot.gripper_close)
        assert result['success'] and result['gripper_close_completed']
        assert arm.move_pose.call_count == 3
        assert arm.move_pose.call_args.args[0].z == pytest.approx(.4)
        assert driver.last_motion_feedback['outcome'] == 'current_limit'
        assert driver.session_info()['max_status_gap_s'] < .5
    finally:
        driver.disconnect()


def owner_exits(config, factory, path):
    driver = ProcessGripper(config, serial_factory=factory)
    driver.start_session()
    Path(path).write_text(str(driver.session_info()['worker_pid']))
    os._exit(0)  # No manager finalizer: worker must detect loss of its owner.


def test_worker_closes_simulated_serial_if_agent_process_exits(config, tmp_path):
    pid_file = tmp_path/'worker_pid'
    owner = mp.get_context('spawn').Process(target=owner_exits,
        args=(config, SerialFactory(str(tmp_path)), str(pid_file)))
    owner.start()
    try:
        owner.join(timeout=5)
        assert owner.exitcode == 0
        until(lambda: (tmp_path/'closed').exists())
        assert not (tmp_path/'commands').exists()
    finally:
        if owner.is_alive():
            owner.terminate()
            owner.join(timeout=2)
        if pid_file.exists() and not (tmp_path/'closed').exists():
            import signal
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass
