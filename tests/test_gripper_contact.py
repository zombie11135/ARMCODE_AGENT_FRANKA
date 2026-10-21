"""No-hardware tests: algorithm boundaries plus complete simulated Modbus trials."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from unittest.mock import Mock, patch

import pytest

from hardware.config import GripperConfig
from hardware.gripper_contact import (ContactExperimentGripper, ContactPolicy, ContactSettings,
                                     EmptyBaseline, export_trial, identity)
from hardware.gripper_contact_cli import console, main
from hardware.robotiq import GripperError, GripperStatus, crc16


@pytest.fixture(autouse=True)
def forbid_real_serial():
    with patch('serial.Serial', side_effect=AssertionError('Tests must not access USB')):
        yield


def state(position=60, current=20, *, obj=0, request=225, goto=True, fault=0):
    return GripperStatus(True, goto, 3, obj, fault, request, position, current)


def baseline(config=None, settings=None):
    settings = settings or ContactSettings(100)
    return EmptyBaseline(identity(config or GripperConfig(), settings), 3,
                         list(range(8, 225, 8)), [20]*28, 'synthetic-empty.json')


def policy(**changes):
    settings = replace(ContactSettings(100, startup_ignore_s=0), **changes)
    return ContactPolicy(settings, baseline(settings=settings))


def test_sustained_contact_anchors_first_sample_and_only_one_target():
    p = policy(extra_close_raw=3)
    p.observe(state(80, 60), .01)
    p.observe(state(81, 60), .02)
    assert p.contact['position_raw'] == 80
    assert p.target_raw == p.pending_target == 83
    p.request_raw, p.pending_target = 83, None  # Acknowledged target update.
    p.observe(state(82, 70, request=83), .03)
    assert p.pending_target is None
    p.observe(state(83, 70, request=83, obj=3), .04)
    assert p.reason == 'contact_travel_limit'


@pytest.mark.parametrize('extra', [0, 1])
def test_confirmation_travel_consumes_extra_allowance(extra):
    p = policy(extra_close_raw=extra)
    p.observe(state(80, 60), .01)
    p.observe(state(82, 60), .02)
    assert p.reason == 'contact_travel_limit'
    assert p.pending_target is None
    assert p.target_raw == 80+extra


def test_single_spike_and_startup_do_not_count_as_contact():
    p = policy(startup_ignore_s=.1)
    for t, current in [(0.01, 60), (.04, 60), (.11, 60), (.13, 20), (.15, 60)]:
        p.observe(state(current=current), t)
    assert p.contact is None
    p.observe(state(current=60), .17)
    assert p.contact['elapsed_s'] == .15


def test_hard_ceiling_preempts_startup_contact_and_stale_echo():
    p = policy(startup_ignore_s=.1)
    p.observe(state(current=100, obj=2, request=0), .01)
    assert p.reason == 'current_limit'
    assert p.contact is None
    p.observe(state(current=0, obj=2), .02)
    assert p.reason == 'current_limit'


def test_native_contact_never_forces_extra_closure():
    p = policy(extra_close_raw=20)
    p.observe(state(current=0, obj=2), .01)
    assert p.reason == 'native_contact'
    assert p.pending_target is None
    assert p.contact['source'] == 'native'


def test_no_contact_at_end_is_not_a_grasp():
    p = policy()
    p.observe(state(225, obj=3), .01)
    assert p.reason == 'travel_limit_no_contact'
    assert p.contact is None


def test_stale_echo_not_accepted_as_contact_or_completion():
    p = policy()
    p.observe(state(current=60, obj=2, request=0), .01)
    assert p.contact is None and p.reason is None
    p.observe(state(request=0), .15)
    p.observe(state(request=0), .22)
    assert p.reason == 'command_echo_timeout'


def test_late_feedback_stops_instead_of_adding_motion():
    p = policy()
    p.observe(state(current=60), .01)
    p.observe(state(current=60), .25)
    assert p.reason == 'feedback_gap'
    assert p.pending_target is None


def test_fault_blocks_contact():
    p = policy()
    p.observe(state(current=60, fault=9), .01)
    assert p.reason == 'not_ready_or_fault'


@pytest.mark.parametrize('changes', [dict(extra_close_raw=-1), dict(extra_close_raw=True),
    dict(current_limit_ma=15), dict(current_limit_ma=0), dict(contact_rise_ma=100),
    dict(confirm_samples=0), dict(speed_raw=256), dict(startup_ignore_s=float('nan'))])
def test_invalid_parameters(changes):
    with pytest.raises(ValueError):
        replace(ContactSettings(100), **changes).validate()


def test_baseline_interpolates_and_checks_device_motion_and_start():
    cfg, options = GripperConfig(port='/dev/mock'), ContactSettings(100)
    b = baseline(cfg, options)
    b.currents_ma[1] = 40
    assert b.at(12) == 30
    assert b.at(0) == 20
    b.check(cfg, options, 3)
    for config, setting, start in ((replace(cfg, port='/dev/other'), options, 3),
                                  (cfg, replace(options, speed_raw=32), 3), (cfg, options, 20)):
        with pytest.raises(ValueError):
            b.check(config, setting, start)


class FakeSerial:
    """One coherent Modbus device shared by foreground and keepalive threads."""
    def __init__(self, current_fn=lambda p: 20, *, native_at=None, missing_ack=False):
        self.current_fn, self.native_at = current_fn, native_at
        self.missing_ack = missing_ack
        self.position, self.request, self.flags = 3, 0, 0x31
        self.rx, self.packets, self.closed = b'', [], False

    def reset_input_buffer(self):
        assert not self.rx, 'interleaved transaction'

    def write(self, packet):
        assert not self.closed and not self.rx
        assert crc16(packet[:-2]) == packet[-2:]
        self.packets.append(packet)
        if packet[1] == 3:
            current = 0
            if self.flags == 0x39:
                self.position += min(2, self.request-self.position)
                current = self.current_fn(self.position)
                if self.native_at is not None and self.position >= self.native_at:
                    self.flags, current = 0xB9, 0
                elif self.position >= self.request:
                    self.flags, current = 0xF9, 0
            payload = bytes([9, 3, 6, self.flags, 0, 0, self.request, self.position, current//10])
        else:
            assert packet[1] == 16
            action = packet[7]
            assert action in (1, 9), 'must not reset/activate during trial'
            if action == 9:
                self.request, self.flags = packet[10], 0x39
                if self.missing_ack:
                    self.missing_ack = False
                    return len(packet)
            else:
                self.flags = 0x31
            payload = bytes.fromhex('09 10 03 e8 00 03')
        self.rx = payload+crc16(payload)
        return len(packet)

    def read(self, count):
        time.sleep(.0003)  # Expose potential cross-thread transaction races.
        data, self.rx = self.rx[:count], self.rx[count:]
        return data

    def flush(self):
        pass

    def close(self):
        self.closed = True

    @property
    def commands(self):
        return [(p[7], p[10], p[11], p[12]) for p in self.packets if p[1] == 16]


@pytest.fixture
def make_driver():
    drivers = []
    def make(**kwargs):
        serial = FakeSerial(**kwargs)
        cfg = GripperConfig(port='/dev/fake-gripper', poll_interval_s=.005, keepalive_interval_s=.025,
                            state_timeout_s=.5, io_timeout_s=.05, motion_timeout_s=2)
        driver = ContactExperimentGripper(cfg, execution_enabled=True, serial_factory=Mock(return_value=serial))
        drivers.append(driver)
        return driver, serial
    yield make
    for driver in drivers:
        driver.disconnect()


def test_complete_baseline_then_contact_trial_and_offline_plot(make_driver, tmp_path):
    driver, serial = make_driver()
    options = ContactSettings(100, startup_ignore_s=.02)
    report = driver.run_trial(options, mode='baseline', output_dir=tmp_path)
    b = EmptyBaseline.from_trial(report)
    assert report['stop_confirmed'] and report['outcome'] == 'travel_limit_no_contact'
    assert serial.commands == [(9, 225, 0, 0), (1, 225, 0, 0)]
    # Simulate an operator explicitly opening and adding an object.
    serial.position = 3
    serial.current_fn = lambda p: 60 if p >= 80 else 20
    report = driver.run_trial(options, baseline=b, output_dir=tmp_path)
    assert report['outcome'] == 'contact_travel_limit'
    assert report['stop_confirmed'] and not report['grasp_verified']
    assert report['contact']['source'] == 'current_rise_suspected'
    assert report['closure_after_contact_raw'] >= 1
    assert report['overshoot_raw'] >= 0
    assert len([c for c in serial.commands if c[0] == 9]) == 2  # No extra move after budget consumed.
    saved = json.loads(Path(report['log_path']).read_text())
    assert saved['outcome'] == report['outcome']
    artifacts = export_trial(saved)
    assert Path(artifacts['plot']).stat().st_size > 1000
    assert Path(artifacts['csv']).stat().st_size > 100


def test_target_update_is_bounded_and_written_only_once(make_driver, tmp_path):
    driver, serial = make_driver(current_fn=lambda p: 60 if p >= 80 else 20)
    options = ContactSettings(100, startup_ignore_s=0, extra_close_raw=8)
    report = driver.run_trial(options, baseline=baseline(driver.config, options), output_dir=tmp_path)
    moves = [c for c in serial.commands if c[0] == 9]
    assert len(moves) == 2
    assert moves[1][1] == report['contact']['position_raw']+8
    assert report['outcome'] == 'contact_travel_limit'
    assert report['overshoot_raw'] == 0


@pytest.mark.parametrize('native, ceiling, expected', [(None, True, 'current_limit'), (70, False, 'native_contact')])
def test_firmware_or_current_stop_never_retries(make_driver, tmp_path, native, ceiling, expected):
    driver, serial = make_driver(current_fn=lambda p: 100 if ceiling and p >= 70 else 20, native_at=native)
    options = ContactSettings(100, extra_close_raw=10)
    report = driver.run_trial(options, baseline=baseline(driver.config, options), output_dir=tmp_path)
    assert report['outcome'] == expected and report['stop_confirmed']
    assert serial.commands == [(9, 225, 0, 0), (1, 225, 0, 0)]
    if ceiling:
        assert report['contact'] is None


def test_lost_motion_ack_still_sends_stop_and_saves_error(make_driver, tmp_path):
    driver, serial = make_driver(missing_ack=True)
    with pytest.raises(GripperError):
        driver.run_trial(ContactSettings(100), mode='baseline', output_dir=tmp_path)
    assert serial.commands == [(9, 225, 0, 0), (1, 225, 0, 0)]
    assert driver.last_trial['stop_confirmed']
    assert json.loads(Path(driver.last_trial['log_path']).read_text())['error']


def test_ctrl_c_stops_without_open_reset_or_retry(make_driver, tmp_path):
    driver, serial = make_driver()
    original = driver._trial_write
    def interrupt(action, position, settings):
        original(action, position, settings)
        if action == 9:
            raise KeyboardInterrupt
    with patch.object(driver, '_trial_write', side_effect=interrupt), pytest.raises(KeyboardInterrupt):
        driver.run_trial(ContactSettings(100), mode='baseline', output_dir=tmp_path)
    assert serial.commands == [(9, 225, 0, 0), (1, 225, 0, 0)]
    assert driver.last_trial['outcome'] == 'interrupted'
    assert driver.last_trial['stop_confirmed']


def test_timeout_sends_stop_and_saves_report(make_driver, tmp_path):
    driver, serial = make_driver()
    driver.config = replace(driver.config, motion_timeout_s=.04)
    with pytest.raises(GripperError, match='timeout'):
        driver.run_trial(ContactSettings(100), mode='baseline', output_dir=tmp_path)
    assert serial.commands == [(9, 225, 0, 0), (1, 225, 0, 0)]
    assert driver.last_trial['stop_confirmed']


def test_stop_failure_is_explicit_and_never_retries_closure(make_driver, tmp_path):
    driver, serial = make_driver(current_fn=lambda p: 100)
    original = driver._write
    def broken_stop(action, *args, **kwargs):
        if action == 1:
            raise OSError('simulated stop communication loss')
        return original(action, *args, **kwargs)
    with patch.object(driver, '_write', side_effect=broken_stop), pytest.raises(GripperError, match='stop failed'):
        driver.run_trial(ContactSettings(100), mode='baseline', output_dir=tmp_path)
    assert driver.last_trial['outcome'] == 'stop_failed'
    assert not driver.last_trial.get('stop_confirmed')
    assert serial.commands == [(9, 225, 0, 0)]
    assert json.loads(Path(driver.last_trial['log_path']).read_text())['stop_error']


def test_baseline_speed_change_invalidates_before_moving_in_console(make_driver, tmp_path):
    driver, serial = make_driver()
    settings = ContactSettings(100)
    lines = iter(['set speed-raw 1', 'grasp', 'quit'])
    output = []
    console(driver.config, settings, execute=True, output_dir=tmp_path,
            baseline=baseline(driver.config, settings), input_fn=lambda _: next(lines),
            output_fn=output.append, driver_factory=lambda *a, **kw: driver)
    assert not serial.commands
    assert any('Capture an empty-close baseline first' in row for row in output)


def test_readonly_console_rejects_baseline(make_driver, tmp_path):
    driver, serial = make_driver()
    driver.execution_enabled = False
    lines = iter(['baseline', 'quit'])
    output = []
    console(driver.config, ContactSettings(100), execute=False, output_dir=tmp_path,
            input_fn=lambda _: next(lines), output_fn=output.append,
            driver_factory=lambda *a, **kw: driver)
    assert not serial.commands
    assert any('execution_enabled' in row for row in output)


def test_missing_baseline_invalid_options_and_readonly_do_not_connect(make_driver, tmp_path):
    driver, serial = make_driver()
    with pytest.raises(ValueError):
        driver.run_trial(ContactSettings(100), output_dir=tmp_path)
    with pytest.raises(ValueError):
        driver.run_trial(ContactSettings(100, extra_close_raw=-1), mode='baseline', output_dir=tmp_path)
    driver.execution_enabled = False
    with pytest.raises(GripperError):
        driver.run_trial(ContactSettings(100), mode='baseline', output_dir=tmp_path)
    assert not serial.packets


def test_wrong_start_and_configured_ceiling_cannot_move(make_driver, tmp_path):
    driver, serial = make_driver()
    driver.config = replace(driver.config, close_current_limit_ma=50)
    with pytest.raises(ValueError, match='ceiling'):
        driver.run_trial(ContactSettings(100), mode='baseline', output_dir=tmp_path)
    assert not serial.packets
    driver.config = replace(driver.config, close_current_limit_ma=None)
    serial.position = 50
    with pytest.raises(ValueError, match='start open'):
        driver.run_trial(ContactSettings(100), mode='baseline', output_dir=tmp_path)
    assert not serial.commands


def test_failed_baseline_never_builds_model(make_driver, tmp_path):
    driver, serial = make_driver(current_fn=lambda p: 100 if p >= 80 else 20)
    report = driver.run_trial(ContactSettings(100), mode='baseline', output_dir=tmp_path)
    with pytest.raises(ValueError, match='normally'):
        EmptyBaseline.from_trial(report)


def test_console_is_idle_until_operator_command_and_settings_do_not_write(make_driver, tmp_path):
    driver, serial = make_driver()
    lines = iter(['settings', 'set extra-close-raw 0', 'set current-limit-ma 110', 'quit'])
    console(driver.config, ContactSettings(100), execute=True, output_dir=tmp_path,
            input_fn=lambda _: next(lines), output_fn=lambda _: None,
            driver_factory=lambda *a, **kw: driver)
    assert not serial.commands
    assert serial.closed


def test_offline_plot_never_loads_hardware_config(make_driver, tmp_path):
    driver, _ = make_driver()
    report = driver.run_trial(ContactSettings(100), mode='baseline', output_dir=tmp_path)
    with patch('hardware.gripper_contact_cli.load_config', side_effect=AssertionError('offline')):
        assert main(['--plot', report['log_path']]) == 0


def test_invalid_cli_parameters_rejected_before_constructing_driver(tmp_path):
    with patch('hardware.gripper_contact_cli.ContactExperimentGripper') as constructor:
        assert main(['--current-limit-ma', '15', '--output-dir', str(tmp_path)]) == 1
        constructor.assert_not_called()
