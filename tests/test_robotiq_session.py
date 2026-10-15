"""Persistent-session tests using an in-memory Modbus device, never a USB port."""
from dataclasses import replace
import io
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from hardware.config import GripperConfig
from hardware.gripper_console import dispatch, run_console
from hardware.robotiq import GripperError, GripperCurrentLimitError, GripperStatus, Robotiq2F85, crc16
from hardware.runtime import LegacyPoseAdapter


class SimulatedSerial:
    def __init__(self, *, ready=True, startup_fault=0, contact=False, motion_reads=2, current_ma=0):
        self.current_ma = current_ma
        self.flags = 0x31 if ready else 0
        self.fault = startup_fault
        self.clear_idle_fault = startup_fault == 9
        self.position = 3
        self.requested = 0
        self.contact = contact
        self.motion_reads = motion_reads
        self.remaining = 0
        self.activation_reads = 0
        self.rx = b""
        self.writes = []
        self.closed = False
        self.fail_reads = False

    def reset_input_buffer(self):
        if self.rx:
            raise AssertionError("An interleaved transaction discarded another response")

    def write(self, packet):
        if self.closed:
            raise OSError("port closed")
        if self.rx:
            raise AssertionError("Request interleaved before the previous response ended")
        assert crc16(packet[:-2]) == packet[-2:]
        self.writes.append(packet)
        if packet[1] == 3:
            if self.activation_reads:
                self.activation_reads -= 1
                if not self.activation_reads:
                    self.flags = 0x31
            if self.remaining:
                self.remaining -= 1
                if not self.remaining:
                    self.flags = 0xB9 if self.contact else 0xF9
                    self.position = 155 if self.contact else self.requested
            response = bytes([9, 3, 6, self.flags, 0, self.fault, self.requested, self.position, self.current_ma // 10])
            if self.clear_idle_fault:
                self.fault = 0
                self.clear_idle_fault = False
        else:
            assert packet[1] == 16
            action = packet[7]
            if action == 0:
                self.flags, self.remaining, self.activation_reads = 0, 0, 0
            elif action == 1:
                if not self.flags & 1:
                    self.flags, self.activation_reads = 0x11, 2
                else:
                    self.flags, self.remaining = 0x31, 0
            elif action == 9:
                assert self.flags & 1
                self.flags, self.requested, self.remaining = 0x39, packet[10], self.motion_reads
            response = bytes.fromhex("09 10 03 E8 00 03")
        self.rx = response + crc16(response)
        return len(packet)

    def read(self, count):
        time.sleep(.0005)  # Release the GIL, exposing interleaving without locks.
        if self.fail_reads:
            self.rx = b""
            raise OSError("simulated cable disconnect")
        data, self.rx = self.rx[:count], self.rx[count:]
        return data

    def flush(self):
        pass

    def close(self):
        self.closed = True

    @property
    def actions(self):
        return [p[7] for p in self.writes if p[1] == 16]


class SessionTests(unittest.TestCase):
    def setUp(self):
        guard = patch("serial.Serial", side_effect=AssertionError("No real serial in tests"))
        guard.start()
        self.addCleanup(guard.stop)
        self.drivers = []

    def tearDown(self):
        for driver in self.drivers:
            driver.disconnect()

    def device(self, *, execute=True, **options):
        serial = SimulatedSerial(**options)
        factory = Mock(return_value=serial)
        cfg = GripperConfig(port="/dev/test-only", keepalive_interval_s=.02, state_timeout_s=.4,
                            poll_interval_s=.005, startup_timeout_s=.1, motion_timeout_s=.5)
        driver = Robotiq2F85(cfg, execution_enabled=execute, serial_factory=factory)
        self.drivers.append(driver)
        return driver, serial, factory

    @staticmethod
    def until(predicate, timeout=1):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError("Timed out waiting for simulated condition")
            time.sleep(.005)

    def test_startup_09_clears_with_reads_and_no_activation(self):
        driver, serial, _ = self.device(startup_fault=9, execute=False)
        self.assertTrue(driver.start_session().ready)
        self.assertEqual(serial.actions, [])
        self.assertGreaterEqual(driver.session_info()["successful_status_reads"], 2)

    def test_startup_09_timeout_never_sends_reset(self):
        driver, serial, _ = self.device(startup_fault=9)
        serial.clear_idle_fault = False
        with self.assertRaisesRegex(GripperError, "did not clear"):
            driver.open()
        self.assertEqual(serial.actions, [])
        self.assertTrue(serial.closed)

    def test_keepalive_runs_while_foreground_waits(self):
        driver, serial, _ = self.device(execute=False)
        driver.start_session()
        before = len(serial.writes)
        time.sleep(.15)
        self.assertGreater(len(serial.writes), before + 3)
        self.assertTrue(driver.assert_healthy(require_ready=True).ready)
        self.assertEqual(serial.actions, [])

    def test_disconnect_joins_polling_before_closing_port(self):
        driver, serial, _ = self.device()
        driver.start_session()
        driver.disconnect()
        count = len(serial.writes)
        time.sleep(.05)
        self.assertEqual(len(serial.writes), count)
        self.assertTrue(serial.closed)
        self.assertFalse(driver.session_info()["monitor_alive"])
        with self.assertRaisesRegex(GripperError, "closed"):
            driver.status()

    def test_motion_packets_and_background_reads_do_not_interleave(self):
        driver, serial, _ = self.device(contact=True, motion_reads=8)
        result = driver.move_raw(240, speed_raw=32, force_raw=48)
        self.assertTrue(result.object_detected_closing)
        packet = next(p for p in serial.writes if p[1] == 16 and p[7] == 9)
        self.assertEqual(packet[10:13], bytes([240, 32, 48]))
        self.assertTrue(driver.session_info()["monitor_alive"])
        self.assertIsNone(driver.session_info()["error"])

    def test_empty_close_is_a_completed_move_not_contact(self):
        driver, _, _ = self.device()
        result = driver.close()
        self.assertEqual(result.position_raw, 255)
        self.assertEqual(result.object_state, 3)
        self.assertFalse(result.object_detected_closing)

    def test_read_only_session_cannot_act(self):
        driver, serial, _ = self.device(execute=False)
        driver.start_session()
        for command in (driver.activate, driver.open, driver.close, driver.stop):
            with self.assertRaises(GripperError):
                command()
        self.assertEqual(serial.actions, [])

    def test_activation_only_occurs_after_explicit_command(self):
        driver, serial, _ = self.device(ready=False)
        state = driver.start_session()
        self.assertFalse(state.activated)
        with self.assertRaisesRegex(GripperError, "activate explicitly"):
            driver.open()
        self.assertEqual(serial.actions, [])
        self.assertTrue(driver.activate().ready)
        self.assertEqual(serial.actions, [0, 1])

    def test_healthy_already_activated_gripper_is_not_reactivated(self):
        driver, serial, _ = self.device()
        self.assertTrue(driver.activate().ready)
        self.assertEqual(serial.actions, [])

    def test_background_cable_error_is_latched_without_reconnect(self):
        driver, serial, factory = self.device()
        driver.start_session()
        serial.fail_reads = True
        self.until(lambda: driver.session_info()["error"])
        serial.fail_reads = False
        driver.status()  # A later good snapshot must not erase the failure.
        with self.assertRaisesRegex(GripperError, "polling failed"):
            driver.close()
        self.assertEqual(serial.actions, [])
        self.assertEqual(factory.call_count, 1)

    def test_fault_09_mid_session_blocks_commands_even_if_it_clears(self):
        driver, serial, _ = self.device()
        driver.start_session()
        serial.fault = 9
        self.until(lambda: driver.session_info()["error"])
        serial.fault = 0
        self.until(lambda: driver.status().fault == 0)
        with self.assertRaisesRegex(GripperError, "0x09 during active"):
            driver.open()
        self.assertEqual(serial.actions, [])

    def test_stale_status_blocks_arm_before_it_starts(self):
        driver, _, _ = self.device()
        driver.start_session()
        driver._monitor_stop.set()
        driver._monitor.join()
        driver._last_status_at -= 1
        arm = Mock()
        adapter = LegacyPoseAdapter(arm, healthcheck=driver.assert_healthy)
        with self.assertRaisesRegex(GripperError, "stale"):
            adapter.set_position(x=300, y=0, z=500, roll=3.14, pitch=0, yaw=0)
        arm.move_pose.assert_not_called()

    def test_explicit_reconnect_does_not_repeat_previous_action(self):
        driver, serial, factory = self.device()
        driver.start_session()
        driver._latch_error("test fault")
        driver.disconnect()
        replacement = SimulatedSerial(startup_fault=9)
        factory.return_value = replacement
        self.assertTrue(driver.start_session().ready)
        self.assertEqual(serial.actions, [])
        self.assertEqual(replacement.actions, [])

    def test_stop_interrupts_blocking_motion_without_opening(self):
        driver, serial, _ = self.device(motion_reads=1000)
        failures = []
        def move():
            try:
                driver.close()
            except GripperError as error:
                failures.append(str(error))
        thread = threading.Thread(target=move)
        thread.start()
        try:
            self.until(lambda: 9 in serial.actions)
            result = driver.stop()
            self.assertFalse(result.go_to)
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertTrue(failures)
            self.assertEqual(serial.actions[0], 9)
            self.assertTrue(all(action == 1 for action in serial.actions[1:]))
        finally:
            thread.join(timeout=2)

    def test_invalid_parameters_never_send_commands(self):
        driver, serial, factory = self.device()
        for args in [(None, {}), (256, {}), (1, {"force_raw": -1}), (1, {"speed_raw": True})]:
            with self.assertRaises(ValueError):
                driver.move_raw(args[0], **args[1])
        factory.assert_not_called()

    def test_motion_timeout_sends_stop_but_never_release_or_reset(self):
        driver, serial, _ = self.device(motion_reads=1000)
        driver.config = replace(driver.config, motion_timeout_s=.04)
        with self.assertRaisesRegex(GripperError, "operation timeout"):
            driver.close()
        self.assertEqual(serial.actions, [9, 1])

    def test_ctrl_c_during_motion_sends_stop(self):
        driver, serial, _ = self.device()
        with patch.object(driver, "_poll", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                driver.open()
        self.assertEqual(serial.actions, [9, 1])

    def test_ctrl_c_during_explicit_activation_aborts_it(self):
        driver, serial, _ = self.device(ready=False)
        with patch.object(driver, "_poll", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                driver.activate()
        self.assertEqual(serial.actions, [0, 0])

    def test_preexisting_motion_is_not_overridden(self):
        driver, serial, _ = self.device()
        serial.flags = 0x39
        with self.assertRaisesRegex(GripperError, "still moving"):
            driver.open()
        self.assertEqual(serial.actions, [])

    def test_console_stays_read_only_without_execute_and_logs_cleanup(self):
        driver, serial, _ = self.device(execute=False, startup_fault=9)
        lines = iter(["status", "open", "quit"])
        output = []
        with tempfile.TemporaryDirectory() as folder, patch("hardware.gripper_console.Robotiq2F85", return_value=driver):
            result = run_console(driver.config, output_dir=folder, input_fn=lambda _: next(lines), output_fn=output.append)
            log = [json.loads(s) for s in Path(result["log_path"]).read_text().splitlines()]
        self.assertEqual(serial.actions, [])
        self.assertTrue(any(row["event"] == "command_failed" for row in log))
        self.assertEqual(log[-1]["event"], "session_closed")
        self.assertFalse(log[-1]["session"]["monitor_alive"])

    def test_console_requires_complete_valid_argument_list(self):
        driver = Mock()
        for words in (["move"], ["move", "128", "32"], ["open", "32"], ["close", "32", "256"], ["status", "junk"]):
            with self.assertRaises(ValueError):
                dispatch(driver, words)
        driver.open.assert_not_called()
        driver.close.assert_not_called()
        driver.move_raw.assert_not_called()

    def test_current_cutoff_stops_closing_and_confirms_stop_without_release(self):
        driver, serial, _ = self.device(motion_reads=20, current_ma=60)
        with tempfile.TemporaryDirectory() as folder:
            driver.config = replace(driver.config, close_current_limit_ma=50, feedback_log_dir=folder)
            with self.assertRaises(GripperCurrentLimitError):
                driver.close()
            self.assertEqual(serial.actions, [9, 1])
            self.assertFalse(driver.assert_healthy().go_to)
            trace = json.loads(Path(driver.last_motion_feedback['log_path']).read_text())
            self.assertEqual(trace['outcome'], 'current_limit_stop')
            self.assertEqual(trace['cutoff_sample']['current_ma'], 60)
            self.assertFalse(trace['samples'][-1]['go_to'])
            self.assertTrue(trace['cutoff_triggered'])
            self.assertFalse(trace['current_is_force_measurement'])
            self.assertGreater(len(trace['samples']), 1)

    def test_current_cutoff_precedes_native_contact_acceptance(self):
        driver, serial, _ = self.device(contact=True, motion_reads=1, current_ma=50)
        driver.config = replace(driver.config, close_current_limit_ma=50)
        with self.assertRaises(GripperCurrentLimitError):
            driver.close()
        self.assertEqual(serial.actions, [9, 1])
        self.assertFalse(driver.assert_healthy().object_detected_closing)

    def test_below_limit_keeps_native_contact_and_low_force_packet(self):
        driver, serial, _ = self.device(contact=True, current_ma=20)
        driver.config = replace(driver.config, close_current_limit_ma=50, force_raw=0, speed_raw=32)
        self.assertTrue(driver.close().object_detected_closing)
        self.assertEqual(serial.actions, [9])
        packet = next(p for p in serial.writes if p[1] == 16)
        self.assertEqual(packet[10:13], bytes([255, 32, 0]))
        self.assertFalse(driver.last_motion_feedback['cutoff_triggered'])

    def test_logging_without_limit_does_not_claim_protection_or_contact(self):
        driver, serial, _ = self.device(current_ma=60)
        with tempfile.TemporaryDirectory() as folder:
            driver.config = replace(driver.config, feedback_log_dir=folder)
            self.assertFalse(driver.close().object_detected_closing)
            trace = json.loads(Path(driver.last_motion_feedback['log_path']).read_text())
            self.assertFalse(trace['cutoff_enabled'])
            self.assertEqual(trace['peak_current_ma'], 60)
            self.assertEqual(serial.actions, [9])

    def test_open_is_not_blocked_by_closing_current_threshold(self):
        driver, serial, _ = self.device(current_ma=100)
        driver.config = replace(driver.config, close_current_limit_ma=50)
        self.assertEqual(driver.open().position_raw, 0)
        self.assertEqual(serial.actions, [9])
        self.assertFalse(driver.last_motion_feedback['cutoff_triggered'])

    def test_background_peak_is_latched_and_stale_request_is_ignored(self):
        driver, _, factory = self.device()
        driver.config = replace(driver.config, close_current_limit_ma=50)
        idle = GripperStatus(True, False, 3, 3, 0, 0, 3, 0)
        moving = GripperStatus(True, True, 3, 0, 0, 255, 10, 60)
        driver._begin_motion_feedback(idle, 255, None, None)
        driver._record_motion_feedback(replace(moving, requested_position_raw=0))
        self.assertIsNone(driver._feedback_trip)
        driver._record_motion_feedback(moving)
        low = replace(moving, current_ma=10)
        driver._record_motion_feedback(low)
        with patch.object(driver, 'status', return_value=low):
            with self.assertRaises(GripperCurrentLimitError):
                driver._poll(lambda state: True)
        driver._finish_motion_feedback('test')
        factory.assert_not_called()

    def test_invalid_current_limit_never_connects(self):
        driver, _, factory = self.device()
        for value in (0, -10, 2551, 15, True, 50.0):
            driver.config = replace(driver.config, close_current_limit_ma=value)
            with self.assertRaises(ValueError):
                driver.close()
        factory.assert_not_called()

    def test_current_trip_clears_for_a_later_explicit_open(self):
        driver, serial, _ = self.device(motion_reads=1, contact=True, current_ma=60)
        driver.config = replace(driver.config, close_current_limit_ma=50)
        with self.assertRaises(GripperCurrentLimitError):
            driver.close()
        serial.contact = False
        self.assertEqual(driver.open().position_raw, 0)
        self.assertEqual(serial.actions, [9, 1, 9])


if __name__ == "__main__":
    unittest.main()
