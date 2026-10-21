"""Robotiq 2F-85 Modbus RTU adapter (pyserial, no ROS 1 dependency).

Construction is passive. status() reads registers only. activate() is an explicit
physical self-calibration; open/close never activate or clear faults implicitly.
"""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import math
import json
from pathlib import Path
import struct
import sys
import threading
import time
import uuid

from .config import GripperConfig, validate_gripper_feedback


class GripperError(RuntimeError):
    pass


class GripperCurrentLimitError(GripperError):
    """A closing-current cutoff is NOT evidence of a successfully held object."""
    pass


def crc16(data: bytes) -> bytes:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return struct.pack("<H", crc)


@dataclass(frozen=True)
class GripperStatus:
    activated: bool
    go_to: bool
    activation_state: int
    object_state: int
    fault: int
    requested_position_raw: int
    position_raw: int
    current_ma: int

    @classmethod
    def decode(cls, data):
        if len(data) != 6:
            raise GripperError("Expected six Robotiq status bytes")
        return cls(bool(data[0] & 1), bool(data[0] & 8), (data[0] >> 4) & 3,
                   (data[0] >> 6) & 3, data[2] & 15, data[3], data[4], data[5] * 10)

    @property
    def ready(self):
        return self.activated and self.activation_state == 3 and self.fault == 0

    @property
    def object_detected_closing(self):
        return self.ready and self.go_to and self.object_state == 2

    def to_dict(self):
        # The manual says gOBJ must be ignored when gGTO=0. In particular, a
        # zero gOBJ on an idle/faulted gripper is not evidence of finger motion.
        label = ("moving", "contact_opening", "contact_closing", "at_requested_position")[self.object_state]
        if not self.go_to or not self.ready:
            label = "not_applicable"
        return {**asdict(self), "ready": self.ready,
                "object_detected_closing": self.object_detected_closing,
                "fault_hex": f"0x{self.fault:02x}",
                "object_state_label": label}


class Robotiq2F85:
    def __init__(self, config: GripperConfig, *, execution_enabled=False, serial_factory=None):
        validate_gripper_feedback(config)
        self.config = config
        self.execution_enabled = execution_enabled
        self._serial_factory = serial_factory
        self._serial = None
        self._io_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._stop_requested = threading.Event()
        self._monitor = None
        self._session_started = False
        self._closed = False
        self._session_error = None
        self._last_status = None
        self._last_status_at = None
        self._read_count = 0
        self._write_count = 0
        self._motion_feedback = None
        self._feedback_started_at = None
        self._feedback_trip = None
        self.last_motion_feedback = None

    def _connect(self):
        if self._closed:
            raise GripperError("Session is closed; call start_session() explicitly to reconnect")
        if self._serial is not None:
            return
        if not self.config.port:
            raise GripperError("gripper.port 尚未配置；请确认 RS-485 设备后填写 /dev/serial/by-id/...，不会自动选端口")
        if self._serial_factory is None:
            from serial import Serial
            factory = Serial
        else:
            factory = self._serial_factory
        self._serial = factory(port=self.config.port, baudrate=self.config.baudrate,
                               bytesize=8, parity="N", stopbits=1,
                               timeout=self.config.io_timeout_s,
                               write_timeout=self.config.io_timeout_s, exclusive=True)

    def _read_exact(self, length):
        data = bytearray()
        deadline = time.monotonic() + self.config.io_timeout_s
        while len(data) < length:
            chunk = self._serial.read(length - len(data))
            data.extend(chunk)
            if not chunk or time.monotonic() > deadline:
                if len(data) != length:
                    raise GripperError("Modbus response timeout / truncated frame")
        return bytes(data)

    def _exchange(self, payload, function):
        # A complete Modbus transaction is atomic, including the reply. The
        # polling thread and command thread must never consume each other's data.
        with self._io_lock:
            return self._exchange_locked(payload, function)

    def _exchange_locked(self, payload, function):
        self._connect()
        self._serial.reset_input_buffer()
        frame = bytes([self.config.slave_id, function]) + payload
        packet = frame + crc16(frame)
        if self._serial.write(packet) != len(packet):
            raise GripperError("Incomplete Modbus request write")
        if function == 16:
            self._write_count += 1
        self._serial.flush()
        header = self._read_exact(2)
        if header[0] != self.config.slave_id:
            raise GripperError("Modbus slave id mismatch")
        if header[1] == (function | 0x80):
            response = header + self._read_exact(3)
        elif header[1] == 3 and function == 3:
            size = self._read_exact(1)
            if size[0] != 6:
                raise GripperError("Unexpected Modbus register byte count")
            response = header + size + self._read_exact(8)
        elif header[1] == 16 and function == 16:
            response = header + self._read_exact(6)
        else:
            raise GripperError("Modbus function mismatch")
        if crc16(response[:-2]) != response[-2:]:
            raise GripperError("Modbus response CRC mismatch")
        if response[1] & 0x80:
            raise GripperError(f"Modbus exception code {response[2]}")
        return response[2:-2]

    def status(self):
        """Read one raw snapshot. This does not activate or start keepalive."""
        with self._io_lock:
            previous_at = self._last_status_at
            payload = self._exchange(struct.pack(">HH", 0x07D0, 3), 3)
            state = GripperStatus.decode(payload[1:])
            self._last_status, self._last_status_at = state, time.monotonic()
            self._record_motion_feedback(state)
            self._read_count += 1
            if self._session_started:
                if previous_at is not None and self._last_status_at - previous_at > self.config.state_timeout_s:
                    self._latch_error("Status update gap exceeded configured state timeout")
                if state.fault not in (0, 5, 7):
                    self._latch_error(f"Robotiq fault 0x{state.fault:02x} during active session")
            return state

    def _begin_motion_feedback(self, state, position, speed_raw, force_raw):
        self._feedback_trip = None
        self._feedback_started_at = time.monotonic()
        log_path = None
        if self.config.feedback_log_dir:
            folder = Path(self.config.feedback_log_dir)
            # Fail before motion if the requested trace cannot be created.
            folder.mkdir(parents=True, exist_ok=True)
            log_path = folder / (uuid.uuid4().hex + '.json')
            log_path.write_text(json.dumps({'outcome': 'in_progress', 'started_unix_s': time.time()}))
        self._motion_feedback = dict(started_unix_s=time.time(),
            initial_status=state.to_dict(), requested_position_raw=position,
            direction='closing' if position > state.position_raw else 'opening_or_same',
            speed_raw=self.config.speed_raw if speed_raw is None else speed_raw,
            force_raw=self.config.force_raw if force_raw is None else force_raw,
            close_current_limit_ma=self.config.close_current_limit_ma,
            cutoff_enabled=self.config.close_current_limit_ma is not None,
            poll_interval_s=self.config.poll_interval_s, peak_current_ma=0,
            cutoff_triggered=False, samples=[], log_path=str(log_path) if log_path else None,
            current_is_force_measurement=False)

    def _record_motion_feedback(self, state):
        trace = self._motion_feedback
        if trace is None:
            return
        sample = {'elapsed_s': time.monotonic()-self._feedback_started_at, **state.to_dict()}
        trace['samples'].append(sample)
        trace['peak_current_ma'] = max(trace['peak_current_ma'], state.current_ma)
        limit = trace['close_current_limit_ma']
        # Never decide contact from current alone. The monitor may latch a trip,
        # but only the owning foreground command sends the stop request.
        if (self._feedback_trip is None and trace['direction'] == 'closing' and limit is not None
                and state.ready and state.go_to
                and state.requested_position_raw == trace['requested_position_raw']
                and state.current_ma >= limit):
            self._feedback_trip = sample
            trace['cutoff_triggered'] = True
            trace['cutoff_sample'] = sample

    def _finish_motion_feedback(self, outcome, error=None):
        with self._io_lock:
            trace, self._motion_feedback = self._motion_feedback, None
            if trace is None:
                return
            trace.update(outcome=outcome, error=error,
                         elapsed_s=time.monotonic()-self._feedback_started_at)
            self.last_motion_feedback = {k:v for k,v in trace.items() if k != 'samples'}
            self.last_motion_feedback['sample_count'] = len(trace['samples'])
            self._feedback_trip = None
        if trace['log_path']:
            try:
                Path(trace['log_path']).write_text(json.dumps(trace, indent=2))
            except OSError as log_error:
                self.last_motion_feedback['log_error'] = str(log_error)
                print(f'[Robotiq] Feedback log failed: {log_error}', file=sys.stderr)
            else:
                print(f"[Robotiq] Feedback log: {trace['log_path']}; outcome={outcome}, "
                      f"peak={trace['peak_current_ma']} mA", flush=True)

    def _latch_error(self, message):
        # Never erase a session failure just because a later read looks healthy.
        if self._session_error is None:
            self._session_error = str(message)

    def start_session(self):
        """Open/read/keepalive only. Startup 0x09 may clear by reading, not reset.

        An inactive gripper stays inactive. Subsequent communication errors are
        latched until explicit disconnect/start_session, never auto-reconnected.
        """
        with self._lifecycle_lock:
            if self._session_started:
                return self.assert_healthy()
            if not .005 <= self.config.keepalive_interval_s < self.config.state_timeout_s < 1:
                raise ValueError("Expected 0.005 <= keepalive interval < state timeout < 1 second")
            if not math.isfinite(self.config.startup_timeout_s) or self.config.startup_timeout_s <= 0:
                raise ValueError("startup_timeout_s must be finite and positive")
            self._closed = False
            self._session_error = None
            self._monitor_stop.clear()
            try:
                deadline = time.monotonic() + self.config.startup_timeout_s
                state = self.status()
                while state.fault == 9:
                    if time.monotonic() >= deadline:
                        raise GripperError("Startup communication fault 0x09 did not clear with status reads; no reset sent")
                    time.sleep(self.config.poll_interval_s)
                    state = self.status()
                if state.fault not in (0, 5, 7):
                    raise GripperError(f"Robotiq fault 0x{state.fault:02x}; no automatic reset")
                self._session_started = True
                self._monitor = threading.Thread(target=self._monitor_loop, name="robotiq-status", daemon=True)
                self._monitor.start()
                return state
            except BaseException:
                self._session_started = False
                self._closed = True
                if self._serial is not None:
                    self._serial.close()
                    self._serial = None
                raise

    def _monitor_loop(self):
        delay = self.config.keepalive_interval_s
        while not self._monitor_stop.wait(delay):
            started = time.monotonic()
            try:
                self.status()
            except Exception as error:
                with self._io_lock:
                    self._latch_error(f"Status polling failed: {error}")
                break  # No reconnect and no write from this thread.
            delay = max(.005, self.config.keepalive_interval_s - (time.monotonic() - started))

    def assert_healthy(self, *, require_ready=False):
        with self._io_lock:
            if not self._session_started or self._last_status_at is None:
                raise GripperError("No active gripper session; call start_session()")
            if time.monotonic() - self._last_status_at > self.config.state_timeout_s:
                self._latch_error("Cached gripper status is stale")
            if self._session_error is not None:
                raise GripperError(self._session_error + "; inspect then reconnect explicitly; no action retried")
            if require_ready and not self._last_status.ready:
                raise GripperError(f"Gripper not ready (fault=0x{self._last_status.fault:02x}); activate explicitly first")
            if require_ready and self._last_status.go_to and self._last_status.object_state == 0:
                raise GripperError("Gripper is still moving; wait for completion or issue stop()")
            return self._last_status

    def session_info(self):
        with self._io_lock:
            return {"active": self._session_started, "monitor_alive": bool(self._monitor and self._monitor.is_alive()),
                    "error": self._session_error, "keepalive_interval_s": self.config.keepalive_interval_s,
                    "last_status_age_s": None if self._last_status_at is None else time.monotonic() - self._last_status_at,
                    "successful_status_reads": self._read_count, "write_register_requests": self._write_count,
                    "execution_enabled": self.execution_enabled,
                    "close_current_limit_ma": self.config.close_current_limit_ma,
                    "last_motion_feedback": self.last_motion_feedback}

    @contextmanager
    def _operation(self):
        self._require_execution()
        if not self._command_lock.acquire(blocking=False):
            raise GripperError("Another gripper operation is in progress; only stop() may interrupt it")
        self._stop_requested.clear()
        try:
            self.start_session()
            yield
        finally:
            self._command_lock.release()

    def _write(self, action, position=0, *, speed_raw=None, force_raw=None):
        if not self.execution_enabled:
            raise GripperError("Gripper writes require execution_enabled=True (CLI: --execute)")
        speed = self.config.speed_raw if speed_raw is None else speed_raw
        force = self.config.force_raw if force_raw is None else force_raw
        data = bytes([action, 0, 0, position, speed, force])
        response = self._exchange(struct.pack(">HHB", 0x03E8, 3, 6) + data, 16)
        if response != struct.pack(">HH", 0x03E8, 3):
            raise GripperError("Modbus write acknowledgement mismatch")

    def _require_execution(self):
        if not self.execution_enabled:
            raise GripperError("Gripper action requires execution_enabled=True (CLI: --execute)")

    def _poll(self, predicate, *, activation=False, stopping=False):
        deadline = time.monotonic() + self.config.motion_timeout_s
        while time.monotonic() < deadline:
            if self._stop_requested.is_set() and not stopping:
                raise GripperError("Gripper operation cancelled by stop()")
            time.sleep(self.config.poll_interval_s)
            state = self.status()
            if self._session_started and not stopping:
                self.assert_healthy()
            if self._stop_requested.is_set() and not stopping:
                raise GripperError("Gripper operation cancelled by stop()")
            if state.fault not in ((0, 5, 7) if activation else (0,)):
                raise GripperError(f"Robotiq fault 0x{state.fault:02x}; no automatic reset/retry")
            if not stopping and self._feedback_trip is not None:
                trace = self._motion_feedback
                raise GripperCurrentLimitError(
                    f"Robotiq closing current {self._feedback_trip['current_ma']} mA reached "
                    f"limit {trace['close_current_limit_ma']} mA; stop closing, no lift/retry; "
                    f"feedback log: {trace['log_path']}")
            if predicate(state):
                return state
        raise GripperError("Robotiq operation timeout")

    def activate(self):
        with self._operation():
            state = self.assert_healthy()
            if state.ready:
                return state
            try:
                self._write(0)
                self._poll(lambda s: not s.activated and s.activation_state == 0, activation=True)
                with self._io_lock:
                    if self._stop_requested.is_set():
                        raise GripperError("Activation cancelled")
                    self.assert_healthy()
                    self._write(1)
                return self._poll(lambda s: s.ready, activation=True)
            except (Exception, KeyboardInterrupt) as error:
                try:
                    self._write(0)  # Abort this explicit activation, never auto-release.
                except Exception as stop_error:
                    raise GripperError(f"Activation failed ({error!r}); abort also failed ({stop_error!r})") from error
                raise

    def move_raw(self, position, *, speed_raw=None, force_raw=None):
        self._require_execution()
        validate_gripper_feedback(self.config)
        for name, value in [("position", position), ("speed_raw", speed_raw), ("force_raw", force_raw)]:
            if value is not None and (type(value) is not int or not 0 <= value <= 255):
                raise ValueError(f"{name} must be an integer in [0, 255]")
        if position is None:
            raise ValueError("position must be an integer in [0, 255]")
        with self._operation():
            self.assert_healthy(require_ready=True)
            outcome, failure = 'failed', None
            try:
                with self._io_lock:
                    if self._stop_requested.is_set():
                        raise GripperError("Gripper operation cancelled by stop()")
                    state = self.assert_healthy(require_ready=True)
                    self._begin_motion_feedback(state, position, speed_raw, force_raw)
                    self._write(9, position, speed_raw=speed_raw, force_raw=force_raw)
                # Check request echo so an old completion cannot satisfy this move.
                state = self._poll(lambda s: s.ready and s.go_to and
                                   s.requested_position_raw == position and s.object_state != 0)
                outcome = 'contact_closing' if state.object_detected_closing else 'completed_without_closing_contact'
                return state
            except (Exception, KeyboardInterrupt) as error:
                failure = repr(error)
                if isinstance(error, GripperCurrentLimitError):
                    outcome = 'current_limit_stop'
                try:
                    # Do not set rACT=1 if a received state says activation was lost.
                    state = self._last_status
                    if state and state.activated and state.activation_state == 3:
                        self._write(1, position)  # rGTO=0; does not open/release.
                        if isinstance(error, GripperCurrentLimitError):
                            self._poll(lambda s: not s.go_to, stopping=True)
                except Exception as stop_error:
                    outcome = 'stop_failed'
                    raise GripperError(f"Motion failed ({error!r}); stop also failed ({stop_error!r})") from error
                raise
            finally:
                self._finish_motion_feedback(outcome, failure)

    def open(self, **kwargs):
        return self.move_raw(self.config.open_raw, **kwargs)

    def close(self, **kwargs):
        return self.move_raw(self.config.close_raw, **kwargs)

    def close_completed(self, state):
        """Command completion, separate from proof of object contact/retention."""
        return bool(state.ready and state.go_to and state.object_state in (2, 3))

    def may_be_holding_object(self):
        """Cleanup hint only; never a force measurement or a motion permission."""
        state = self._last_status
        return bool(state is not None and state.object_detected_closing)

    def stop(self):
        self._require_execution()
        # Allowed concurrently with a blocking command; it does not wait for the
        # command lock. The command's poll loop observes this cancellation flag.
        self._stop_requested.set()
        state = self.status()
        if not state.activated:
            return state
        if state.activation_state != 3:
            self._write(0)
            return self._poll(lambda s: not s.activated, activation=True, stopping=True)
        self._write(1, state.requested_position_raw)
        return self._poll(lambda s: not s.go_to, stopping=True)

    def disconnect(self):
        with self._lifecycle_lock:
            if not self._command_lock.acquire(blocking=False):
                raise GripperError("Cannot disconnect during a command; stop it first")
            try:
                self._monitor_stop.set()
                # Join before closing the port: no poll can reopen it on exit.
                if self._monitor is not None:
                    self._monitor.join(timeout=self.config.io_timeout_s * 4 + 1)
                    if self._monitor.is_alive():
                        raise GripperError("Status monitor did not stop; port not closed")
                with self._io_lock:
                    self._session_started = False
                    self._closed = True
                    if self._serial is not None:
                        self._serial.close()
                        self._serial = None
                    self._monitor = None
            finally:
                self._command_lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.disconnect()
