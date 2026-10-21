"""Gripper-only contact experiments. No arm, camera, ROS or perception imports.

Current is a contact proxy, never a calibrated force measurement. This module is
opt-in and does not change the agent's existing open/close implementation.
"""
from bisect import bisect_left
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import median
import time
import uuid

from .robotiq import GripperError, Robotiq2F85


@dataclass(frozen=True)
class ContactSettings:
    current_limit_ma: int  # Explicit experimental ceiling, not a force rating.
    contact_rise_ma: int = 30
    confirm_samples: int = 2
    extra_close_raw: int = 1
    max_close_raw: int = 225
    speed_raw: int = 0
    force_raw: int = 0
    startup_ignore_s: float = .10
    max_sample_gap_s: float = .20

    def validate(self):
        for name in ('current_limit_ma', 'contact_rise_ma'):
            value = getattr(self, name)
            if type(value) is not int or not 10 <= value <= 2550 or value % 10:
                raise ValueError(f'{name}: integer 10..2550 mA, in steps of 10')
        if self.contact_rise_ma >= self.current_limit_ma:
            raise ValueError('contact_rise_ma must be below current_limit_ma')
        for name, lo, hi in (('confirm_samples', 1, 10), ('extra_close_raw', 0, 20),
                             ('max_close_raw', 1, 255), ('speed_raw', 0, 255), ('force_raw', 0, 255)):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'{name}: integer {lo}..{hi}')
        for name, lo, hi in (('startup_ignore_s', 0, .5), ('max_sample_gap_s', .02, .5)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{name}: finite {lo}..{hi} seconds')


def identity(config, settings):
    return dict(port=config.port, baudrate=config.baudrate, slave_id=config.slave_id,
                speed_raw=settings.speed_raw, force_raw=settings.force_raw,
                max_close_raw=settings.max_close_raw, startup_ignore_s=settings.startup_ignore_s)


@dataclass
class EmptyBaseline:
    device: dict
    start_raw: int
    positions: list
    currents_ma: list
    source_log: str

    def validate(self):
        if len(self.positions) < 5 or len(self.positions) != len(self.currents_ma):
            raise ValueError('Baseline needs at least five position bins')
        if type(self.start_raw) is not int or not 0 <= self.start_raw <= 255:
            raise ValueError('Invalid baseline start position')
        for values, lo, hi in ((self.positions, 0, 255), (self.currents_ma, 0, 2550)):
            if any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or not lo <= x <= hi for x in values):
                raise ValueError('Invalid baseline values')
        if any(a >= b for a, b in zip(self.positions, self.positions[1:])):
            raise ValueError('Baseline positions must strictly increase')
        if any(b-a > 24 for a, b in zip(self.positions, self.positions[1:])):
            raise ValueError('Baseline has a large gap; repeat slower / improve polling')

    def check(self, config, settings, start_raw):
        self.validate()
        if self.device != identity(config, settings):
            raise ValueError('Baseline device/speed/force/travel/startup settings differ; capture baseline again')
        if abs(start_raw-self.start_raw) > 4:
            raise ValueError('Start opening differs from baseline; issue open first')

    def at(self, position):
        """Piecewise-linear, binned empty-close current; endpoints clamp."""
        index = bisect_left(self.positions, position)
        if index == 0:
            return self.currents_ma[0]
        if index == len(self.positions):
            return self.currents_ma[-1]
        a, b = self.positions[index-1:index+1]
        u, v = self.currents_ma[index-1:index+1]
        return u + (v-u)*(position-a)/(b-a)

    @classmethod
    def from_trial(cls, report):
        if report['mode'] != 'baseline' or report['outcome'] != 'travel_limit_no_contact':
            raise ValueError('Empty-close did not complete normally; no baseline accepted')
        samples = [s for s in report['samples'] if s['phase'] == 'search'
                   and s['ready'] and s['go_to'] and s['object_state'] == 0
                   and s['requested_position_raw'] == report['settings']['max_close_raw']
                   and s['elapsed_s'] >= report['settings']['startup_ignore_s']]
        if len(samples) < 12:
            raise ValueError('Too few moving samples for a baseline')
        bins = {}
        for sample in samples:
            bins.setdefault(sample['position_raw']//8, []).append(sample)
        groups = [bins[key] for key in sorted(bins)]
        result = cls(report['device'], report['initial_status']['position_raw'],
                     [median(s['position_raw'] for s in group) for group in groups],
                     [median(s['current_ma'] for s in group) for group in groups], report['log_path'])
        result.validate()
        start, end = result.start_raw, report['settings']['max_close_raw']
        if end-start < 60 or result.positions[0]-start > 24 or end-result.positions[-1] > 16:
            raise ValueError('Baseline does not cover enough travel; repeat from open at lower speed')
        return result


class ContactPolicy:
    """Pure decision logic; every status read, including keepalive, is observed."""
    def __init__(self, settings, baseline=None, *, mode='grasp'):
        settings.validate()
        if mode not in ('grasp', 'baseline'):
            raise ValueError('Invalid experiment mode')
        if mode == 'grasp' and baseline is None:
            raise ValueError('Capture an empty-close baseline first')
        self.settings, self.baseline, self.mode = settings, baseline, mode
        self.phase = 'search'
        self.request_raw = settings.max_close_raw
        self.target_raw = None
        self.pending_target = None
        self.reason = None
        self.contact = None
        self.last_time = None
        self.last_valid_time = 0.
        self.run = []

    def observe(self, state, elapsed):
        s = self.settings
        base = self.baseline.at(state.position_raw) if self.baseline else None
        row = dict(elapsed_s=elapsed, **state.to_dict(), phase=self.phase,
                   baseline_ma=base, excess_ma=None if base is None else state.current_ma-base,
                   gap_ms=1000*(elapsed if self.last_time is None else elapsed-self.last_time),
                   target_raw=self.target_raw)
        gap = elapsed if self.last_time is None else elapsed-self.last_time
        self.last_time = elapsed
        if self.reason:
            return row
        if gap > s.max_sample_gap_s:
            self.reason = 'feedback_gap'
        elif not state.ready:
            self.reason = 'not_ready_or_fault'
        # The hard ceiling applies even during startup and an echo transition.
        elif state.go_to and state.current_ma >= s.current_limit_ma:
            self.reason = 'current_limit'
        elif not state.go_to or state.requested_position_raw != self.request_raw:
            if elapsed-self.last_valid_time > s.max_sample_gap_s:
                self.reason = 'command_echo_timeout'
        else:
            self.last_valid_time = elapsed
            if state.object_detected_closing:
                # Firmware already stopped on contact: never force another move.
                self.contact = self.contact or dict(source='native', position_raw=state.position_raw,
                                                   elapsed_s=elapsed, confirmed_elapsed_s=elapsed)
                self.reason = 'native_contact'
            elif self.phase == 'advance' and state.position_raw >= self.target_raw:
                self.reason = 'contact_travel_limit'
            elif state.object_state == 3 or state.position_raw >= s.max_close_raw:
                self.reason = 'contact_target_reached' if self.phase == 'advance' else 'travel_limit_no_contact'
            elif state.object_state != 0:
                self.reason = 'unexpected_object_state'
            elif self.phase == 'search' and self.mode == 'grasp' and elapsed >= s.startup_ignore_s:
                if row['excess_ma'] >= s.contact_rise_ma:
                    self.run.append((state.position_raw, elapsed))
                else:
                    self.run.clear()
                if len(self.run) >= s.confirm_samples:
                    # Anchor at the FIRST sustained rise, not the later confirmation.
                    position, timestamp = self.run[0]
                    self.contact = dict(source='current_rise_suspected', position_raw=position,
                                        elapsed_s=timestamp, confirmed_elapsed_s=elapsed)
                    self.target_raw = min(s.max_close_raw, position+s.extra_close_raw)
                    self.phase = 'advance'
                    if state.position_raw >= self.target_raw:
                        self.reason = 'contact_travel_limit'
                    else:
                        self.pending_target = self.target_raw
        return row


class ContactExperimentGripper(Robotiq2F85):
    """Dedicated experimental driver. Motion always requires execution_enabled."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trial = None
        self._policy = None
        self.last_trial = None

    def _record_motion_feedback(self, state):
        super()._record_motion_feedback(state)
        if self._trial is not None:
            row = self._policy.observe(state, time.monotonic()-self._trial_started)
            self._trial['samples'].append(row)

    def run_trial(self, settings, *, baseline=None, mode='grasp', output_dir):
        self._require_execution()
        settings.validate()
        policy = ContactPolicy(settings, baseline, mode=mode)
        if self.config.close_current_limit_ma is not None and settings.current_limit_ma > self.config.close_current_limit_ma:
            raise ValueError('Experiment ceiling exceeds the configured closing-current limit')
        folder = Path(output_dir).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]+'.json')
        with self._operation():
            initial = self.status()
            self.assert_healthy(require_ready=True)
            if initial.position_raw >= settings.max_close_raw:
                raise ValueError('Gripper already at/beyond travel limit; issue open first')
            if mode == 'baseline' and initial.position_raw > self.config.open_raw+8:
                raise ValueError('Baseline must start open; issue open first with empty jaws')
            if baseline is not None:
                baseline.check(self.config, settings, initial.position_raw)
            report = dict(schema_version=1, mode=mode, started_unix_s=time.time(),
                          log_path=str(path), settings=asdict(settings), device=identity(self.config, settings),
                          initial_status=initial.to_dict(), baseline=None if baseline is None else asdict(baseline),
                          current_is_force_measurement=False, grasp_verified=False,
                          outcome='in_progress', samples=[], commands=[])
            path.write_text(json.dumps(report, indent=2))  # Check log access BEFORE motion.
            self.last_trial = report
            motion_possible = False
            self._trial_started = time.monotonic()
            try:
                with self._io_lock:
                    self._policy, self._trial = policy, report
                    motion_possible = True  # A lost acknowledgement may still mean motion.
                    self._trial_write(9, settings.max_close_raw, settings)
                deadline = time.monotonic()+self.config.motion_timeout_s
                while True:
                    if self._stop_requested.is_set():
                        raise GripperError('Experiment cancelled by stop()')
                    self.status()
                    self.assert_healthy()
                    with self._io_lock:
                        if policy.reason is not None:
                            report['outcome'] = policy.reason
                            report['decision_elapsed_s'] = time.monotonic()-self._trial_started
                            break
                        if policy.pending_target is not None:
                            if self._stop_requested.is_set():
                                raise GripperError('Experiment cancelled before extra closure')
                            target = policy.pending_target
                            self._trial_write(9, target, settings)
                            policy.request_raw, policy.pending_target = target, None
                    if time.monotonic() >= deadline:
                        raise GripperError('Experiment motion timeout')
                    time.sleep(self.config.poll_interval_s)
                self._trial_stop(settings)
                motion_possible = False
            except BaseException as error:
                policy.reason = policy.reason or ('interrupted' if isinstance(error, KeyboardInterrupt) else 'error')
                report.update(outcome=policy.reason, error=repr(error))
                if motion_possible:
                    try:
                        self._trial_stop(settings)
                    except BaseException as stop_error:
                        report.update(outcome='stop_failed', stop_error=repr(stop_error))
                        raise GripperError(f'Experiment failed ({error!r}); stop failed ({stop_error!r})') from error
                raise
            finally:
                with self._io_lock:
                    self._trial = None
                    report.update(contact=policy.contact, target_raw=policy.target_raw,
                                  elapsed_s=time.monotonic()-self._trial_started)
                    if self._last_status is not None:
                        report['final_status'] = self._last_status.to_dict()
                        if policy.contact:
                            report['closure_after_contact_raw'] = self._last_status.position_raw-policy.contact['position_raw']
                            report['overshoot_raw'] = (None if policy.target_raw is None else
                                max(0, self._last_status.position_raw-policy.target_raw))
                    if report.get('stop_confirmed'):
                        stop_time = report['stop_confirmed_elapsed_s']
                        stop_commands = [c for c in report['commands'] if c['action'] == 1]
                        report['stop_latency_ms'] = 1000*(stop_time-stop_commands[-1]['elapsed_s'])
                        if policy.contact:
                            report['contact_to_stop_ms'] = 1000*(stop_time-policy.contact['elapsed_s'])
                    rows = report['samples']
                    report['peak_current_ma'] = max((x['current_ma'] for x in rows), default=0)
                    report['max_sample_gap_ms'] = max((x['gap_ms'] or 0 for x in rows), default=0)
                    report['sample_count'] = len(rows)
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            return report

    def _trial_write(self, action, position, settings):
        if action == 9 and self._stop_requested.is_set():
            raise GripperError('Experiment cancelled before motion write')
        entry = dict(elapsed_s=time.monotonic()-self._trial_started, action=action,
                     position_raw=position, speed_raw=settings.speed_raw, force_raw=settings.force_raw)
        self._trial['commands'].append(entry)
        self._write(action, position, speed_raw=settings.speed_raw, force_raw=settings.force_raw)
        entry['ack_elapsed_s'] = time.monotonic()-self._trial_started

    def _trial_stop(self, settings):
        with self._io_lock:
            state = self._last_status
            if state is None or not state.activated or state.activation_state != 3:
                raise GripperError('Cannot send a normal stop to an unready gripper; no activation/reset attempted')
            self._trial_write(1, self._policy.request_raw, settings)
        state = self._poll(lambda s: not s.go_to, stopping=True)
        self._trial['stop_confirmed_elapsed_s'] = time.monotonic()-self._trial_started
        self._trial['stop_confirmed'] = not state.go_to


def export_trial(report):
    """Offline CSV and PNG. Plotting runs only AFTER motion has stopped."""
    import csv
    path = Path(report['log_path'])
    rows = report['samples']
    csv_path = path.with_suffix('.csv')
    if rows:
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), constrained_layout=True)
    try:
        t = [r['elapsed_s'] for r in rows]
        axes[0].plot(t, [r['current_ma'] for r in rows], label='Motor current (not force)')
        axes[0].axhline(report['settings']['current_limit_ma'], color='red', linestyle='--', label='Current ceiling')
        if report.get('baseline'):
            axes[0].plot(t, [r['baseline_ma'] for r in rows], label='Empty-close baseline')
            axes[0].plot(t, [r['baseline_ma']+report['settings']['contact_rise_ma'] for r in rows],
                         linestyle=':', label='Contact threshold')
        axes[0].set_ylabel('mA')
        axes[1].plot(t, [r['position_raw'] for r in rows], label='Position (larger = more closed)')
        if report.get('target_raw') is not None:
            axes[1].axhline(report['target_raw'], linestyle='--', label='Contact travel limit')
        axes[1].set_ylabel('Position raw')
        axes[1].set_xlabel('Seconds since close command')
        for axis in axes:
            if report.get('contact'):
                axis.axvline(report['contact']['elapsed_s'], color='green', linestyle=':', label='Contact candidate')
            if report.get('stop_confirmed_elapsed_s') is not None:
                axis.axvline(report['stop_confirmed_elapsed_s'], color='black', linestyle=':', label='Stop confirmed')
            axis.legend(fontsize=8)
            axis.grid(alpha=.25)
        fig.suptitle(report['mode']+' / '+report['outcome'])
        png_path = path.with_suffix('.png')
        fig.savefig(png_path, dpi=140)
    finally:
        plt.close(fig)
    return dict(csv=str(csv_path), plot=str(png_path))
