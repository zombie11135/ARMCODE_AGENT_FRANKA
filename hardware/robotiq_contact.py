"""Agent adapter for the same contact policy validated in the gripper console.

Construction loads files only. No implicit baseline capture, open, activation,
fallback to a full close, plotting, or motion retry. Raw status stays truthful:
gGTO is zero after the policy's explicit stop.
"""
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

from .gripper_contact import ContactExperimentGripper, ContactSettings, EmptyBaseline
from .robotiq import GripperError


@dataclass(frozen=True)
class ContactCloseSetup:
    profile_path: Path
    baseline_path: Path
    baseline: EmptyBaseline
    settings: ContactSettings
    output_dir: Path
    poll_interval_s: float
    keepalive_interval_s: float


def load_contact_close_setup(config):
    """Fail offline before any hardware session if the saved baseline is unusable."""
    if not config.contact_close_file:
        raise ValueError('gripper.contact_close_file is not configured')
    path = Path(config.contact_close_file).expanduser().resolve()
    data = json.loads(path.read_text())
    keys = {'schema_version', 'baseline_file', 'settings', 'output_dir',
            'poll_interval_s', 'keepalive_interval_s'}
    if not isinstance(data, dict) or set(data) != keys or type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ValueError(f'Invalid contact-close profile: {path}; expected schema 1 and keys {sorted(keys)}')
    if not isinstance(data['settings'], dict):
        raise ValueError('contact-close settings must be a mapping')
    settings = ContactSettings(**data['settings'])
    settings.validate()
    if config.close_current_limit_ma is not None and settings.current_limit_ma > config.close_current_limit_ma:
        raise ValueError('Contact-close current ceiling exceeds gripper.close_current_limit_ma')
    for key in ('poll_interval_s', 'keepalive_interval_s'):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not .005 <= value <= .1:
            raise ValueError(f'contact-close {key} must be 0.005..0.1 seconds')
        if value >= min(config.state_timeout_s, settings.max_sample_gap_s):
            raise ValueError(f'contact-close {key} must be below feedback/session timeouts')

    def resolve(key):
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'contact-close {key} must be a nonempty path')
        file = Path(value).expanduser()
        return (file if file.is_absolute() else path.parent/file).resolve()

    baseline_path = resolve('baseline_file')
    baseline = EmptyBaseline(**json.loads(baseline_path.read_text()))
    baseline.check(config, settings, baseline.start_raw)
    return ContactCloseSetup(path, baseline_path, baseline, settings, resolve('output_dir'),
                             data['poll_interval_s'], data['keepalive_interval_s'])


class AgentContactGripper(ContactExperimentGripper):
    NORMAL_COMPLETION = frozenset(('contact_travel_limit', 'contact_target_reached',
                                   'native_contact', 'travel_limit_no_contact', 'current_limit'))

    def __init__(self, config, *, execution_enabled=False, serial_factory=None):
        self.contact_setup = load_contact_close_setup(config)
        cfg = replace(config, poll_interval_s=self.contact_setup.poll_interval_s,
                      keepalive_interval_s=self.contact_setup.keepalive_interval_s)
        super().__init__(cfg, execution_enabled=execution_enabled, serial_factory=serial_factory)
        self.last_contact_close = None
        self._contact_close_completed = False

    def move_raw(self, position, **kwargs):
        # Open/manual moves invalidate any previous close-completion/holding hint.
        self._contact_close_completed = False
        return super().move_raw(position, **kwargs)

    def close(self, *, speed_raw=None, force_raw=None):
        self._require_execution()
        setup = self.contact_setup
        for name, value in (('speed_raw', speed_raw), ('force_raw', force_raw)):
            if value is not None and (type(value) is not int or value != getattr(setup.settings, name)):
                raise ValueError(f'Contact close uses calibrated {name}={getattr(setup.settings, name)}; '
                                 'changing it requires a matching baseline/profile')
        self._contact_close_completed = False
        self.last_contact_close = None
        previous = self.last_trial
        try:
            report = self.run_trial(setup.settings, baseline=setup.baseline,
                                    mode='grasp', output_dir=setup.output_dir)
            outcome = report['outcome']
            if (outcome not in self.NORMAL_COMPLETION or not report.get('stop_confirmed')
                    or report.get('error') or report.get('stop_error')):
                raise GripperError(f'Robotiq contact close failed: {outcome}; log: {report["log_path"]}')
            state = self.assert_healthy(require_ready=True)
            if state.go_to:
                raise GripperError(f'Robotiq contact close stop not reflected in status; log: {report["log_path"]}')
            # A confirmed current-limited stop completes this close command.
            # Keep outcome=current_limit and grasp_verified=False in telemetry;
            # command completion does not prove contact or object retention.
            self._contact_close_completed = True
            return state
        finally:
            if self.last_trial is not previous:
                self.last_contact_close = self.last_trial
                report = self.last_contact_close
                self.last_motion_feedback = {key: report.get(key) for key in (
                    'log_path', 'outcome', 'contact', 'target_raw', 'closure_after_contact_raw',
                    'overshoot_raw', 'peak_current_ma', 'stop_confirmed', 'sample_count',
                    'error', 'stop_error', 'grasp_verified')}
                self.last_motion_feedback.update(strategy='contact_close',
                    completed=self._contact_close_completed, current_is_force_measurement=False,
                    extra_close_raw=setup.settings.extra_close_raw, baseline_file=str(setup.baseline_path))
                print('[Robotiq contact close] '+json.dumps(self.last_motion_feedback, ensure_ascii=False), flush=True)

    def close_completed(self, state):
        return bool(self._contact_close_completed and state.ready and not state.go_to)

    def may_be_holding_object(self):
        # gOBJ is invalid after rGTO=0. Retain the observed contact only as a
        # cleanup hint. Current limiting can precede a recorded contact, so it
        # also preserves the existing manual release console on task exit.
        state = self._last_status
        return bool(self._contact_close_completed and self.last_contact_close
                    and (self.last_contact_close.get('contact') is not None
                         or self.last_contact_close.get('outcome') == 'current_limit')
                    and state is not None and state.ready and not state.go_to)
