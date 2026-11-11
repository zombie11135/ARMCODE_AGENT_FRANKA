"""Task wall-clock spans, including nested work on explicitly shared threads.

Observing timing never constructs a hardware or model backend.
Durations are inclusive; concurrent or nested spans must not be added together.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
import hashlib
import os
from pathlib import Path
import threading
import time
from uuid import uuid4


_recorder = ContextVar('task_timing_recorder', default=None)
_parent = ContextVar('task_timing_parent', default=None)


def _result_attributes(result):
    success = (result if type(result) is bool else
               result.get('success', result.get('ok')) if isinstance(result, dict) else
               getattr(result, 'success', None))
    attrs = {}
    if type(success) is bool:
        attrs['status'] = 'ok' if success else 'failed'
    if isinstance(result, dict):
        for key in ('inference_cache_hit', 'dry_run'):
            if type(result.get(key)) is bool:
                attrs[key] = result[key]
    data = result if isinstance(result, dict) else getattr(result, 'data', None)
    if isinstance(data, dict):
        for key in ('failure_type', 'backend', 'confidence', 'target_visible_in_container', 'held_by_gripper'):
            if key in data:
                attrs[key] = data[key]
    return attrs


class TimingRecorder:
    def __init__(self, goal, report_path, metadata=None):
        self.run_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:8]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started = time.perf_counter()
        self.monotonic_started = time.monotonic()
        self.owner_thread = threading.get_ident()
        self.goal, self.report_path = goal, Path(report_path)
        self.metadata = dict(metadata or {})
        self.status = 'ok'
        self.spans = []
        self._lock = threading.Lock()
        self._next_id = 0

    def next_id(self):
        with self._lock:
            self._next_id += 1
            return self._next_id

    def finish(self):
        duration = time.perf_counter() - self.started
        spans = sorted(self.spans, key=lambda row: row['id'])
        totals = {}
        for row in spans:
            key = (row['name'], row['detail'], row['background'])
            total = totals.setdefault(key, dict(name=key[0], detail=key[1], background=key[2],
                                               calls=0, duration_s=0., failed_calls=0))
            total['calls'] += 1
            total['duration_s'] += row['duration_s']
            total['failed_calls'] += row['status'] != 'ok'
        history = self.report_path.parent / 'timing' / (self.run_id + '.json')
        report = dict(schema='armcode_task_timing_v1', run_id=self.run_id, goal=self.goal,
            started_at=self.started_at, status=self.status, duration_s=duration,
            metadata=self.metadata, spans=spans, summary=list(totals.values()),
            history_path=str(history),
            note='Wall-clock inclusive durations; nested and concurrent spans overlap. '
                 'Do not sum stages as task duration. The mechanical-arm RPC span includes planning and execution; '
                 'MOTION_PLAN and MOVE are separate worker measurements when present. '
                 'GPU model calls include queueing; preview/console spans include human waiting.')
        from .experiment_timing import phases, recovery_events, write_trial_csv
        report['phases'] = phases(report)
        report['recovery_events'] = recovery_events(report)
        report['phase_schema'] = 'fr3_phase_timing_v1'
        try:
            content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            for path in (history, self.report_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + '.' + self.run_id + '.tmp')
                temporary.write_text(content, encoding='utf-8')
                os.replace(temporary, path)
            write_trial_csv(report, history.with_suffix(".csv"))
        except OSError as error:
            # A diagnostic write must not turn a completed manipulation into a failure.
            print(f'[timing] 报告写入失败: {type(error).__name__}')
        print(f'[timing] 任务总耗时: {duration:.3f}s ({self.status})；嵌套与并行阶段不可直接相加。')
        print(f'[timing] 报告: {self.report_path}；历史: {history}')


@contextmanager
def task_timing(goal, report_path, metadata=None):
    recorder = TimingRecorder(goal, report_path, metadata)
    token, parent_token = _recorder.set(recorder), _parent.set(None)
    try:
        yield recorder
    except BaseException as error:
        recorder.status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'error'
        raise
    finally:
        _parent.reset(parent_token)
        _recorder.reset(token)
        recorder.finish()


def timed_task(function):
    @wraps(function)
    def invoke(self, *args, **kwargs):
        if _recorder.get() is not None:
            return function(self, *args, **kwargs)
        config = self.config
        extra_metadata = experiment_metadata(config)
        report = Path(getattr(config, 'plan_json', 'runs/current_plan/plan_graph.json')).with_name('task_timing.json')
        with task_timing(getattr(config, 'goal', ''), report, dict(execute_robot=config.execute_robot,
                requested_arm_speed_scaling=getattr(config, 'arm_speed_scaling', None),
                policy_backend=getattr(config, 'policy_backend', 'template'),
                policy_pipeline=getattr(config, 'enable_policy_pipeline', True),
                perception_pipeline=getattr(config, 'enable_perception_pipeline', True),
                grasp_planning_retry_attempts=getattr(config, 'grasp_planning_retry_attempts', 1), **extra_metadata)) as recorder:
            result = function(self, *args, **kwargs)
            recorder.status = _result_attributes(result).get('status', 'ok')
            return result
    return invoke


@contextmanager
def timing_stage(name, *, detail='', quiet=False):
    recorder = _recorder.get()
    attributes = {}
    if recorder is None:
        yield attributes
        return
    started = time.perf_counter()
    row = dict(id=recorder.next_id(), parent_id=_parent.get(), name=name, detail=str(detail),
               start_s=started-recorder.started, background=threading.get_ident() != recorder.owner_thread,
               thread=threading.current_thread().name)
    token = _parent.set(row['id'])
    try:
        yield attributes
    except BaseException as error:
        attributes.update(status='error', error_type=type(error).__name__)
        raise
    finally:
        _parent.reset(token)
        row.update(duration_s=time.perf_counter()-started, status=attributes.pop('status', 'ok'),
                   attributes=attributes)
        with recorder._lock:
            recorder.spans.append(row)
        if not quiet:
            label = name + (f' ({detail})' if detail else '')
            suffix = ' [后台]' if row['background'] else ''
            suffix += ' [缓存命中]' if attributes.get('inference_cache_hit') else ''
            print(f'[timing] {label}: {row["duration_s"]:.3f}s [{row["status"]}]{suffix}')


def timed_stage(name, *, detail=None, quiet=False):
    def decorate(function):
        @wraps(function)
        def invoke(*args, **kwargs):
            if _recorder.get() is None:
                return function(*args, **kwargs)
            label = detail(*args, **kwargs) if detail is not None else ''
            with timing_stage(name, detail=label, quiet=quiet) as attributes:
                result = function(*args, **kwargs)
                attributes.update(_result_attributes(result))
                return result
        return invoke
    return decorate


def experiment_metadata(config):
    # Whitelist only experimental conditions; never serialize API credentials.
    fields = ('arm_speed_scaling', 'policy_backend', 'enable_policy_pipeline',
        'enable_perception_pipeline', 'grasp_planning_retry_attempts', 'enable_replan',
        'max_replan_attempts', 'verify_place_vdm', 'verify_grasp_vdm', 'disable_grip',
        'visualize_selected_grasp', 'use_sam_for_grasp', 'enable_geometry_occlusion',
        'use_llm_planner', 'camera_warmup_s', 'grasp_selection_mode', 'grasp_random_seed',
        'grasp_random_select_top_k', 'vdm_model', 'planner_llm_model', 'vlm_model',
        'enable_occlusion_recovery', 'enable_llm_occlusion_recovery')
    source_root = Path(__file__).resolve().parents[1]
    sources = [path for directory in ('agents', 'runtime', 'hardware', 'skill_lib', 'services', 'app', 'core')
               for path in (source_root / directory).rglob('*.py')]
    digest = hashlib.sha256()
    for source in sorted(sources):
        digest.update(str(source.relative_to(source_root)).encode())
        digest.update(source.read_bytes())
    metadata = dict(conditions={k:getattr(config,k,None) for k in fields},
                    implementation_sha256=digest.hexdigest())
    for key in ('experiment', 'trial', 'layout'):
        metadata[key] = os.environ.get('ARMCODE_TIMING_'+key.upper(), '')
    path = getattr(config, 'hardware_config_path', None)
    if path and Path(path).is_file():
        content = Path(path).read_bytes()
        metadata['hardware_sha256'] = hashlib.sha256(content).hexdigest()
        import yaml
        profile = yaml.safe_load(content)
        metadata['configured_stack'] = profile.get('arm', {}).get('stack', 'real')
        metadata['configured_grasp_backend'] = profile.get('perception', {}).get('grasp_backend')
        metadata['hardware_config_path'] = str(path)
    return metadata


def clock_identity():
    # Docker normally shares the host monotonic clock. Never infer cross-namespace alignment.
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip() + ':' + os.readlink('/proc/self/ns/time')
    except OSError:
        return None


def record_worker_timing(result, host_clock):
    recorder = _recorder.get()
    if recorder is None:
        return
    aligned = bool(host_clock and result.get('timing_clock') == host_clock)
    for sample in result.get('timing', []):
        attrs = dict(sample.get('attributes', {}), request_id=result['request_id'],
                     operation=result.get('operation'), stack=result.get('stack'), clock_aligned=aligned,
                     host_log_path=result.get('host_log_path'),
                     worker_start_monotonic_s=sample['start_monotonic_s'])
        if sample.get('phase'):
            attrs['phase'] = sample['phase']
        row = dict(id=recorder.next_id(), parent_id=_parent.get(), name=sample['name'],
                   detail=result['request_id'], start_s=(sample['start_monotonic_s']-recorder.monotonic_started if aligned else None),
                   duration_s=sample['duration_s'], status=sample['status'], attributes=attrs,
                   background=threading.get_ident()!=recorder.owner_thread, thread=threading.current_thread().name)
        with recorder._lock:
            recorder.spans.append(row)


def record_hardware_profile(config):
    recorder = _recorder.get()
    if recorder is not None:
        from dataclasses import asdict
        recorder.metadata['hardware_profile'] = asdict(config)
