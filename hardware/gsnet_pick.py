"""Fresh wrist capture -> one GSNet grasp -> FR3 approach -> close -> lift.

Default is capture/inference/pregrasp planning only. --execute enables the whole
physical sequence, including gripper activation if necessary. The existing
MoveIt PTP interface is used for every arm stage; straight paths are not implied.
"""
import argparse
from dataclasses import replace
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import sys
import subprocess
import time

from .config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from .fr3 import FrankaMoveItArm
from .gripper_console import dispatch
from .gsnet_single import capture, infer, targets, validate_roi, envelope_min_z, write_json
from .interfaces import Pose6D
from .robotiq import GripperError, Robotiq2F85


def contact_guard(gripper):
    state = gripper.assert_healthy(require_ready=True)
    if not state.object_detected_closing:
        raise GripperError('No closing contact detected; lift/transport is stopped')
    return state


def preview_o3d(root, continuation):
    # Keep the GUI event loop out of the process maintaining serial keepalive.
    subprocess.run([sys.executable, '-m', 'hardware.gsnet_o3d', str(root),
                    '--continuation', continuation], check=True)
    return {'viewer': 'Open3D', 'run_dir': str(root), 'hardware_accessed_by_viewer': False}


def hold_console(gripper, emit, *, input_fn=input):
    print('夹爪通信保持中。status：查看；open：张开释放；stop：停止夹爪；quit：退出。', flush=True)
    print('退出不会自动张开，但会停止通信轮询。', flush=True)
    while True:
        try:
            words = input_fn('pick> ').strip().split()
        except KeyboardInterrupt:
            break
        except EOFError:
            # A pipe ending must not silently terminate keepalive while holding.
            print('标准输入已关闭，继续保持通信；Ctrl+C 退出。', flush=True)
            try:
                while True:
                    gripper.assert_healthy()
                    time.sleep(.2)
            except KeyboardInterrupt:
                break
        else:
            if words == ['quit']:
                break
            if not words:
                continue
            if len(words) != 1 or words[0] not in ('status', 'open', 'stop'):
                print('输入 status / open / stop / quit。', flush=True)
                continue
            try:
                state = dispatch(gripper, words)
                emit('manual_gripper', command=words[0], status=state.to_dict())
                print(json.dumps(state.to_dict(), ensure_ascii=False), flush=True)
            except (Exception, KeyboardInterrupt) as error:
                emit('manual_gripper_failed', error=repr(error))
                print(f'夹爪命令停止：{error!r}', flush=True)


def run(root, cfg, gsnet_root, *, roi=None, execute=False, lift_m=.05,
        hold=True, input_fn=input, visualize=False):
    """One attempt, with fresh capture and durable per-stage logs; never retry."""
    if cfg.arm.stack != 'real':
        raise ValueError('Fresh wrist capture requires the real robot pose stack')
    if not math.isfinite(lift_m) or lift_m < 0:
        raise ValueError('lift_m must be finite and nonnegative, in metres')
    if execute and not cfg.gripper.port:
        raise ValueError('Configure the Robotiq serial port before execution')
    gsnet_root = Path(gsnet_root).expanduser().resolve()
    for relative in ('realsense_capture.py', 'checkpoints/graspness_realsense.tar'):
        if not (gsnet_root / relative).is_file():
            raise FileNotFoundError(gsnet_root / relative)
    roi = list(roi) if roi is not None else [0, 0, cfg.cameras.width, cfg.cameras.height]
    validate_roi(roi, {'color_intrinsics': {'width': cfg.cameras.width, 'height': cfg.cameras.height}})
    # Request-local override. Ranked screening and other clients keep their
    # existing endpoint policy; no global configuration is edited.
    cfg = replace(cfg, arm=replace(cfg.arm, singularity_screening='advisory'))
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    observation = root / 'observation'
    begun = time.monotonic()
    report = {'schema_version': 1, 'ok': False, 'run_dir': str(root),
              'execute': execute, 'roi_xyxy': roi, 'lift_m': lift_m,
              'o3d_enabled': visualize,
              'singularity_screening': 'advisory', 'jaw_symmetry_flipped': False,
              'arm_motion_attempted': False, 'gripper_actuation_attempted': False,
              'contact_detected': False, 'lift_completed': False,
              'grasp_success_visually_verified': False, 'steps': []}
    arm, gripper, session_started = None, None, False
    print(f'RUN_DIR={root}', flush=True)
    print('执行模式：拍摄→推理→张开→预抓取→抓取→闭合→接触后抬升。' if execute else
          '仅拍摄、推理和预抓取规划；不激活或移动机械臂/夹爪。', flush=True)
    with (root / 'events.jsonl').open('x') as log:
        def emit(event, **data):
            log.write(json.dumps({'time': datetime.now().astimezone().isoformat(),
                                  'event': event, **data}, ensure_ascii=False, allow_nan=False) + '\n')
            log.flush()

        def step(name, operation):
            report['stage'] = name
            write_json(root / 'run.json', report)
            print(f'[{name}]', flush=True)
            started = time.monotonic()
            emit('stage_started', stage=name)
            value = operation()
            result = value.to_dict() if hasattr(value, 'to_dict') else (
                value.values if isinstance(value, Pose6D) else value)
            entry = {'stage': name, 'elapsed_s': time.monotonic() - started, 'result': result}
            report['steps'].append(entry)
            emit('stage_completed', **entry)
            write_json(root / 'run.json', report)
            return value

        def move_arm(name, pose):
            report['arm_motion_attempted'] = True
            result = step(name, lambda: arm.move_pose(pose))
            # Host adapter normally raises on these; keep the workflow boundary
            # explicit so a failed stage can never proceed to closing/lifting.
            if not result.get('ok') or not result.get('cleanup_ok'):
                raise RuntimeError(f'{name}: arm movement/cleanup did not succeed')
            return result

        try:
            if execute:
                gripper = Robotiq2F85(cfg.gripper, execution_enabled=True)
                step('gripper_communication', gripper.start_session)
                session_started = True
            step('capture', lambda: capture(observation, cfg, gsnet_root))
            step('gsnet_inference', lambda: infer(observation, cfg, gsnet_root, roi))
            geometry = step('transform', lambda: targets(observation, cfg))
            report['selected'] = geometry['selected']
            if visualize:
                step('o3d_before_motion', lambda: preview_o3d(root, 'execute' if execute else 'plan'))
            for name in ('pregrasp', 'grasp'):
                if (geometry['workcell'].get('grasp_table_clearance_check', True)
                        and not geometry['targets'][name]['above_table_margin']):
                    raise RuntimeError(f'{name}: tool envelope below the table clearance')
            pregrasp = Pose6D(*geometry['targets']['pregrasp']['flange_xyz_rpy'])
            grasp = Pose6D(*geometry['targets']['grasp']['flange_xyz_rpy'])
            print('抓取法兰目标 [x,y,z,rx,ry,rz]，m/rad：' + json.dumps(grasp.values), flush=True)
            arm = FrankaMoveItArm(cfg.arm, execution_enabled=execute)
            # Plan before any gripper actuation; move_pose replans from fresh
            # feedback when executing each stage, as in the tested interface.
            step('plan_pregrasp', lambda: arm.plan_pose(pregrasp))
            if execute:
                state = gripper.assert_healthy()
                report['gripper_actuation_attempted'] = True
                if not state.ready:
                    if state.activation_state == 1:
                        raise GripperError('Gripper activation already in progress')
                    step('activate_gripper', gripper.activate)
                opened = step('open_gripper', gripper.open)
                if not opened.ready or opened.object_state != 3:
                    raise GripperError('Gripper did not reach its open position')
                arm.motion_guard = lambda: gripper.assert_healthy(require_ready=True)
                move_arm('move_pregrasp', pregrasp)
                move_arm('move_grasp', grasp)
                gripper.assert_healthy(require_ready=True)
                closed = step('close_gripper', gripper.close)
                report['contact_detected'] = closed.object_detected_closing
                if not closed.object_detected_closing:
                    raise GripperError('闭合完成但未检测到夹持接触；停止，不抬升、不自动重抓')
                if lift_m > 0:
                    arm.motion_guard = lambda: contact_guard(gripper)
                    contact_guard(gripper)
                    actual = step('read_pose_before_lift', arm.get_pose)
                    lifted = Pose6D(actual.x, actual.y, actual.z + lift_m,
                                    actual.rx, actual.ry, actual.rz)
                    if (geometry['workcell'].get('grasp_table_clearance_check', True)
                            and envelope_min_z(lifted, geometry['workcell']) < (
                                geometry['workcell']['table_z_m'] + geometry['workcell']['table_clearance_m'])):
                        raise RuntimeError('Lift target violates table clearance')
                    report['lift_target_xyz_rpy'] = lifted.values
                    move_arm('lift', lifted)
                    contact_guard(gripper)
                    report['lift_completed'] = True
                report['outcome'] = 'contact_and_lift_completed' if lift_m else 'contact_detected'
            else:
                report['outcome'] = 'capture_inference_and_pregrasp_plan_only'
            report['ok'] = True
            report['stage'] = 'completed'
            print('流程完成。' if execute else '预抓取规划完成；没有发送动作。', flush=True)
            if visualize and execute:
                write_json(root / 'run.json', report)
                try:
                    preview_o3d(root, 'hold' if hold else 'done')
                except (Exception, KeyboardInterrupt) as error:
                    report['o3d_result_error'] = repr(error)
                    emit('o3d_result_failed', error=repr(error))
                    print(f'动作已完成，结果可视化未完成：{error!r}', flush=True)
        except (Exception, KeyboardInterrupt) as error:
            report['error'] = repr(error)
            if arm is not None:
                report['last_arm_result'] = arm.last_result
            emit('workflow_failed', stage=report.get('stage'), error=repr(error))
            print(f'流程停止于 {report.get("stage")}：{error!r}；不自动恢复或重试。', flush=True)
        finally:
            report['sequence_elapsed_s'] = time.monotonic() - begun
            write_json(root / 'run.json', report)
            try:
                if hold and session_started:
                    hold_console(gripper, emit, input_fn=input_fn)
            except (Exception, KeyboardInterrupt) as error:
                report['hold_error'] = repr(error)
                report['ok'] = False
            finally:
                for resource, cleanup in ((arm, 'close'), (gripper, 'disconnect')):
                    if resource is not None:
                        try:
                            getattr(resource, cleanup)()
                        except Exception as error:
                            report.setdefault('cleanup_errors', []).append(repr(error))
                            report['ok'] = False
                if gripper is not None:
                    report['gripper_session'] = gripper.session_info()
                    print('夹爪通信会话已关闭；没有自动张开。', flush=True)
                emit('finished', ok=report['ok'])
                write_json(root / 'run.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--gsnet-root', type=Path, default=Path(sys.prefix) / 'src/gsnet')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--roi', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='Target pixel ROI; default full color image, not the old capture ROI')
    parser.add_argument('--lift', type=float, default=.05, metavar='METRES',
                        help='Lift goal along base +Z after contact; default .05 m; 0 stops after closing')
    parser.add_argument('--execute', action='store_true',
                        help='Enable arm motion and gripper activation/open/close')
    parser.add_argument('--no-hold', action='store_true',
                        help='Exit after sequence instead of maintaining the gripper communication console')
    parser.add_argument('--o3d', action='store_true',
                        help='Show Open3D before motion; closing resumes the selected mode; show feedback after execution')
    args = parser.parse_args(argv)
    if not math.isfinite(args.lift) or args.lift < 0:
        parser.error('--lift must be finite and nonnegative (metres)')
    root = args.output_dir or PROJECT_ROOT / 'runs/gsnet_pick' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    lock_path = PROJECT_ROOT / 'runs/gsnet_pick/.workflow.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run(root, load_config(args.config), args.gsnet_root, roi=args.roi,
                         execute=args.execute, lift_m=args.lift, hold=not args.no_hold, visualize=args.o3d)
        print('结果文件：' + str(Path(result['run_dir']) / 'run.json'), flush=True)
        return 0 if result['ok'] else 1
    except (Exception, KeyboardInterrupt) as error:
        print(f'Stopped: {error!r}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
