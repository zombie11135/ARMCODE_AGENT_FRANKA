"""FR3 branches for the existing mm/rad skill API. Pure geometry, no devices."""
import math
import random
import numpy as np

from .gsnet_single import build_targets, pose_difference
from .interfaces import Pose6D
from .transforms import pose_matrix
from runtime.observation_memory import scene_mutation


def capture_pose(capture, transforms):
    record = capture.get('capture_geometry')
    if not record:
        raise ValueError('FR3 requires capture_geometry from the calibrated wrist camera')
    transforms._check_capture('wrist', record['serial'], record['optical_frame'])
    if record['calibration'] != transforms.summary():
        raise ValueError('Calibration changed since capture; capture a new frame')
    before, after = Pose6D(*record['before']), Pose6D(*record['after'])
    difference = pose_difference(before, after)
    if difference['translation_m'] > .001 or difference['rotation_rad'] > .01:
        raise ValueError('Capture was not stationary')
    return before


def point_to_base(capture, point, transforms):
    pose = capture_pose(capture, transforms)
    record = capture['capture_geometry']
    return transforms.camera_point_to_base(point, 'wrist', pose,
        device_serial=record['serial'], optical_frame=record['optical_frame'])


def mmrad(values):
    return [float(v) * 1000 for v in values[:3]] + [float(v) for v in values[3:]]


class GraspSelectionError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def validate_pose_selection(*, target_rx=math.pi, target_ry=0., max_rx_tilt_deg=65.,
                            max_ry_tilt_deg=65., lam_rx=1., lam_ry=1.,
                            grasp_selection_mode='best_good', random_select_top_k=5,
                            random_seed=None, fallback_to_raw_when_no_good=False):
    for name, value in [('target_rx', target_rx), ('target_ry', target_ry),
                        ('max_rx_tilt_deg', max_rx_tilt_deg), ('max_ry_tilt_deg', max_ry_tilt_deg),
                        ('lam_rx', lam_rx), ('lam_ry', lam_ry)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
    if not (0 <= max_rx_tilt_deg <= 180 and 0 <= max_ry_tilt_deg <= 180):
        raise ValueError('Tilt limits must be in [0,180] degrees')
    if lam_rx < 0 or lam_ry < 0:
        raise ValueError('Tilt penalties must be nonnegative')
    if grasp_selection_mode not in ('best_good', 'random_top_k_good', 'random_good'):
        raise ValueError(f'Unknown grasp_selection_mode: {grasp_selection_mode}')
    if type(random_select_top_k) is not int or random_select_top_k < 1:
        raise ValueError('random_select_top_k must be a positive integer')
    if random_seed is not None and type(random_seed) is not int:
        raise ValueError('random_seed must be an integer or None')
    if fallback_to_raw_when_no_good:
        raise ValueError('FR3 does not bypass pose constraints when no good candidate remains')


def grasp_plan(grasp_input, result, transforms, workcell, *, target_rx=math.pi, target_ry=0.,
               max_rx_tilt_deg=65., max_ry_tilt_deg=65., lam_rx=1., lam_ry=1.,
               grasp_selection_mode='best_good', random_select_top_k=5, random_seed=None,
               fallback_to_raw_when_no_good=False):
    options = dict(target_rx=target_rx, target_ry=target_ry, max_rx_tilt_deg=max_rx_tilt_deg,
                   max_ry_tilt_deg=max_ry_tilt_deg, lam_rx=lam_rx, lam_ry=lam_ry,
                   grasp_selection_mode=grasp_selection_mode, random_select_top_k=random_select_top_k,
                   random_seed=random_seed, fallback_to_raw_when_no_good=fallback_to_raw_when_no_good)
    validate_pose_selection(**options)
    pose = capture_pose(grasp_input, transforms)
    record = grasp_input['capture_geometry']
    evaluated, good = [], []
    table_check = workcell.get('grasp_table_clearance_check', True)
    for candidate in result['candidate_grasps_camera']:
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = candidate['rotation_matrix'], candidate['translation']
        geometry = build_targets(matrix, pose, transforms, record['serial'], record['optical_frame'], workcell)
        flange = geometry['targets']['grasp']['flange_xyz_rpy']
        # Same wrapped RPY differences as the legacy selector, on the calibrated
        # FR3 flange in fr3_link0. Never flip or overwrite a candidate rotation.
        tilt_rx = abs(math.atan2(math.sin(flange[3] - target_rx), math.cos(flange[3] - target_rx)))
        tilt_ry = abs(math.atan2(math.sin(flange[4] - target_ry), math.cos(flange[4] - target_ry)))
        raw_score = float(candidate['score'])
        if not math.isfinite(raw_score):
            raise ValueError('Candidate score must be finite')
        tilt_ok = tilt_rx <= math.radians(max_rx_tilt_deg) + 1e-12 and tilt_ry <= math.radians(max_ry_tilt_deg) + 1e-12
        table_ok = all(t['above_table_margin'] for t in geometry['targets'].values())
        row = dict(candidate_index=candidate['index'], raw_score=raw_score,
                   effective_score=raw_score-lam_rx*tilt_rx-lam_ry*tilt_ry,
                   flange_xyz_rpy=flange, tilt_rx_deg=math.degrees(tilt_rx), tilt_ry_deg=math.degrees(tilt_ry),
                   tilt_rx_rad=tilt_rx, tilt_ry_rad=tilt_ry,
                   within_tilt_limits=bool(tilt_ok), tool_above_table=bool(table_ok),
                   accepted=bool(tilt_ok and (table_ok or not table_check)))
        evaluated.append(row)
        if row['accepted']:
            good.append((row, candidate, geometry, matrix))
    report = dict(options=options, base_frame='fr3_link0', orientation_frame='fr3_link8',
                  table_clearance_check_enabled=table_check,
                  score_formula='raw_score - lam_rx * wrapped_rx_error_rad - lam_ry * wrapped_ry_error_rad',
                  input_candidates=len(evaluated), eligible_candidates=len(good), candidates=evaluated,
                  motion_command_sent=False)
    if not good:
        constraints = 'rx/ry 姿态约束和工具桌面间隙' if table_check else 'rx/ry 姿态约束'
        raise GraspSelectionError(f'FR3 没有候选满足{constraints}；不退化为原始最高分', report)
    good.sort(key=lambda item: item[0]['effective_score'], reverse=True)
    pool = good[:1] if grasp_selection_mode == 'best_good' else (
        good[:random_select_top_k] if grasp_selection_mode == 'random_top_k_good' else good)
    selected_entry = pool[0] if grasp_selection_mode == 'best_good' else random.Random(random_seed).choice(pool)
    chosen_row, chosen, geometry, matrix = selected_entry
    report.update(selected_candidate_index=chosen['index'], chosen_candidate=chosen_row,
                  ranked_candidate_indices=[entry[1]['index'] for entry in good],
                  choose_pool_indices=[entry[1]['index'] for entry in pool], choose_pool_size=len(pool))
    flange = geometry['targets']['grasp']['flange_xyz_rpy']
    pregrasp = geometry['targets']['pregrasp']['flange_xyz_rpy']
    lift = list(flange)
    lift[2] += .05
    selected = dict(rank=1, source_candidate_index=chosen['index'], score=float(chosen['score']),
                    width_m=float(chosen.get('width', 0.)), height_m=float(chosen.get('height', 0.)),
                    depth_m=float(chosen.get('depth', 0.)), translation_m=chosen['translation'],
                    rotation_matrix=chosen['rotation_matrix'], T_input_grasp=matrix.tolist())
    backend = result.get('backend', 'gsnet')
    label = 'AnyGrasp' if backend == 'anygrasp' else 'GSNet'
    reason = (f"{label} 基座系姿态筛选 {len(good)}/{len(evaluated)}；{grasp_selection_mode}，"
              f"候选 {chosen['index']}，原始分 {chosen_row['raw_score']:.4f}，"
              f"综合分 {chosen_row['effective_score']:.4f}，"
              f"rx/ry 偏差 {chosen_row['tilt_rx_deg']:.2f}°/{chosen_row['tilt_ry_deg']:.2f}°")
    return {**result, 'backend': backend, 'base_frame': 'fr3_link0', 'target_frame': 'fr3_link8',
            'num_candidates': 1, 'candidate_grasps_camera': [chosen], 'selected': selected,
            'selection_pending': False, 'pose_selection': report,
            'best_grasp_camera': chosen, 'best_grasp_base_mrad': flange,
            'best_grasp_xarm_mmrad': mmrad(flange), 'approach_pose_mmrad': mmrad(pregrasp),
            'lift_pose_mmrad': mmrad(lift), 'lift_distance_m': .05,
            'current_ee_pose_mrad': pose.values, 'geometry': geometry,
            'grasp_input': grasp_input, 'choose_reason': reason}


@scene_mutation('grasp')
def execute_grasp(context, plan, gripper_open, gripper_close):
    """Approach -> grasp -> completed close -> lift; contact is telemetry only."""
    arm, gripper = context.require('arm'), context.require('gripper')
    if gripper_open is None or gripper_close is None:
        raise RuntimeError('FR3 grasp requires Robotiq open/close callbacks; disable_grip cannot grasp')
    pregrasp = plan['approach_pose_mmrad']
    arm.plan_pose(Pose6D(*[v / 1000 for v in pregrasp[:3]], *pregrasp[3:]))
    if not gripper_open():
        from .robotiq import GripperError
        raise GripperError('Robotiq open failed')
    steps = []
    for name, key in [('move_pregrasp', 'approach_pose_mmrad'), ('move_grasp', 'best_grasp_xarm_mmrad')]:
        values = plan[key]
        pose = Pose6D(*[v / 1000 for v in values[:3]], *values[3:])
        steps.append({'stage': name, 'result': arm.move_pose(pose)})
    if not gripper_close():
        from .robotiq import GripperError
        raise GripperError('Robotiq 闭合未正常完成；不抬升、不自动重试')
    closed_state = gripper.assert_healthy(require_ready=True)
    actual = arm.get_pose().values
    actual[2] += plan.get('lift_distance_m', .05)
    previous_guard = arm.motion_guard
    arm.motion_guard = lambda: gripper.assert_healthy(require_ready=True)
    try:
        steps.append({'stage': 'lift', 'result': arm.move_pose(Pose6D(*actual))})
    finally:
        arm.motion_guard = previous_guard
    return {'success': True, 'steps': steps, 'gripper_close_completed': True,
            'contact_required': False, 'contact_detected': bool(closed_state.object_detected_closing)}
