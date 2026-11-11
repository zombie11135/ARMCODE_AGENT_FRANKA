"""Staged FR3 placement. Heights refer to fr3_link8, never to the TCP.

A lower release needs measured support height and the held object's downward
extent at the RELEASE orientation. This is position control, not force control.
"""
import math
import time

from hardware.interfaces import Pose6D
from hardware.robotiq import GripperError


def release_height(config, original_z_mm):
    support = config.place_support_z_m
    offset = config.place_object_bottom_offset_m
    if support is None and offset is None:
        return float(original_z_mm), 'configured_flange_height'
    if support is None or offset is None:
        raise ValueError('Lower release requires BOTH support Z and measured flange-to-object-bottom offset')
    if not all(math.isfinite(v) for v in (support, offset, config.place_release_gap_m)):
        raise ValueError('Non-finite release geometry')
    return 1000 * (support + offset + config.place_release_gap_m), 'measured_object_bottom'


def prepare_place_plan(plan, config):
    updated = dict(plan)
    release = list(plan['place_pose_mmrad'])
    if config.place_height_mode == 'stack_rgbd':
        support = plan.get('stack_support')
        if not support or config.place_object_bottom_offset_m is None:
            raise ValueError('Stack placement requires a fresh RGB-D support estimate and measured object bottom offset')
        center = support.get('center_xy_m')
        if center is not None and any(abs(release[i]/1000 - center[i]) > config.position_tolerance_m for i in (0, 1)):
            raise ValueError('Placement slot differs from observed support patch; reobserve that slot')
        release[2] = 1000 * (support['support_z_m'] + config.place_object_bottom_offset_m
                             + config.place_release_gap_m)
        mode = 'stack_rgbd'
    else:
        release[2], mode = release_height(config, release[2])
    approach = list(plan['approach_pose_mmrad'])
    retreat = list(plan['lift_pose_mmrad'])
    # Keep the established transit height when lowering the release point.
    approach[2] = max(approach[2], release[2] + config.place_final_descent_m * 1000)
    retreat[2] = max(retreat[2], approach[2])
    final_approach = list(release)
    final_approach[2] += config.place_final_descent_m * 1000
    updated.update(place_pose_mmrad=release, approach_pose_mmrad=approach,
                   pre_release_pose_mmrad=final_approach, lift_pose_mmrad=retreat,
                   release_height_mode=mode, place_height_mode=mode,
                   gentle_place=True, place_flange_z_mm=release[2])
    return updated


def execute_place(context, plan, gripper_open):
    arm, gripper = context.require('arm'), context.require('gripper')
    if gripper_open is None:
        raise GripperError('FR3 placement requires a working gripper release callback')
    cfg = arm.config
    original_speed = cfg.speed_scaling
    steps = []
    progress = dict(operation='place', stage='precheck', released=False,
                    holding_state='possibly_holding', motion_started=False)
    context.manipulation_progress = progress
    if plan['release_height_mode'] in ('stack_rgbd', 'measured_object_bottom'):
        import json
        from pathlib import Path
        from hardware.gsnet_single import envelope_min_z
        from hardware.fr3 import MotionError
        values = plan['place_pose_mmrad']
        release_pose = Pose6D(*[v / 1000 for v in values[:3]], *values[3:])
        workcell = json.loads((Path(cfg.workspace) / 'config/workcell.json').read_text())
        support_z = (plan['stack_support']['support_z_m'] if plan['release_height_mode'] == 'stack_rgbd'
                     else cfg.place_support_z_m)
        if envelope_min_z(release_pose, workcell) <= support_z + cfg.position_tolerance_m:
            raise MotionError(dict(error='Tool envelope reaches the estimated support surface; do not lower/release'))
    poses = [(label, Pose6D(*[v / 1000 for v in plan[key][:3]], *plan[key][3:]))
             for label, key in [('approach', 'approach_pose_mmrad'),
                                ('pre_release', 'pre_release_pose_mmrad'),
                                ('release', 'place_pose_mmrad'),
                                ('retreat', 'lift_pose_mmrad')]]
    # Endpoint feasibility only; each move still plans from the actual state.
    for label, pose in poses:
        progress['stage'] = 'precheck_' + label
        arm.plan_pose(pose)
    try:
        for label, pose in poses[:3]:
            progress['stage'] = label
            gripper.assert_healthy(require_ready=True)
            arm.set_speed_scaling(original_speed if label == 'approach' else
                                  min(original_speed, cfg.place_descent_speed_scaling))
            progress['motion_started'] = True
            steps.append(dict(stage=label, result=arm.move_pose(pose)))
        progress['stage'] = 'settle'
        time.sleep(cfg.place_settle_s)
        gripper.assert_healthy(require_ready=True)
        progress['stage'] = 'open_gripper'
        # An interrupted opening is ambiguous: never infer that the object stayed held.
        progress['holding_state'] = 'unknown'
        if not gripper_open():
            raise GripperError('Placement release did not complete; stop without automatic retreat')
        progress.update(released=True, holding_state='empty')
        time.sleep(cfg.place_settle_s)
        progress['stage'] = 'retreat'
        arm.set_speed_scaling(min(original_speed, cfg.place_retreat_speed_scaling))
        gripper.assert_healthy(require_ready=True)
        steps.append(dict(stage='retreat', result=arm.move_pose(poses[-1][1])))
        progress['stage'] = 'complete'
        return dict(success=True, released=True, steps=steps,
                    release_height_mode=plan['release_height_mode'])
    finally:
        arm.set_speed_scaling(original_speed)
