"""Wrist-primary, fixed-auxiliary RGB-D experiment, without robot control or fusion.

Each view is interpreted independently in its optical frame. The fixed view is
an unassociated inventory, never a source of confirmed wrist-view occlusion.
All physical actions and cross-view geometric inference remain disabled.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from perception_v2.proposal_refinement import select_visible_hypotheses
from .camera_only import capture
from .real_single_view import save_view


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_candidates(capture_dir, analysis_dir, camera):
    """Reject old/mixed snapshots and depth arrays before reusing model outputs."""
    snapshot = Path(capture_dir)/camera
    directory = Path(analysis_dir)/camera
    packet = _read(directory/'observation/scene_observation.json')
    metadata = _read(snapshot/'grasp_input.json')
    report = _read(Path(capture_dir)/'capture_report.json')
    frame, provenance = packet['frame'], packet['provenance']
    if (packet.get('schema') != 'occlusion_observation_v1'
            or frame['camera_id'] != camera or metadata['camera_id'] != camera
            or frame['capture_group_id'] != report['capture_group_id']
            or metadata['capture_group_id'] != report['capture_group_id']
            or frame['timestamp_s'] != metadata['capture_timestamp_s']
            or frame['timestamp_domain'] != metadata['timestamp_domain']
            or metadata['device_serial'] != report['cameras'][camera]['device_serial']
            or frame['depth_units'] != 'metres' or frame['depth_definition'] != 'optical_Z'
            or metadata['depth_aligned_to_color'] is not True
            or frame['rgb_depth_alignment'] != 'recorded_aligned'):
        raise ValueError('Analysis does not belong to the requested aligned capture')
    intrinsic = metadata['depth_intrinsics']
    expected_k = dict(fx=intrinsic['fx'], fy=intrinsic['fy'], cx=intrinsic['ppx'], cy=intrinsic['ppy'])
    if frame['intrinsics'] != expected_k:
        raise ValueError('Analysis intrinsics differ from captured intrinsics')
    if (provenance['rgb_sha256'] != _sha(snapshot/'color.png')
            or provenance['depth_sha256'] != _sha(snapshot/'depth_m.npy')):
        raise ValueError('Source RGB-D hash changed')
    with np.load(directory/'observation'/packet['artifacts']['arrays'], allow_pickle=False) as stored:
        depth = np.load(snapshot/'depth_m.npy', allow_pickle=False)
        if not np.array_equal(stored['depth_m'], depth, equal_nan=True):
            raise ValueError('Analysis depth differs from original captured depth')
        rows = {p['source_id']: p for p in _read(directory/'segmentation_report.json')['proposals']}
        if len(rows) != len(packet['proposals']) or set(rows) != {p['source_id'] for p in packet['proposals']}:
            raise ValueError('Segmentation proposal IDs differ from observation')
        candidates = [dict(rows[p['source_id']], mask=stored[p['raw_mask_key']].copy())
                      for p in packet['proposals']]
    backend = _read(directory/'segmentation_report.json')
    backend.pop('proposals')
    backend['reused_analysis_dir'] = str(directory.resolve())
    return candidates, backend


def _view_summary(directory, role):
    packet = _read(directory/'observation/scene_observation.json')
    local = _read(directory/'boundary/relations.json')
    planar = _read(directory/'planar_relations.json')
    camera = packet['frame']['camera_id']
    nodes = []
    with np.load(directory/'observation/observation_arrays.npz', allow_pickle=False) as arrays:
        for index, node in enumerate(packet['relation_graph']['nodes']):
            mask = arrays[node['raw_mask_key']]
            issues = []
            if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                issues.append('truncated_at_image_border')
            if node['quality']['depth_supported_pixels'] == 0:
                issues.append('no_depth_support')
            if node['quality']['overlap_with_other_nodes_pixels']/max(1, node['quality']['raw_mask_pixels']) > .05:
                issues.append('overlapping_visible_mask_assignments')
            nodes.append({**node, 'display_id': f'N{index}', 'camera_id': camera,
                          'role': role, 'quality_issues': issues})
        by_id = {n['node_id']: n for n in nodes}
        edges, withheld = [], []
        for edge in [*local['relation_graph']['edges'], *planar['edges']]:
            if edge['source'] not in by_id or edge['target'] not in by_id:
                raise ValueError('Relation points outside its own view')
            a, b = (arrays[by_id[edge[key]]['raw_mask_key']] for key in ('source', 'target'))
            overlap = np.count_nonzero(a & b)/max(1, min(a.sum(), b.sum()))
            edge = {**edge, 'camera_id': camera, 'role': role, 'confirmed': False,
                    'calibrated_probability': None,
                    'source_display_id': by_id[edge['source']]['display_id'],
                    'target_display_id': by_id[edge['target']]['display_id'],
                    'review_required': True}
            if overlap > .05:
                withheld.append({**edge, 'reason': 'overlapping_visible_mask_assignments'})
            else:
                edges.append(edge)
    pairs = {}
    for edge in edges:
        if edge['relation'] != 'possible_occludes':
            continue
        pair = (edge['source'], edge['target'])
        if pair not in pairs:
            pairs[pair] = {k: edge[k] for k in ('source', 'target', 'source_display_id',
                                               'target_display_id', 'camera_id', 'role')}
            pairs[pair].update(confirmed=False, evidence=[], calibrated_probability=None)
        pairs[pair]['evidence'].append(edge)
    for (source, target), pair in pairs.items():
        pair['opposite_direction_hypothesis_present'] = (target, source) in pairs
    status = ('no_instances_detected_unknown' if not nodes else
              'possible_occlusion_requires_review' if pairs else 'no_occlusion_evidence_not_clear_scene')
    selection = _read(directory/'segmentation_report.json')['visible_hypothesis_selection']
    return dict(camera_id=camera, role=role, frame=packet['frame'], status=status,
                instance_hypotheses=len(nodes), relation_graph=dict(nodes=nodes, edges=edges),
                possible_pairs=list(pairs.values()), withheld_evidence=withheld,
                boundary_pair_statuses=dict(Counter(p['status'] for p in local['pairs'])),
                segmentation_selection=selection,
                artifacts=dict(evidence=str((directory/'evidence.png').resolve()),
                               observation=str((directory/'observation/scene_observation.json').resolve())),
                target_identity_status='not_evaluated', fully_hidden_objects_observable=False)


def build_report(capture_dir, analysis_dir, output_dir, *, wrist_only=False):
    capture_dir, analysis_dir, output_dir = map(Path, (capture_dir, analysis_dir, output_dir))
    capture_report = _read(capture_dir/'capture_report.json')
    if capture_report.get('status') == 'capture_failed' or 'wrist' not in capture_report['cameras']:
        raise ValueError('A completed wrist capture is required')
    cameras = ['wrist']
    if not wrist_only and 'global' in capture_report['cameras']:
        cameras.append('global')
    # Validate all inputs before creating a result that could look complete.
    inputs = {camera: load_candidates(capture_dir, analysis_dir, camera) for camera in cameras}
    output_dir.mkdir(parents=True, exist_ok=False)
    views, panels = {}, []
    for camera in cameras:
        candidates, backend = inputs[camera]
        selected, selection = select_visible_hypotheses(candidates)
        backend['visible_hypothesis_selection'] = selection
        directory = output_dir/camera
        save_view(capture_dir/camera, selected, backend, directory, raw_proposals=candidates)
        role = 'primary' if camera == 'wrist' else 'auxiliary'
        views[camera] = _view_summary(directory, role)
        panel = cv2.imread(str(directory/'evidence.png'))
        panel = cv2.copyMakeBorder(panel, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, f'{role.upper()} | {camera} | independent local evidence', (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 0, 0, width-p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    if not cv2.imwrite(str(output_dir/'overview.png'), np.vstack(panels)):
        raise IOError('Failed to save primary/auxiliary overview')
    primary = views['wrist']
    result = dict(schema='wrist_primary_fixed_aux_occlusion_v1', mode='independent_view_evidence',
        capture_group_id=capture_report['capture_group_id'], observation_epoch=capture_report['capture_group_id'],
        primary_camera='wrist', auxiliary_camera='global' if 'global' in views else None,
        status=primary['status'], views=views, pairing=capture_report['pairing'],
        cross_view_geometry_used=False, cross_view_identity_association_performed=False,
        robot_control_initialized=False, robot_motion_commanded=False, robot_state_queried=False,
        capture_robot_state_queried=capture_report['robot_state_queried'],
        accuracy_evaluated=False, human_ground_truth_available=False,
        source=dict(capture_dir=str(capture_dir.resolve()), analysis_dir=str(analysis_dir.resolve()),
                    capture_report_sha256=_sha(capture_dir/'capture_report.json')),
        limitations=['predicted_instances_are_not_physical_object_counts',
                     'auxiliary_instances_not_automatically_matched_to_wrist_instances',
                     'fully_hidden_or_missed_objects_remain_unknown',
                     'no_candidate_edges_does_not_mean_no_occlusion',
                     'convex_and_planar_completion_are_unverified_priors',
                     'single_capture_is_not_an_accuracy_evaluation'])
    _write(output_dir/'report.json', result)
    _write(output_dir/'agent_observation.json', dict(
        schema='occlusion_agent_review_input_v1', observation_epoch=result['observation_epoch'],
        validity='this_capture_only_invalidate_after_any_scene_or_camera_change',
        primary_relation_graph=primary['relation_graph'],
        primary_observation_file=primary['artifacts']['observation'],
        primary_evidence_file=primary['artifacts']['evidence'],
        auxiliary_view=views.get('global'), cross_view_associations=[],
        target_identity_status='not_evaluated', occlusion_status=primary['status'],
        suggested_next_step='review_primary_candidates' if primary['possible_pairs'] else 'review_detection_and_visibility',
        execution_enabled=False, agent_runtime_connected=False,
        report=str((output_dir/'report.json').resolve())))
    lines = ['腕部主、固定辅助：真机静止观测实验', '',
             '本次仅采集相机并离线分析；不调用机械臂或夹爪控制。',
             '固定视角独立保留，不进行跨视角身份合并、点云融合或遮挡确认。',
             '节点是分割假设，N 编号仅在各自画面内有效；多条平面证据不重复计数。', '']
    for camera, view in views.items():
        lines.append(f"{camera} ({view['role']}): {view['instance_hypotheses']} 个选中分割假设，"
                     f"{len(view['possible_pairs'])} 对潜在遮挡；"
                     f"保留 {len(view['segmentation_selection']['alternatives'])} 个冲突备选供复查。")
        for pair in view['possible_pairs']:
            gaps = [e['depth_gap_median_m']*1000 for e in pair['evidence'] if 'depth_gap_median_m' in e]
            lines.append(f"  {pair['source_display_id']} → {pair['target_display_id']}: 潜在遮挡，"
                         f"平面延伸深度差中位数(mm)={gaps}；未经人工真值验证。")
        lines.append(f"  状态: {view['status']}")
    lines += ['', '零候选不等于无遮挡；完全不可见物体的位置和身份仍未知。',
              '当前只导出 agent_observation.json 供审阅，尚未接入 agent 自动执行。']
    (output_dir/'review.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def run_experiment(*, capture_dir=None, analysis_dir=None, output_dir=None,
                   config='config/hardware/fr3.yaml', checkpoint='/home/seur/data/wpy/models/sam3/sam3.pt',
                   device='cuda', wrist_only=False, max_skew_s=.10):
    if analysis_dir is not None and capture_dir is None:
        raise ValueError('--analysis-dir requires --capture-dir')
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(output_dir)
    root = (Path(capture_dir) if capture_dir is not None else
            capture(config, camera_mode='wrist' if wrist_only else 'dual', max_skew_s=max_skew_s))
    output = Path(output_dir) if output_dir is not None else root/'primary_aux'
    if output.exists():
        raise FileExistsError(output)
    captured = _read(root/'capture_report.json')
    if captured.get('status') == 'capture_failed' or 'wrist' not in captured['cameras']:
        raise ValueError('A completed wrist capture is required')
    if analysis_dir is None:
        analysis_dir = root/'primary_aux_analysis'
        cameras = ['wrist']
        if not wrist_only and 'global' in captured['cameras']:
            cameras.append('global')
        subprocess.run([sys.executable, '-u', '-m', 'occlusion_reasoning.real_single_view',
                        '--capture-dir', str(root), '--output-dir', str(analysis_dir),
                        '--checkpoint', str(checkpoint), '--device', device,
                        '--cameras', *cameras], check=True)
    result = build_report(root, analysis_dir, output, wrist_only=wrist_only)
    print(json.dumps(dict(output_dir=str(output.resolve()), status=result['status'],
        primary_possible_pairs=len(result['views']['wrist']['possible_pairs']),
        auxiliary_possible_pairs=len(result['views'].get('global', {}).get('possible_pairs', [])),
        robot_motion_commanded=False, cross_view_geometry_used=False)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, help='Reuse saved capture; otherwise capture live cameras')
    parser.add_argument('--analysis-dir', type=Path, help='Reuse saved SAM3 predictions for the same capture')
    parser.add_argument('--output-dir', type=Path, help='New result directory, default CAPTURE/primary_aux')
    parser.add_argument('--config', default='config/hardware/fr3.yaml')
    parser.add_argument('--checkpoint', default='/home/seur/data/wpy/models/sam3/sam3.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--wrist-only', action='store_true', help='Explicitly disable fixed auxiliary view')
    parser.add_argument('--max-skew-s', type=float, default=.10)
    run_experiment(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
