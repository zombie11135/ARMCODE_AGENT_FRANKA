"""Independent-view contracts and inference isolation, with no devices/models."""

import copy
import json
import time

import numpy as np
import pytest

from occlusion_reasoning.camera_only import save_pair
from occlusion_reasoning.real_single_view import save_view
from occlusion_reasoning import real_primary_aux as module
from perception_v2.proposal_refinement import select_visible_hypotheses


def proposals():
    back = np.zeros((60, 80), bool)
    back[10:50, 10:60] = True
    front = np.zeros_like(back)
    front[20:55, 25:45] = True
    back &= ~front
    fragment = np.zeros_like(back)
    fragment[10:20, 10:30] = True
    return [dict(source_id=name, mask=mask, confidence=score, bbox_xyxy=[10, 10, 60, 55], prompt='box')
            for name, mask, score in [('back', back, .95), ('front', front, .9), ('fragment', fragment, .6)]]


def fixture(tmp_path, *, cameras=('global', 'wrist'), empty=False):
    root = tmp_path/'capture'
    root.mkdir(parents=True)
    now = time.time()
    intrinsic = dict(width=80, height=60, fx=100., fy=100., ppx=40., ppy=30., coeffs=[0.]*5)
    rows = [] if empty else proposals()
    depth = np.full((60, 80), 600, np.uint16)
    if rows:
        depth[rows[0]['mask']] = 500
        depth[rows[1]['mask']] = 400
    pair = {camera: dict(color=np.full((60, 80, 3), 100, np.uint8), depth=depth,
                timestamp=now, color_intrinsics=intrinsic, depth_intrinsics=intrinsic,
                depth_aligned_to_color=True, depth_unit_m_per_raw_unit=.001,
                device_serial=camera+'serial') for camera in cameras}
    save_pair(pair, root, 0., .10)
    analysis = root/'analysis'
    for camera in cameras:
        save_view(root/camera, copy.deepcopy(rows), {'backend': 'synthetic_predictions'}, analysis/camera)
    return root, analysis


def test_visible_selection_removes_contained_alternative_without_changing_geometry():
    rows = proposals()
    original = copy.deepcopy(rows)
    selected, report = select_visible_hypotheses(rows)
    assert [p['source_id'] for p in selected] == ['back', 'front']
    assert report['alternatives'][0]['source_id'] == 'fragment'
    assert report['alternatives'][0]['competes_with'] == 'back'
    for after, before in zip(rows, original):
        np.testing.assert_array_equal(after['mask'], before['mask'])
        assert after['confidence'] == before['confidence']
    # A second real visible object may lie inside the first object's bounding
    # rectangle while having disjoint visible pixels: keep both, irrespective of boxes.
    assert np.count_nonzero(selected[0]['mask'] & selected[1]['mask']) == 0


def test_report_keeps_auxiliary_nodes_and_edges_out_of_primary_graph(tmp_path):
    root, analysis = fixture(tmp_path)
    report = module.build_report(root, analysis, root/'result')
    wrist = report['views']['wrist']
    assert wrist['instance_hypotheses'] == 2
    assert len(wrist['possible_pairs']) == 1
    assert len(wrist['possible_pairs'][0]['evidence']) >= 1
    assert not report['cross_view_geometry_used']
    assert not report['cross_view_identity_association_performed']
    assert not report['robot_state_queried']
    assert not report['accuracy_evaluated']
    packet = json.loads((root/'result/agent_observation.json').read_text())
    assert packet['cross_view_associations'] == []
    assert not packet['execution_enabled'] and not packet['agent_runtime_connected']
    assert all('/wrist/' in node['node_id'] for node in packet['primary_relation_graph']['nodes'])
    assert all('/global/' in node['node_id'] for node in packet['auxiliary_view']['relation_graph']['nodes'])
    assert all(not edge['confirmed'] for edge in packet['primary_relation_graph']['edges'])
    assert (root/'result/overview.png').is_file()
    assert (root/'result/wrist/raw_proposal_masks.npz').is_file()


def test_auxiliary_changes_cannot_change_primary_conclusion(tmp_path):
    root, analysis = fixture(tmp_path)
    first = module.build_report(root, analysis, root/'first')
    # An entirely different segmentation of the same auxiliary image.
    other = root/'other_analysis'
    save_view(root/'global', [], {'backend': 'empty_auxiliary'}, other/'global')
    from shutil import copytree
    copytree(analysis/'wrist', other/'wrist')
    second = module.build_report(root, other, root/'second')
    assert first['views']['wrist']['relation_graph'] == second['views']['wrist']['relation_graph']
    assert first['status'] == second['status']
    assert second['views']['global']['instance_hypotheses'] == 0


def test_empty_masks_leave_unknown_without_inventing_hidden_object(tmp_path):
    root, analysis = fixture(tmp_path, cameras=('wrist',), empty=True)
    report = module.build_report(root, analysis, root/'result')
    assert report['status'] == 'no_instances_detected_unknown'
    assert report['auxiliary_camera'] is None
    assert report['views']['wrist']['relation_graph'] == dict(nodes=[], edges=[])
    assert not report['views']['wrist']['fully_hidden_objects_observable']


def test_zero_edges_do_not_claim_clear_scene(tmp_path):
    root, _ = fixture(tmp_path, cameras=('wrist',))
    analysis = root/'single_object'
    save_view(root/'wrist', [proposals()[0]], {'backend': 'one'}, analysis/'wrist')
    report = module.build_report(root, analysis, root/'result')
    assert report['status'] == 'no_occlusion_evidence_not_clear_scene'


@pytest.mark.parametrize('tamper', ['group', 'image', 'depth', 'intrinsics'])
def test_mixed_or_changed_capture_is_rejected_before_export(tmp_path, tamper):
    root, analysis = fixture(tmp_path)
    path = analysis/'wrist/observation/scene_observation.json'
    packet = json.loads(path.read_text())
    if tamper == 'group':
        packet['frame']['capture_group_id'] = 'old_observation'
    elif tamper == 'intrinsics':
        packet['frame']['intrinsics']['fx'] += 1
    elif tamper == 'image':
        (root/'wrist/color.png').write_bytes(b'changed image')
    else:
        array_path = path.parent/'observation_arrays.npz'
        with np.load(array_path) as original:
            arrays = dict(original)
        arrays['depth_m'] = arrays['depth_m']+.01
        np.savez_compressed(array_path, **arrays)
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        module.build_report(root, analysis, root/'result')
    assert not (root/'result').exists()


def test_explicit_wrist_only_ignores_unavailable_auxiliary_analysis(tmp_path):
    root, analysis = fixture(tmp_path)
    from shutil import rmtree
    rmtree(analysis/'global')
    report = module.build_report(root, analysis, root/'result', wrist_only=True)
    assert report['auxiliary_camera'] is None
    assert set(report['views']) == {'wrist'}


def test_live_orchestration_closes_capture_before_model_and_never_runs_fusion(tmp_path, monkeypatch):
    root, analysis = fixture(tmp_path)
    calls = []
    def capture(*args, **kwargs):
        assert kwargs['camera_mode'] == 'dual'
        calls.append('capture_closed')
        return root
    def run(argv, check):
        assert calls == ['capture_closed']
        assert check and 'occlusion_reasoning.real_single_view' in argv
        assert argv[-3:] == ['--cameras', 'wrist', 'global']
        from shutil import copytree
        copytree(analysis, root/'primary_aux_analysis')
        calls.append('model_finished')
    monkeypatch.setattr(module, 'capture', capture)
    monkeypatch.setattr(module.subprocess, 'run', run)
    report = module.run_experiment(output_dir=root/'result')
    assert calls == ['capture_closed', 'model_finished']
    assert not report['robot_motion_commanded']


def test_failed_capture_never_runs_model_or_publishes_report(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('device unavailable')
    monkeypatch.setattr(module, 'capture', fail)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: pytest.fail('model must not run'))
    with pytest.raises(RuntimeError, match='device unavailable'):
        module.run_experiment(output_dir=tmp_path/'result')
    assert not (tmp_path/'result').exists()


def test_new_entry_can_reanalyze_a_wrist_only_capture_without_requesting_global(tmp_path, monkeypatch):
    root, analysis = fixture(tmp_path, cameras=('wrist',))
    def run(argv, check):
        assert argv[-2:] == ['--cameras', 'wrist']
        from shutil import copytree
        copytree(analysis, root/'primary_aux_analysis')
    monkeypatch.setattr(module.subprocess, 'run', run)
    report = module.run_experiment(capture_dir=root)
    assert report['auxiliary_camera'] is None
