"""Observation contract and single-detection policy; CPU-only, no hardware/models."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import pytest

from core.context import RuntimeContext
from core.skill_base import SkillResult
from services.object_reference_service import ObjectReferenceService, ReferenceSettings, image_digest, load_settings
from services.reference_encoder import MOBILECLIP2_PREPROCESSING
from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
from skill_lib.perception.reference_recognition import RecognizeObjectSkill
from skill_lib.simple_actions import SimpleRobotActions
from nesyro_gate.scene_memory_builder import medicine_observation_to_object_state


class ColorEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, images, batch_size=16):
        self.calls += 1
        return np.asarray([np.asarray(im).mean(axis=(0, 1)) + .01 for im in images])


@pytest.fixture
def scene(tmp_path):
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'unit-test-encoder')
    settings = ReferenceSettings(gallery_dir=str(tmp_path/'gallery'), checkpoint=str(checkpoint),
        log_dir=str(tmp_path/'logs'), device='cpu', encoder_model='mobileclip2_b',
        recognition_policy='reference_only', min_similarity=.8, min_class_margin=.02)
    encoder = ColorEncoder()
    service = ObjectReferenceService(settings, encoder=encoder)
    for name, color in [('布洛芬', 'red'), ('西瓜霜', 'blue')]:
        reference = tmp_path/f'{color}.png'
        Image.new('RGB', (40, 40), color).save(reference)
        service.gallery.register(name=name, image_path=reference, category='medicine_box',
            detector_prompt='medicine package', aliases=['芬必得'] if color == 'red' else [])
    image = tmp_path/'scene.png'
    pixels = Image.new('RGB', (80, 40), 'red')
    pixels.paste('blue', (40, 0, 80, 40)); pixels.save(image)
    depth = tmp_path/'depth.png'
    Image.fromarray(np.full((40, 80), 500, dtype=np.uint16)).save(depth)
    detector, ocr, camera, anygrasp = Mock(), Mock(), Mock(), Mock()
    detector.detect.return_value = {'detections': [
        dict(object_id='obj_1', bbox=[0, 0, 40, 40], confidence=.9),
        dict(object_id='obj_2', bbox=[40, 0, 80, 40], confidence=.8)]}
    capture = dict(color_path=str(image), depth_path=str(depth), depth_scale=1000,
                   color_intrinsics=dict(fx=100, fy=100, cx=40, cy=20))
    camera.capture_wrist.return_value = capture
    context = RuntimeContext(camera=camera, detector=detector, ocr=ocr,
                             anygrasp=anygrasp, object_references=service)
    return context, image, encoder, capture


def test_observe_preserves_instances_indices_depth_and_gate_identity(scene):
    context, image, _, capture = scene
    # FR3 supplies calibrated capture intrinsics; legacy xArm uses a fallback.
    context.transforms = Mock()
    robot = SimpleRobotActions(context)
    result = robot._observe_medicine_once('西瓜霜')
    assert result.success, result.error
    data = result.data
    assert {'image_path', 'prompts', 'detections', 'instances', 'selected_instance',
            'selection_result', 'ranked_candidates', 'ocr_verification', 'grasp_input'} <= data.keys()
    assert len(data['instances']) == 2
    selected = data['selected_instance']
    assert data['selection_result']['selected_index'] == 1
    assert data['instances'][1] == selected
    assert data['grasp_input']['bbox'] == selected['bbox'] == [40, 0, 80, 40]
    for key in ('color_path', 'depth_path', 'depth_scale', 'color_intrinsics'):
        assert data['grasp_input'][key] == capture[key]
    assert selected['ocr_text'] == '' and selected['ocr_texts'] == []
    assert data['ocr_verification'] is None and Path(selected['crop_path']).is_file()
    assert data['identity_evidence']['encoder_model'] == 'mobileclip2_b'
    context.detector.detect.assert_called_once()
    assert context.detector.detect.call_args.kwargs['target_name'] == 'medicine_box'
    assert context.ocr.mock_calls == [] and context.anygrasp.mock_calls == []
    state = medicine_observation_to_object_state('西瓜霜', robot.observations['西瓜霜'])
    assert state.identity_verified and state.reachable and state.graspable
    assert state.ocr_match_score == state.vlm_match_score == 0


def test_two_targets_and_alias_share_detection_and_batch_features(scene):
    context, image, encoder, _ = scene
    finder = FindAndGraspMedicineSkill(context)
    for query in ['布洛芬', '西瓜霜', '芬必得']:
        result = finder._recognize_image(image_path=str(image), query=query)
        assert result.success, result.error
        assert result.data['identity_evidence']['query'] == query
    assert context.detector.detect.call_count == 1
    assert encoder.calls == 2  # One reference batch and one scene batch, independent of query count.
    assert context.ocr.mock_calls == []
    Image.new('RGB', (80, 40), 'blue').save(image)
    result = finder._recognize_image(image_path=str(image), query='西瓜霜')
    assert result.success
    assert context.detector.detect.call_count == 2 and encoder.calls == 3


@pytest.mark.parametrize('condition,stage', [('ambiguous', 'reference_uncertain'),
    ('empty', 'reference_uncertain'), ('unknown', 'unregistered_target'), ('error', 'reference_error')])
def test_reference_only_failures_never_call_legacy_ocr(scene, condition, stage):
    context, image, encoder, _ = scene
    query = '布洛芬'
    if condition == 'ambiguous': Image.new('RGB', (80, 40), (255, 0, 255)).save(image)
    elif condition == 'empty': context.detector.detect.return_value = {'detections': []}
    elif condition == 'unknown': query = '未登记药品'
    else: encoder.encode = Mock(side_effect=RuntimeError('encoder unavailable'))
    finder = FindAndGraspMedicineSkill(context)
    with patch.object(finder, '_recognize_legacy') as legacy:
        result = finder.run(query=query, plan_grasp=False, execute=False)
    assert not result.success and result.data['stage'] == stage
    assert not result.data.get('selected_instance') and 'grasp_input' not in result.data
    legacy.assert_not_called()
    assert context.ocr.mock_calls == [] and context.anygrasp.mock_calls == []
    assert context.detector.detect.call_count == (0 if condition == 'unknown' else 1)


@pytest.mark.parametrize('option', ['skip_reference_matching', 'ocr_verify_strict_mode'])
def test_explicit_legacy_request_retains_ocr_semantics(scene, option):
    context, image, _, _ = scene
    finder = FindAndGraspMedicineSkill(context)
    with patch.object(finder, '_recognize_legacy', return_value=SkillResult(True, {})) as legacy:
        result = finder._recognize_image(image_path=str(image), query='布洛芬', **{option: True})
    assert result.success
    legacy.assert_called_once()
    context.detector.detect.assert_not_called()


def test_custom_prompt_list_is_one_detection_request(scene):
    context, image, _, _ = scene
    result = FindAndGraspMedicineSkill(context)._recognize_image(image_path=str(image), query='布洛芬',
        target_prompts=['medicine_box', 'rectangular medicine carton', 'small carton'])
    assert result.success
    context.detector.detect.assert_called_once()
    prompt = context.detector.detect.call_args.kwargs['target_name']
    assert prompt.count('rectangular medicine carton') == 1 and 'small carton.' in prompt


def test_shared_name_resolution_preserves_gate_query_and_detection_parameters(scene):
    context, image, _, capture = scene
    service = context.object_references
    for name in ['布洛芬', '西瓜霜']:
        service.gallery.update(name, descriptors=['共同简称'])
    context.detector.detect.return_value['detections'] = context.detector.detect.return_value['detections'][:1]
    response = FindAndGraspMedicineSkill(context).run(query='共同简称', detection_threshold=.31,
                                                     plan_grasp=False, execute=False)
    assert response.success, response.error
    context.detector.detect.assert_called_once()
    assert context.detector.detect.call_args.kwargs['threshold'] == .31
    assert response.data['identity_evidence']['query'] == '共同简称'
    observation = dict(result=response.data, grasp_input=response.data['grasp_input'],
                       selected_instance=response.data['selected_instance'])
    assert medicine_observation_to_object_state('共同简称', observation).identity_verified


def test_encoder_change_invalidates_persistent_feature_space(scene):
    context, _, encoder, _ = scene
    service = context.object_references
    assert service._load_features() == 'built'
    old_key = service._feature_key
    other = ObjectReferenceService(replace(service.settings, encoder_model='mobileclip_blt'), encoder=encoder)
    assert other._load_features() == 'built' and other._feature_key != old_key


def write_prepared_bank(service, directory):
    objects = service.gallery.read()['objects']
    rows = [dict(r, object_id=o['object_id']) for o in reversed(objects) for r in o['references']]
    features = np.zeros((len(rows), 512), dtype=np.float32)
    for i in range(len(rows)): features[i, i] = 1
    directory.mkdir()
    index = directory/'index.npz'
    np.savez_compressed(index, features=features, object_ids=[r['object_id'] for r in rows],
                        image_sha256=[r['sha256'] for r in rows])
    meta = dict(model_key='mobileclip2_b', checkpoint_sha256=image_digest(service.settings.checkpoint),
                preprocessing=MOBILECLIP2_PREPROCESSING, index_sha256=image_digest(index), references=rows)
    (directory/'metadata.json').write_text(json.dumps(meta))
    return meta, features


def test_prepared_bank_maps_rows_by_identity_and_does_not_encode_gallery(scene, tmp_path):
    context, _, _, _ = scene
    service = context.object_references
    directory = tmp_path/'prepared'
    _, bank = write_prepared_bank(service, directory)
    fresh = ObjectReferenceService(replace(service.settings, vector_library_dir=str(directory)))
    assert fresh._load_features() == 'prepared'
    np.testing.assert_array_equal(fresh._features, bank[::-1])
    assert fresh._encoder is None
    assert fresh._load_features() == 'memory'


@pytest.mark.parametrize('damage', ['wrong_model', 'wrong_checkpoint', 'wrong_preprocess', 'corrupt_index'])
def test_incompatible_prepared_banks_are_rejected(scene, tmp_path, damage):
    context, _, _, _ = scene
    service = context.object_references
    directory = tmp_path/'prepared'
    meta, _ = write_prepared_bank(service, directory)
    if damage == 'wrong_model': meta['model_key'] = 'mobileclip_blt'
    elif damage == 'wrong_checkpoint': meta['checkpoint_sha256'] = 'wrong'
    elif damage == 'wrong_preprocess': meta['preprocessing'] = 'stretch'
    else: (directory/'index.npz').write_bytes(b'corrupt')
    (directory/'metadata.json').write_text(json.dumps(meta))
    fresh = ObjectReferenceService(replace(service.settings, vector_library_dir=str(directory)))
    with pytest.raises(ValueError): fresh._load_features()


def test_shipped_config_is_loadable_and_selects_production_encoder():
    root = Path(__file__).resolve().parents[1]
    cfg = load_settings(root/'config/perception/references.example.yaml')
    assert cfg.encoder_model == 'mobileclip2_b' and cfg.recognition_policy == 'reference_only'
    assert (cfg.min_similarity, cfg.min_class_margin, cfg.top_k_references) == (.8, .02, 3)
