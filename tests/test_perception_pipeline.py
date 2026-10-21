"""Deterministic concurrency tests with fake backends; no hardware or models."""
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from runtime.perception_pipeline import ordered_bounded_map, serialized_resource


def test_gpu_is_exclusive_but_cpu_ocr_can_overlap():
    overlap = threading.Barrier(2)
    @serialized_resource('gpu')
    def gpu(): overlap.wait(timeout=2)
    @serialized_resource('ocr_cpu')
    def ocr(): overlap.wait(timeout=2)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(gpu), pool.submit(ocr)]
        for future in futures:
            future.result(timeout=3)

    active, maximum = [0], [0]
    first, second_attempted, second_entered, release = (threading.Event() for _ in range(4))
    @serialized_resource('gpu')
    def guarded(index):
        active[0] += 1
        maximum[0] = max(maximum[0], active[0])
        try:
            if index == 0:
                first.set()
                assert release.wait(2)
            else:
                second_entered.set()
        finally:
            active[0] -= 1
    def second():
        second_attempted.set()
        guarded(1)
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(guarded, 0)
        try:
            assert first.wait(2)
            b = pool.submit(second)
            assert second_attempted.wait(2)
            assert not second_entered.wait(.05)
        finally:
            release.set()
        a.result(timeout=2)
        b.result(timeout=2)
    assert maximum[0] == 1


def test_map_is_bounded_ordered_and_does_not_eagerly_submit_everything():
    both_started, release = threading.Barrier(3), threading.Event()
    submitted = []
    def items():
        for i in range(8):
            submitted.append(i)
            yield i
    def process(i):
        if i < 2:
            both_started.wait(timeout=2)
            assert release.wait(2)
        return i * 10
    with ThreadPoolExecutor(1) as owner:
        future = owner.submit(lambda: list(ordered_bounded_map(process, items(), 2)))
        try:
            both_started.wait(timeout=2)
            assert submitted == [0, 1]
        finally:
            release.set()
        assert future.result(timeout=3) == list(range(0, 80, 10))


def test_failed_job_propagates_and_stops_future_submissions():
    visited = []
    def process(i):
        visited.append(i)
        if i == 0:
            raise RuntimeError('model error')
        return i
    with pytest.raises(RuntimeError, match='model error'):
        list(ordered_bounded_map(process, range(10), 2))
    assert set(visited) <= {0, 1}
    assert list(ordered_bounded_map(lambda i: i, [0, 1], 1)) == [0, 1]
    with pytest.raises(ValueError):
        list(ordered_bounded_map(lambda i: i, [], 0))


def test_real_recognition_branch_pipelines_candidates_and_publishes_on_owner_thread(tmp_path, monkeypatch):
    from core.context import RuntimeContext
    from core.skill_base import SkillResult
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    from skill_lib.perception.seg_obj import SegmentObjectSkill
    monkeypatch.chdir(tmp_path)
    ocr_a_started, segmentation_b_started, segmentation_a_finished = (threading.Event() for _ in range(3))
    prefixes, published_threads = [], []
    owner_thread = threading.get_ident()

    @serialized_resource('gpu')
    def gpu_segment(skill, **kwargs):
        prefix = kwargs['output_prefix']
        prefixes.append(prefix)
        if kwargs['bbox'][0] == 20:
            segmentation_b_started.set()
            assert ocr_a_started.wait(2), 'Candidate B SAM must overlap candidate A OCR'
        else:
            segmentation_a_finished.set()
        skill.context.emit_text('worker segmentation event')
        return SkillResult(True, data={'instances': [dict(success=True, bbox=kwargs['bbox'],
            crop_path=prefix + '_crop.png', mask_path=prefix + '_mask.png')]})
    def segment(skill, **kwargs):
        if kwargs['bbox'][0] == 20:
            # Schedule A first without occupying the GPU while waiting for it.
            assert segmentation_a_finished.wait(2)
        return gpu_segment(skill, **kwargs)
    monkeypatch.setattr(SegmentObjectSkill, 'run', segment)

    class OCR:
        @serialized_resource('ocr_cpu')
        def read_image(self, **kwargs):
            prefix = kwargs['output_prefix']
            prefixes.append(prefix)
            if 'candidate_00' in prefix:
                ocr_a_started.set()
                assert segmentation_b_started.wait(2), 'Candidate A OCR must overlap candidate B SAM'
            return dict(texts=['西瓜霜'], confidences=[.9], items=[])
        @serialized_resource('ocr_cpu')
        def read_region(self, **kwargs):
            prefixes.append(kwargs['output_prefix'])
            return dict(texts=['西瓜霜'], confidences=[.9], items=[])

    camera = Mock()
    camera.capture_wrist.side_effect = AssertionError('Worker must use saved frame only')
    context = RuntimeContext(camera=camera, ocr=OCR(), perception_workers=2,
        event_callback=lambda event: published_threads.append(threading.get_ident()))
    finder = FindAndGraspMedicineSkill(context)
    finder._detect_with_prompt_fusion = Mock(return_value=[
        dict(object_id='a', bbox=[0, 0, 10, 10], confidence=.9),
        dict(object_id='b', bbox=[20, 0, 30, 10], confidence=.8)])
    result = finder._recognize_legacy(image_path='immutable/source.png', query='西瓜霜',
                                     use_sam_for_selection=True, verify_ocr=False)
    assert result.success
    assert [i['object_id'] for i in result.data['instances']] == ['a', 'b']
    assert all(i['ocr_success'] for i in result.data['instances'])
    assert result.data['selected_instance']['object_id'] == 'a'
    assert len(prefixes) == len(set(prefixes))
    assert published_threads and set(published_threads) == {owner_thread}
    camera.capture_wrist.assert_not_called()
    # A second image analysis receives a different run directory.
    old_prefixes = set(prefixes)
    prefixes.clear()
    second = finder._recognize_legacy(image_path='immutable/source.png', query='西瓜霜',
                                      use_sam_for_selection=True, verify_ocr=False)
    assert second.success and not old_prefixes.intersection(prefixes)


def test_service_initialization_is_single_flight_under_concurrent_ocr_requests(monkeypatch):
    from services import runtime_service_cache as module
    monkeypatch.setattr(module, '_SERVICE_CACHE', {})
    barrier = threading.Barrier(3)
    service = object()
    factory = Mock(return_value=service)
    def load():
        barrier.wait(timeout=2)
        return module.get_or_create_service('test_ocr', {'lang': 'ch'}, factory)
    with ThreadPoolExecutor(2) as pool:
        a, b = pool.submit(load), pool.submit(load)
        barrier.wait(timeout=2)
        assert a.result(timeout=2) is service
        assert b.result(timeout=2) is service
    factory.assert_called_once()
