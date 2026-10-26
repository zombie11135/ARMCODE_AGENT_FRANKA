"""Replay the production recognition path on saved RGB photos, without hardware.

Example (from repo root):
  .venv-mobileclip2/bin/python -B skill_lab/test_observe_gallery_offline.py \
    --images data/object_gallery/test_image --output runs/observe_gallery_replay

Uses real DINO/MobileCLIP2 inference; does not start a camera, AnyGrasp or robot.
RGB-D construction and Gate compatibility are tested separately in CPU tests.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'config/perception/references.yaml')
    parser.add_argument('--dino-model', type=Path, default=ROOT/'models/grounding-dino-tiny')
    parser.add_argument('--visual-reference', type=Path)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    import torch
    from core.context import RuntimeContext
    from services.grounding_dino_detector_service import GroundingDINODetectorService
    from services.object_reference_service import ObjectReferenceService, atomic_json, load_settings
    from services.reference_encoder import build_reference_encoder
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill

    settings = load_settings(args.config)
    assert settings.encoder_model == 'mobileclip2_b' and settings.recognition_policy == 'reference_only'
    if settings.device.startswith('cuda'):
        assert torch.cuda.is_available(), 'GPU unavailable; run in an environment with GPU access'
    args.output.mkdir(parents=True, exist_ok=True)

    class CountingDetector:
        def __init__(self):
            self.calls = 0
            self.backend = GroundingDINODetectorService(str(args.dino_model), device=settings.device,
                                                        local_files_only=True)
        def detect(self, **kwargs):
            self.calls += 1
            return self.backend.detect(**kwargs)

    class CountingEncoder:
        def __init__(self):
            self.calls = 0
            self.backend = None
        def encode(self, images, batch_size=16):
            self.calls += 1
            if self.backend is None:
                self.backend = build_reference_encoder(settings.encoder_model, settings.checkpoint, settings.device)
            return self.backend.encode(images, batch_size=batch_size)

    class ForbiddenOCR:
        def __getattr__(self, name):
            raise AssertionError('OCR must not be used in reference_only replay: '+name)

    detector, encoder = CountingDetector(), CountingEncoder()
    service = ObjectReferenceService(settings, encoder=encoder)
    gallery_cache = service._load_features()
    gallery_encoding_calls = encoder.calls
    context = RuntimeContext(detector=detector, object_references=service, ocr=ForbiddenOCR())
    finder = FindAndGraspMedicineSkill(context)
    queries = [obj['name'] for obj in service.catalog() if obj['category'] == 'medicine_box']
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    assert paths, 'No input photos'
    truth = json.loads(args.visual_reference.read_text())['images'] if args.visual_reference else None
    runs, correctly_localized, false_accepts = [], 0, 0

    def overlap(a, b):
        inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/union if union else 0

    for image in paths:
        before = detector.calls
        before_encoding = encoder.calls
        for query in queries:
            started = time.perf_counter()
            response = finder._recognize_image(image_path=str(image.resolve()), query=query,
                use_sam_for_selection=False, use_mask_for_grasp=False, verify_ocr=True)
            assert response.data.get('recognition_schedule', {}).get('source') == 'reference_gallery', response.error
            assert response.data['recognition_schedule']['calls']['ocr'] == 0
            if truth is not None and response.success:
                box = response.data['selected_instance']['bbox']
                correct = any(r['label'] == query and overlap(box, r['bbox']) >= .5 for r in truth[image.name])
                correctly_localized += int(correct)
                false_accepts += int(not correct)
            record = dict(image=image.name, query=query, success=response.success, error=response.error,
                          elapsed_s=time.perf_counter()-started, data=response.data)
            runs.append(record)
            print(image.name, query, response.success,
                  round(response.data.get('identity_evidence', {}).get('score', 0), 4), flush=True)
        assert detector.calls-before == 1, 'Expected one DINO inference per new photo across all drug queries'
        assert encoder.calls-before_encoding == 1, 'Expected one batched candidate encoding per new photo'
    summary = dict(hardware_accessed=False, rgb_only=True, config=str(args.config.resolve()),
        model=settings.encoder_model, policy=settings.recognition_policy,
        thresholds=dict(similarity=settings.min_similarity, margin=settings.min_class_margin),
        image_count=len(paths), queries=queries, detector_calls=detector.calls,
        candidate_encoding_batches=encoder.calls-gallery_encoding_calls, gallery_cache=gallery_cache,
        successful_requests=sum(r['success'] for r in runs), rejected_requests=sum(not r['success'] for r in runs),
        correctly_localized=correctly_localized if truth is not None else None,
        false_accepts=false_accepts if truth is not None else None,
        note='Real production recognition code on saved RGB; no live RGB-D or robot validation. Adjacent frames, if present, are not independent test scenes.')
    atomic_json(args.output/'results.json', dict(summary=summary, runs=runs))
    atomic_json(args.output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if truth is not None:
        expected = sum(sum(r['label'] in queries for r in truth[p.name]) for p in paths)
        assert correctly_localized == expected and false_accepts == 0, 'Photo regression failed; inspect results.json'


if __name__ == '__main__':
    main()
