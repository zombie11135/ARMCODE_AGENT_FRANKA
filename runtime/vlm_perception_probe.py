"""Measure local VLM coexistence using saved RGB and an offline AnyGrasp request."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from runtime.local_vlm import ROOT, running_models
from runtime.vlm_benchmark import MemorySamples, gpu_used_mib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--bbox', type=int, nargs=4, required=True)
    parser.add_argument('--anygrasp-request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vlm-model', required=True)
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--release-idle-cache', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    if args.vlm_model not in running_models(args.port):
        parser.error('Start the requested local VLM first')
    args.output.mkdir(parents=True)
    import cv2
    import torch
    from PIL import Image
    from hardware.config import load_config
    from services.wrist_occlusion_service import WristOcclusionService
    from services.grounding_dino_detector_service import GroundingDINODetectorService
    from services.sam3_bbox_segmenter_service import SAM3BBoxSegmenterService
    from services.object_reference_service import ObjectReferenceService, load_settings
    from services.openai_chatgpt_vlm_client import OpenAIChatGPTVLMClient

    config = load_config(ROOT/'config/hardware/fr3.yaml')
    records = []
    models = []
    def stage(name, function):
        started = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        records.append(dict(stage=name, seconds=time.perf_counter()-started,
            device_used_mib=gpu_used_mib(), torch_allocated_mib=torch.cuda.memory_allocated()/2**20,
            torch_reserved_mib=torch.cuda.memory_reserved()/2**20))
        print(json.dumps(records[-1]), flush=True)
        return value

    result = dict(vlm_model=args.vlm_model, robot_or_camera_accessed=False,
        note='Resource probe only; RGB model probes and saved grasp request are independent historical inputs.',
        release_idle_cache=args.release_idle_cache)
    with MemorySamples() as samples:
        try:
            dino = stage('load_dino', lambda: GroundingDINODetectorService(
                config.perception.grounding_dino_model, device='cuda', local_files_only=True))
            models.append(dino)
            stage('dino_inference', lambda: dino.detect(str(args.image), 'medicine_box'))
            references = ObjectReferenceService(load_settings(config.perception.reference_config))
            models.append(references)
            stage('mobileclip_inference', lambda: references._encode([Image.open(args.image).convert('RGB')]))
            sam = stage('load_bbox_sam3', lambda: SAM3BBoxSegmenterService(
                config.perception.sam3_model_dir, device='cuda', prompt_mode='instance'))
            models.append(sam)
            stage('bbox_sam3_inference', lambda: sam.segment_by_bbox(str(args.image), args.bbox,
                output_prefix=str(args.output/'bbox_sam3')))
            occlusion = WristOcclusionService(str(Path(config.perception.sam3_model_dir)/'sam3.pt'))
            stage('load_occlusion_sam3', occlusion._ensure_backend)
            models.append(occlusion)
            result['sam3_weights_shared'] = sam.model is occlusion.backend.model
            stage('occlusion_sam3_inference', lambda: occlusion.backend.predict(cv2.imread(str(args.image)),
                ['box', 'bottle', 'plate', 'cube']))
            if args.release_idle_cache:
                stage('release_idle_cache', torch.cuda.empty_cache)
            request = json.loads(args.anygrasp_request.read_text())
            request['output_dir'] = str((args.output/'anygrasp').resolve())
            request_path = args.output/'anygrasp_request.json'
            request_path.write_text(json.dumps(request, indent=2))
            from services.anygrasp_fr3_service import AnyGraspFR3Service
            # This factory validates paths/transforms only; inference reads saved data.
            service = AnyGraspFR3Service(ROOT/'config/hardware/fr3.yaml')
            with (args.output/'anygrasp_worker.log').open('w') as log:
                stage('anygrasp_inference', lambda: service._run_inference(request_path.resolve(), log))
            if args.release_idle_cache:
                stage('release_idle_cache_before_vlm', torch.cuda.empty_cache)
            client = OpenAIChatGPTVLMClient(f'http://127.0.0.1:{args.port}/v1', 'local', args.vlm_model)
            result['vlm_response'] = stage('vlm_with_resident_perception', lambda:
                client.describe_images([str(args.image)], '只用一句中文描述桌面可见物体。', max_tokens=100))
            result['completed'] = True
        except Exception as error:
            result.update(completed=False, error_type=type(error).__name__)
            raise
        finally:
            result.update(stages=records, device_peak_used_mib=max(samples.values) if samples.values else None)
            (args.output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
