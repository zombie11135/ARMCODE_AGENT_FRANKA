"""Run real perception and local VLM inference on a saved RGB image, without devices."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/offline_stack')
    parser.add_argument('--hardware-config', type=Path, default=ROOT/'config/hardware/fr3.yaml')
    parser.add_argument('--skip-vlm', action='store_true')
    parser.add_argument('--vlm-model', choices=['qwen3vl-8b-awq', 'qwen35-9b-awq'], default='qwen3vl-8b-awq')
    parser.add_argument('--vlm-port', type=int, default=8010)
    args = parser.parse_args()
    args.image = args.image.expanduser().resolve(strict=True)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    report = dict(image=str(args.image), hardware_accessed=False, rgb_only=True, stages={})

    def stage(name, function):
        start = time.monotonic()
        value = function()
        report['stages'][name] = dict(seconds=round(time.monotonic()-start, 3), result=value)
        (args.output/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(f'[offline] {name}: OK ({report["stages"][name]["seconds"]} s)', flush=True)
        return value

    try:
        import torch
        from hardware.config import load_config
        from services.grounding_dino_detector_service import GroundingDINODetectorService
        from services.object_reference_service import ObjectReferenceService, load_settings
        from services.sam3_bbox_segmenter_service import SAM3BBoxSegmenterService
        from core.context import RuntimeContext
        from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; run on the host with GPU access')
        profile = load_config(args.hardware_config)
        settings = load_settings(profile.perception.reference_config)
        report['gpu'] = torch.cuda.get_device_name(0)
        detector = GroundingDINODetectorService(profile.perception.grounding_dino_model,
                                               device=settings.device, local_files_only=True)
        detections = stage('grounding_dino', lambda: detector.detect(
            image_path=str(args.image), target_name='medicine_box'))
        if not detections['detections']:
            raise RuntimeError('No medicine boxes detected; choose a saved medicine scene')

        references = ObjectReferenceService(settings)
        finder = FindAndGraspMedicineSkill(RuntimeContext(detector=detector, object_references=references))

        def recognize():
            results = []
            for item in references.catalog():
                if item['category'] != 'medicine_box':
                    continue
                response = finder._recognize_image(image_path=str(args.image), query=item['name'],
                    use_sam_for_selection=False, use_mask_for_grasp=False, verify_ocr=False)
                if response.data.get('recognition_schedule', {}).get('source') != 'reference_gallery':
                    raise RuntimeError(response.error or 'Reference gallery recognition did not run')
                results.append(dict(query=item['name'], success=response.success, error=response.error, data=response.data))
            if not results:
                raise RuntimeError('No medicine references in the configured gallery')
            return results

        stage('mobileclip2_recognition', recognize)

        def segment():
            segmenter = SAM3BBoxSegmenterService(profile.perception.sam3_model_dir,
                device=settings.device, prompt_mode='instance')
            result = segmenter.segment_by_bbox(str(args.image), detections['detections'][0]['bbox'],
                output_prefix=str(args.output/'sam3'))
            if result['mask_area'] <= 0:
                raise RuntimeError('SAM3 produced an empty mask')
            return result

        stage('sam3_segmentation', segment)
        if not args.skip_vlm:
            from runtime.local_vlm import start_server
            from services.openai_chatgpt_vlm_client import OpenAIChatGPTVLMClient
            start_server(args.vlm_model, port=args.vlm_port)
            client = OpenAIChatGPTVLMClient(f'http://127.0.0.1:{args.vlm_port}/v1', 'local', args.vlm_model)

            def text_response():
                result = client.chat('这是本地部署连通性测试，请只回复：OK')
                if not result.strip():
                    raise RuntimeError('Local VLM returned empty text')
                return result

            def vision_response():
                result = client.describe_images([str(args.image)],
                    '请用中文简短描述图中有哪些物体，以及药盒之间是否有遮挡。只描述图片，不执行动作。')
                if not result.strip():
                    raise RuntimeError('Local VLM returned an empty visual response')
                return result

            stage('local_vlm_text', text_response)
            stage('local_vlm_vision', vision_response)
        report['passed'] = True
        report['note'] = 'Runtime smoke check only; no robot, camera, depth geometry or AnyGrasp inference. Recognition quality requires independent labeled scenes.'
    except Exception as error:
        report.update(passed=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (args.output/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(passed=True, results=str(args.output/'results.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
