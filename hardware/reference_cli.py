"""Offline reference-gallery management and image replay. Never accesses hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from services.object_reference_service import (
    ObjectReferenceService, REFERENCE_VIEWPOINTS, atomic_json, image_digest, load_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def import_snapshots(service, directory):
    imported = []
    for snapshot in sorted(Path(directory).iterdir()):
        source = snapshot / "grasp_input.json"
        if not source.exists() or not (snapshot / "crop.png").exists():
            continue
        data = json.loads(source.read_text())
        name = data.get("query") or snapshot.name.split("_", 2)[-1]
        if not str(name).strip():
            continue
        color = snapshot / "color.png"
        imported.append(service.gallery.register(name=name, image_path=snapshot / "crop.png",
            category="medicine_box", detector_prompt="medicine package",
            source_group=snapshot.name, provenance="historical_snapshot_label_not_new_manual_annotation",
            source_image_sha256=image_digest(color) if color.exists() else None))
    return {"imported_snapshots": len(imported), "catalog": service.catalog(),
            "note": "历史标签作为参考先验；不是当前场景的人工标注验证。"}


def build_image_context(service, config):
    from core.context import RuntimeContext
    from hardware.config import load_config
    from services.grounding_dino_detector_service import GroundingDINODetectorService
    from services.grasp_service import LazySAM3
    from services.recognition_runtime import build_lazy_ocr
    profile = load_config(config)
    return RuntimeContext(object_references=service,
        detector=GroundingDINODetectorService(model_name_or_path=profile.perception.grounding_dino_model,
            device=service.settings.device, threshold=.22, text_threshold=.18, local_files_only=True),
        segmenter=LazySAM3(profile.perception.sam3_model_dir, prompt_mode="instance"),
        ocr=build_lazy_ocr(dict(lang="ch", device="cpu", use_textline_orientation=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/perception/references.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("warmup")
    register = sub.add_parser("register")
    register.add_argument("name")
    register.add_argument("--images", nargs="+", required=True)
    register.add_argument("--category", required=True)
    register.add_argument("--prompt", required=True, help="English candidate detection description")
    register.add_argument("--alias", action="append", default=[])
    register.add_argument("--brand")
    register.add_argument("--specification")
    register.add_argument("--descriptor", action="append", default=[], help="Shared description; not a unique alias")
    register.add_argument("--bbox", nargs=4, type=float)
    register.add_argument("--viewpoint", choices=REFERENCE_VIEWPOINTS,
                          help="Optional photo label; all views keep the same object identity")
    update = sub.add_parser('update', help='Update registered names without re-importing photos')
    update.add_argument('object')
    update.add_argument('--name', help='New canonical name; old name is retained as an alias')
    update.add_argument('--alias', action='append', default=[])
    update.add_argument('--remove-alias', action='append', default=[])
    update.add_argument('--brand')
    update.add_argument('--specification')
    update.add_argument('--descriptor', action='append', default=[])
    resolve = sub.add_parser('resolve', help='Resolve names; models load only if --image is provided')
    resolve.add_argument('query')
    resolve.add_argument('--image')
    resolve.add_argument('--bind', action='append', default=[], metavar='NAME=OBJECT')
    resolve.add_argument('--hardware-config', default=str(ROOT / 'config/hardware/fr3.yaml'))
    resolve.add_argument('--output')
    imp = sub.add_parser("import-snapshots")
    imp.add_argument("directory", type=Path)
    match = sub.add_parser("match")
    match.add_argument("--image", required=True)
    match.add_argument("--query", required=True)
    match.add_argument("--bbox", nargs=4, type=float, required=True)
    replay = sub.add_parser("recognize")
    replay.add_argument("--image", required=True)
    replay.add_argument("--query", required=True)
    replay.add_argument("--segment", action="store_true")
    replay.add_argument("--legacy-only", action="store_true")
    replay.add_argument("--repeat", type=int, default=1)
    replay.add_argument("--hardware-config", default=str(ROOT / "config/hardware/fr3.yaml"),
                        help="Read model paths only; never constructs hardware clients")
    replay.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate-snapshots")
    evaluate.add_argument("directory", type=Path)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args()
    service = ObjectReferenceService(load_settings(args.config))
    started = time.perf_counter()
    if args.command == "list":
        result = {"catalog": service.catalog(), "gallery_dir": service.settings.gallery_dir}
    elif args.command == "register":
        result = {"references": [service.gallery.register(name=args.name, image_path=image,
            category=args.category, detector_prompt=args.prompt, aliases=args.alias, bbox=args.bbox,
            attributes={k: getattr(args, k) for k in ('brand', 'specification') if getattr(args, k) is not None},
            descriptors=args.descriptor, viewpoint=args.viewpoint)
            for image in args.images]}
    elif args.command == 'update':
        obj = service.gallery.update(args.object, name=args.name, aliases=args.alias,
            attributes={k: getattr(args, k) for k in ('brand', 'specification') if getattr(args, k) is not None},
            descriptors=args.descriptor, remove_aliases=args.remove_alias)
        result = {k: v for k, v in obj.items() if k != 'references'}
        result['reference_count'] = len(obj['references'])
    elif args.command == 'resolve':
        from services.object_names import ObjectNameResolver
        bindings = {}
        for value in args.bind:
            if '=' not in value:
                parser.error('--bind needs NAME=OBJECT')
            key, value = value.split('=', 1)
            bindings[key.strip()] = value.strip()
        resolver = ObjectNameResolver(service, bindings)
        result = resolver.lookup(args.query)
        if args.image and result['status'] == 'needs_scene':
            context = build_image_context(service, args.hardware_config)
            result = resolver.resolve_in_image(args.query, context, args.image)
        result['hardware_accessed'] = False
        result['command_elapsed_s'] = time.perf_counter() - started
        if args.output:
            atomic_json(args.output, result)
    elif args.command == "import-snapshots":
        result = import_snapshots(service, args.directory)
    elif args.command == "warmup":
        result = {"gallery_cache": service._load_features(), "catalog": service.catalog()}
    elif args.command == "match":
        result = service.match(image_path=args.image, candidates=[{"bbox": args.bbox}], query=args.query)
    elif args.command == "recognize":
        from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
        context = build_image_context(service, args.hardware_config)
        context.event_callback = lambda event: print(event.get("content", "")) if event.get("type") != "image" else None
        finder = FindAndGraspMedicineSkill(context)
        runs = []
        for _ in range(max(1, args.repeat)):
            tick = time.perf_counter()
            response = finder._recognize_image(image_path=args.image, query=args.query,
                use_sam_for_selection=args.segment, use_mask_for_grasp=args.segment,
                skip_reference_matching=args.legacy_only)
            runs.append(dict(success=response.success, error=response.error, data=response.data,
                             elapsed_s=time.perf_counter() - tick))
        result = {"hardware_accessed": False, "runs": runs, "elapsed_s": time.perf_counter()-started}
        atomic_json(args.output, result)
        result = {"hardware_accessed": False, "output": str(Path(args.output).resolve()),
                  "success": [r["success"] for r in runs], "elapsed_s": [r["elapsed_s"] for r in runs],
                  "route": [r["data"].get("recognition_schedule", {}).get("source") for r in runs]}
    else:
        cases = []
        for snapshot in sorted(args.directory.iterdir()):
            source = snapshot / "grasp_input.json"
            if not source.exists() or not (snapshot / "color.png").exists():
                continue
            data = json.loads(source.read_text())
            name = data.get("query") or snapshot.name.split("_", 2)[-1]
            if not data.get("bbox"):
                continue
            matched = service.match(image_path=snapshot / "color.png", candidates=[{"bbox": data["bbox"]}],
                                    query=name, exclude_groups=[snapshot.name])
            cases.append(dict(snapshot=snapshot.name, expected=name, result=matched))
        accepted = sum(c["result"]["accepted"] for c in cases)
        result = dict(hardware_accessed=False, evaluation="historical_known_bbox_identity_only_leave_capture_out",
                      warning="历史标签/目标框并非独立人工真值；同批物体，不代表新场景或端到端成功率。",
                      case_count=len(cases), accepted=accepted, cases=cases)
        atomic_json(args.output, result)
        result = {k: v for k, v in result.items() if k != "cases"}
        result["output"] = str(Path(args.output).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
