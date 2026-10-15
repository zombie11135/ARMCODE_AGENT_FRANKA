"""Command line tools for the multimodal Gate prototype."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gate_learning.cache import cache_manifest
from gate_learning.encoders import cosine_similarity_matrix, create_frozen_encoder
from gate_learning.input_adapter import MULTIMODAL_STATE_FEATURE_NAMES, load_rgb_depth
from gate_learning.model import HybridGateConfig, HybridMultimodalGate, trainable_parameter_count
from gate_learning.train import train_from_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOBILECLIP = PROJECT_ROOT / "dependencies/yoloe/mobileclip_blt.pt"


def _encoder(args):
    return create_frozen_encoder(
        args.encoder,
        checkpoint=args.encoder_checkpoint,
        device=args.encoder_device,
        local_files_only=not args.allow_download,
    )


def run_smoke(args) -> dict:
    rgb, depth = load_rgb_depth(args.rgb, args.depth)
    encoder = _encoder(args)
    started = time.perf_counter()
    image_features = encoder.encode_images([rgb])
    text_features = encoder.encode_texts([args.task, args.second_task])
    encoder_seconds = time.perf_counter() - started
    torch.manual_seed(7)
    config = HybridGateConfig(
        image_feature_dim=encoder.feature_dim,
        text_feature_dim=encoder.feature_dim,
        dropout=0.0,
    )
    model = HybridMultimodalGate(config).eval()
    state = torch.zeros(1, len(MULTIMODAL_STATE_FEATURE_NAMES))
    mask = torch.tensor([[True, True, True, False]])
    started = time.perf_counter()
    with torch.inference_mode():
        output_a = model(
            image_features=image_features,
            text_features=text_features[0:1],
            depth_m=torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0),
            state=state,
            action_mask=mask,
        )
        output_b = model(
            image_features=image_features,
            text_features=text_features[1:2],
            depth_m=torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0),
            state=state,
            action_mask=mask,
        )
    head_seconds = time.perf_counter() - started
    report = {
        "schema_version": "multimodal-gate-smoke-v1",
        "rgb": str(Path(args.rgb).resolve()),
        "depth": str(Path(args.depth).resolve()),
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "depth_unit": "m",
        "tasks": [args.task, args.second_task],
        "text_cosine_similarity": float(cosine_similarity_matrix(text_features)[0, 1]),
        "encoder_seconds_for_one_image_two_texts": encoder_seconds,
        "fusion_seconds_for_two_forward_passes": head_seconds,
        "encoder": encoder.metadata(),
        "fusion_trainable_parameters": trainable_parameter_count(model),
        "masked_replan_value": float(output_a["masked_action_values"][0, 3]),
        "raw_action_values_task_a": output_a["action_values"][0].tolist(),
        "raw_action_values_task_b": output_b["action_values"][0].tolist(),
        "task_conditioned_value_linf_delta": float(
            (output_a["action_values"] - output_b["action_values"]).abs().max()
        ),
        "checkpoint_loaded": False,
        "prediction_semantically_valid": False,
        "note": "This validates wiring and masks only; the fusion head is intentionally untrained.",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="RGB-D + task multimodal Gate tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_encoder_options(command) -> None:
        command.add_argument("--encoder", choices=["mobileclip", "siglip2"], default="mobileclip")
        command.add_argument("--encoder-checkpoint", default=str(DEFAULT_MOBILECLIP))
        command.add_argument("--encoder-device", default="cpu")
        command.add_argument("--allow-download", action="store_true")

    smoke = subparsers.add_parser("smoke", help="validate raw RGB-D/task encoding and fusion")
    smoke.add_argument("--rgb", required=True)
    smoke.add_argument("--depth", required=True)
    smoke.add_argument("--task", default="把燕麦盒放入紫色容器")
    smoke.add_argument("--second-task", default="把饮料罐放入紫色容器")
    smoke.add_argument("--output", default="runs/gate_learning/smoke_report.json")
    add_encoder_options(smoke)

    cache = subparsers.add_parser("cache", help="precompute frozen image/text features")
    cache.add_argument("--manifest", required=True)
    cache.add_argument("--output-dir", required=True)
    add_encoder_options(cache)

    train = subparsers.add_parser("train", help="train the lightweight fusion/action heads")
    train.add_argument("--cache-index", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default="cpu")

    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke(args)
    elif args.command == "cache":
        result = cache_manifest(args.manifest, args.output_dir, _encoder(args))
    else:
        result = train_from_cache(
            index_path=args.cache_index, output_dir=args.output_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            seed=args.seed, device=args.device,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
