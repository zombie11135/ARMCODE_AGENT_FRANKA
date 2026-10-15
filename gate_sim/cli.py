"""Command-line entry points for demos, validation and small comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gate_sim.runner import (
    environment_audit,
    record_gate_action_videos,
    record_manipulation_probe_video,
    render_scene_gallery,
    run_comparison,
    run_demos,
    run_restore_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="GateOccludedPickPlace-v0 simulation-only tools")
    parser.add_argument("command", choices=["audit", "render", "manip-video", "gate-videos", "demo", "restore", "compare", "validate"])
    parser.add_argument("--output-dir", default="runs/gate_sim/phase1")
    parser.add_argument("--seeds", type=int, default=8, help="fixed seeds for compare; hard-capped at 20")
    parser.add_argument("--width", type=int, default=1280, help="overview width for render")
    parser.add_argument("--height", type=int, default=720, help="overview height for render")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.command == "audit":
        result = environment_audit()
    elif args.command == "render":
        result = render_scene_gallery(args.output_dir, args.width, args.height)
    elif args.command == "manip-video":
        result = record_manipulation_probe_video(args.output_dir, args.width, args.height)
    elif args.command == "gate-videos":
        result = record_gate_action_videos(args.output_dir, args.width, args.height)
    elif args.command == "demo":
        result = run_demos(args.output_dir)
    elif args.command == "restore":
        result = run_restore_validation(args.output_dir)
    elif args.command == "compare":
        if args.seeds < 1 or args.seeds > 20:
            parser.error("--seeds must be in [1, 20]")
        result = run_comparison(args.output_dir, range(args.seeds))
    else:
        demos = run_demos(args.output_dir)
        restore = run_restore_validation(args.output_dir)
        comparison = run_comparison(args.output_dir, range(min(args.seeds, 20)))
        result = {
            "audit": environment_audit(),
            "demo_success": [row["task_success"] for row in demos["results"]],
            "fork_evaluation_ready": restore["fork_evaluation_ready"],
            "comparison_summary": comparison["summary"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
