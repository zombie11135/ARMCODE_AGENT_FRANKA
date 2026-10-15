import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.planning_agent import PlanningAgent
from core.context import RuntimeContext


def print_event(event: dict, verbose: bool = False):
    event_type = event.get("type")

    if not verbose and event_type not in ["warning", "error"]:
        return

    if event_type == "image":
        print(f"[Image] {event.get('content')}: {event.get('image_path')}")
    elif event_type == "text":
        print(f"[Text] {event.get('content')}")
    elif event_type == "warning":
        print(f"[Warning] {event.get('content')}")
    elif event_type == "error":
        print(f"[Error] {event.get('content')}")
    else:
        print(event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--goal",
        default="把布洛芬放到托盘里",
    )
    parser.add_argument("--scene_text", default="")
    parser.add_argument("--use_vdm", action="store_true")
    parser.add_argument("--use_llm_planner", action="store_true")
    parser.add_argument("--output_json", default="runs/current_plan/plan_graph.json")
    parser.add_argument("--output_md", default="runs/current_plan/plan_graph.md")
    parser.add_argument("--verbose_events", action="store_true")

    # VDM/VLM
    parser.add_argument(
        "--vlm_base_url",
        default=os.getenv("PLANNER_LLM_BASE_URL", "http://10.201.102.163:8001/v1"),
    )
    parser.add_argument(
        "--vlm_api_key",
        default=os.getenv("PLANNER_LLM_API_KEY", "wpywpywpy"),
    )
    parser.add_argument(
        "--vlm_model",
        default=os.getenv("PLANNER_LLM_MODEL", "qwen-vl-8b"),
    )
    parser.add_argument("--vlm_timeout", type=int, default=120)
    parser.add_argument(
        "--fallback_vlm_base_url",
        default=os.getenv("PLANNER_VDM_FALLBACK_BASE_URL", os.getenv("PLANNER_API_BASE_URL", "")),
    )
    parser.add_argument(
        "--fallback_vlm_api_key",
        default=os.getenv("PLANNER_VDM_FALLBACK_API_KEY", os.getenv("PLANNER_API_KEY", "")),
    )
    parser.add_argument(
        "--fallback_vlm_model",
        default=os.getenv("PLANNER_VDM_FALLBACK_MODEL", os.getenv("PLANNER_API_MODEL", "")),
    )
    parser.add_argument(
        "--planner_llm_base_url",
        default=os.getenv("PLANNER_LLM_FALLBACK_BASE_URL", os.getenv("PLANNER_API_BASE_URL", "")),
    )
    parser.add_argument(
        "--planner_llm_api_key",
        default=os.getenv("PLANNER_LLM_FALLBACK_API_KEY", os.getenv("PLANNER_API_KEY", "")),
    )
    parser.add_argument(
        "--planner_llm_model",
        default=os.getenv("PLANNER_LLM_FALLBACK_MODEL", os.getenv("PLANNER_API_MODEL", "")),
    )

    # camera
    parser.add_argument("--global_serial", default="213522251050")
    parser.add_argument("--wrist_serial", default="243222070053")
    args = parser.parse_args()

    camera_service = None

    try:
        context = RuntimeContext(event_callback=lambda event: print_event(event, args.verbose_events))

        if args.use_vdm:
            from services.realsense_dual_camera_service import (
                CameraConfig,
                RealSenseDualCameraService,
            )

            camera_service = RealSenseDualCameraService(
                global_config=CameraConfig(
                    name="global",
                    serial=args.global_serial,
                    color_width=1280,
                    color_height=720,
                    color_fps=30,
                    enable_depth=False,
                ),
                wrist_config=CameraConfig(
                    name="wrist",
                    serial=args.wrist_serial,
                    color_width=1280,
                    color_height=720,
                    color_fps=30,
                    enable_depth=True,
                    depth_width=1280,
                    depth_height=720,
                    depth_fps=30,
                ),
                save_dir="runs/current_capture",
                show_window=True,
            )
            camera_service.start()
            time.sleep(2.0)
            context.camera = camera_service

        if args.use_vdm or args.use_llm_planner:
            from services.vlm_client import QwenVLClient

            context.vlm = QwenVLClient(
                base_url=args.vlm_base_url,
                api_key=args.vlm_api_key,
                model_name=args.vlm_model,
                timeout=args.vlm_timeout,
            )

            if (
                args.fallback_vlm_base_url
                and args.fallback_vlm_api_key
                and args.fallback_vlm_model
            ):
                context.fallback_vlm = QwenVLClient(
                    base_url=args.fallback_vlm_base_url,
                    api_key=args.fallback_vlm_api_key,
                    model_name=args.fallback_vlm_model,
                    timeout=args.vlm_timeout,
                )

            if (
                args.planner_llm_base_url
                and args.planner_llm_api_key
                and args.planner_llm_model
            ):
                context.planner_llm = QwenVLClient(
                    base_url=args.planner_llm_base_url,
                    api_key=args.planner_llm_api_key,
                    model_name=args.planner_llm_model,
                    timeout=args.vlm_timeout,
                )

        planner = PlanningAgent(
            context=context,
            use_llm=args.use_llm_planner,
        )

        graph = planner.plan(
            task_goal=args.goal,
            scene_text=args.scene_text,
            use_vdm=args.use_vdm,
            output_path=args.output_json,
        )

        md = graph.to_markdown()
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(md, encoding="utf-8")

        print("\n=== Plan Graph ===")
        print(md)
        print("\njson:", args.output_json)
        print("markdown:", args.output_md)

    finally:
        if camera_service is not None:
            camera_service.stop()


if __name__ == "__main__":
    main()
