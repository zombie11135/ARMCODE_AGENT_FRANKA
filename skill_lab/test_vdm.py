# skill_lab/test_describe_scene.py

import time

from core.context import RuntimeContext
from services.realsense_dual_camera_service import (
    RealSenseDualCameraService,
    CameraConfig,
)
from services.vlm_client import QwenVLClient
from skill_lib.perception.vdm import VDMSkill


def print_event(event: dict):
    event_type = event.get("type")

    if event_type == "image":
        print(f"[Image] {event.get('content')}: {event.get('image_path')}")
    elif event_type == "text":
        print(f"[Text] {event.get('content')}")
    elif event_type == "error":
        print(f"[Error] {event.get('content')}")
    else:
        print(event)


def main():
    camera_service = RealSenseDualCameraService(
        global_config=CameraConfig(
            name="global",
            serial='213522251050',
            color_width=1280,
            color_height=720,
            color_fps=30,
            enable_depth=False,
        ),
        wrist_config=CameraConfig(
            name="wrist",
            serial='243222070053',
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

    vlm_client = QwenVLClient(
        base_url="http://10.201.102.163:8001/v1",
        api_key="wpywpywpy",
        model_name="qwen-vl-8b",
        timeout=120,
    )

    try:
        camera_service.start()
        time.sleep(2.0)

        context = RuntimeContext(
            camera=camera_service,
            vlm=vlm_client,
            event_callback=print_event,
        )

        describe_scene = VDMSkill(context)

        print("\n=== Test describe_scene ===")
        result = describe_scene.run()

        print("\n=== SkillResult ===")
        print("success:", result.success)
        print("message:", result.message)
        print("error:", result.error)

        if result.success:
            print("\n=== Scene Text ===")
            print(result.data["scene_text"])

        print("\n按 Enter 再描述一次，Ctrl+C 退出。")

        while True:
            input()
            result = describe_scene.run(use_wrist=True)

            if result.success:
                print("\n=== Scene Text ===")
                print(result.data["scene_text"])
            else:
                print("\n=== Error ===")
                print(result.error)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        camera_service.stop()


if __name__ == "__main__":
    main()