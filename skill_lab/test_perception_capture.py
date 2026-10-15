# skill_lab/test_perception_capture.py

import time
from core.context import RuntimeContext
from skill_lib.perception.capture import (
    CaptureGlobalImageSkill,
    CaptureWristRGBDSkill,
)
from services.realsense_dual_camera_service import (
    RealSenseDualCameraService,
    CameraConfig,
)


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

    try:
        camera_service.start()

        # 等待相机缓冲几帧
        time.sleep(3.0)

        context = RuntimeContext(
            camera=camera_service,
            event_callback=print_event,
        )

        capture_global = CaptureGlobalImageSkill(context)
        capture_wrist = CaptureWristRGBDSkill(context)

        print("\n=== Test capture_global_image ===")
        result_global = capture_global.run()
        print(result_global)

        print("\n=== Test capture_wrist_rgbd ===")
        result_wrist = capture_wrist.run()
        print(result_wrist)

        print("\n按 Enter 再拍一次，Ctrl+C 退出。")

        while True:
            input()

            result_global = capture_global.run()
            result_wrist = capture_wrist.run()

            print("[Global Result]", result_global.data)
            print("[Wrist Result]", result_wrist.data)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        camera_service.stop()


if __name__ == "__main__":
    main()