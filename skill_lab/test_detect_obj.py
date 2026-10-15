# skill_lab/test_detect_object.py

import time

from core.context import RuntimeContext
from services.realsense_dual_camera_service import (
    RealSenseDualCameraService,
    CameraConfig,
)
from services.grounding_dino_detector_service import GroundingDINODetectorService
from skill_lib.perception.detect_obj import DetectObjectSkill


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

    detector_service = GroundingDINODetectorService(
        model_name_or_path="/home/wpy/models/grounding-dino-tiny",
        device="cuda",
        threshold=0.30,
        text_threshold=0.25,
        nms_iou_threshold=0.50,
        local_files_only=True,
        use_fp16=False,
    )

    try:
        camera_service.start()
        time.sleep(2.0)

        context = RuntimeContext(
            camera=camera_service,
            detector=detector_service,
            event_callback=print_event,
        )

        detect_skill = DetectObjectSkill(context)

        print("\n=== Test medicine_box ===")
        result = detect_skill.run(
            target_name="medicine_box",
            camera_name="wrist",
            threshold=0.35,
            text_threshold=0.35
        )

        print_result(result)

        print("\n=== Test fruit ===")
        result = detect_skill.run(
            target_name="fruit",
            camera_name="wrist",
            threshold=0.35,
            text_threshold=0.35,
        )

        print_result(result)

        print("\n=== Test snack ===")
        result = detect_skill.run(
            target_name="snack",
            camera_name="wrist",
            threshold=0.35,
            text_threshold=0.35,
        )

        print_result(result)

        print("\n按 Enter 继续交互测试，Ctrl+C 退出。")

        while True:
            input()

            target = input("请输入目标类别 medicine_box / fruit / apple / orange / snack: ").strip()
            if not target:
                target = "medicine_box"

            result = detect_skill.run(
                target_name=target,
                camera_name="wrist",
            )

            print_result(result)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        camera_service.stop()


def print_result(result):
    print("success:", result.success)
    print("message:", result.message)
    print("error:", result.error)

    if result.success:
        print("text_prompt:", result.data.get("text_prompt"))
        print("vis_path:", result.data.get("vis_path"))
        print("detections:")
        for det in result.data["detections"]:
            print(det)


if __name__ == "__main__":
    main()