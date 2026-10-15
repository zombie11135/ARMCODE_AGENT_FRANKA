# skill_lab/test_find_object_by_text.py

import time

from core.context import RuntimeContext

from services.realsense_dual_camera_service import (
    RealSenseDualCameraService,
    CameraConfig,
)

from services.grounding_dino_detector_service import GroundingDINODetectorService
from services.sam3_bbox_segmenter_service import SAM3BBoxSegmenterService
from services.ocr_service import PaddleOCRService

from skill_lib.combound.find_obj_by_text import FindObjectByTextSkill


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
            serial="213522251050",
            color_width=1280,
            color_height=720,
            color_fps=30,
            enable_depth=False,
        ),
        wrist_config=CameraConfig(
            name="wrist",
            serial="243222070053",
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

    segmenter_service = SAM3BBoxSegmenterService(
        model_dir="/home/wpy/models/sam3",
        checkpoint_path=None,
        device="cuda",
        confidence_threshold=0.0,
        use_autocast=True,
        enable_inst_interactivity=True,
        compile_model=False,
    )

    ocr_service = PaddleOCRService(
        lang="ch",
        device="cpu",
        use_textline_orientation=True,
    )

    try:
        camera_service.start()
        time.sleep(2.0)

        context = RuntimeContext(
            camera=camera_service,
            detector=detector_service,
            segmenter=segmenter_service,
            ocr=ocr_service,
            event_callback=print_event,
        )

        skill = FindObjectByTextSkill(context)

        result = skill.run(
            target_name="medicine_box",
            query="抗病毒口服液",
            camera_name="wrist",
            min_score=0.45,
        )

        print("success:", result.success)
        print("message:", result.message)
        print("error:", result.error)

        if result.success:
            selected = result.data["selected_instance"]
            print("\n=== Selected Instance ===")
            print("object_id:", selected.get("object_id"))
            print("bbox:", selected.get("bbox"))
            print("ocr_text:", selected.get("ocr_text"))
            print("crop_path:", selected.get("crop_path"))
            print("mask_path:", selected.get("mask_path"))
            print("overlay_path:", selected.get("overlay_path"))

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        camera_service.stop()


if __name__ == "__main__":
    main()