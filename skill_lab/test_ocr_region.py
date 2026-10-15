# skill_lab/test_ocr_region.py

from core.context import RuntimeContext
from services.ocr_service import PaddleOCRService
from skill_lib.perception.ocr_region import OCRRegionSkill


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
    ocr_service = PaddleOCRService(
    lang="ch",
    device="cpu",
    use_textline_orientation=True,
)

    context = RuntimeContext(
        ocr=ocr_service,
        event_callback=print_event,
    )

    skill = OCRRegionSkill(context)

    # 改成你实际生成的 crop_path
    image_path = "runs/current_capture/sam3_gdino_segment_01_obj_1_crop.png"

    result = skill.run(
        image_path=image_path,
        min_confidence=0.0,
        output_prefix="runs/current_capture/ocr_test",
    )

    print("success:", result.success)
    print("message:", result.message)
    print("error:", result.error)

    if result.success:
        print("texts:", result.data["texts"])
        print("text:", result.data["text"])
        print("avg_confidence:", result.data["avg_confidence"])
        print("vis_path:", result.data["vis_path"])


if __name__ == "__main__":
    main()