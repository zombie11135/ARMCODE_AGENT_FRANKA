from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def draw_boxes(image_path, results, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    for box, score, label in zip(
        results["boxes"],
        results["scores"],
        results["labels"],
    ):
        x1, y1, x2, y2 = map(int, box.tolist())

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        text = f"{label}: {float(score):.2f}"

        cv2.putText(
            image,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
    return output_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "/home/wpy/models/grounding-dino-tiny"

    image_path = "runs/current_capture/wrist_color.png"
    output_path = "runs/current_capture/grounding_dino_local_test.png"

    text_prompt = (
        "medicine box. drug box.  pharmaceutical box. "
    )

    print(f"[GroundingDINO] device = {device}")
    print(f"[GroundingDINO] model path = {model_id}")

    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True,
    )

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.20,
    text_threshold=0.12,
    target_sizes=[image.size[::-1]],
)[0]

    print("=== Detection Results ===")

    for box, score, label in zip(
        results["boxes"],
        results["scores"],
        results["labels"],
    ):
        print({
            "label": label,
            "score": float(score),
            "box": [round(v, 2) for v in box.tolist()],
        })

    vis_path = draw_boxes(image_path, results, output_path)
    print(f"[GroundingDINO] visualization saved to: {vis_path}")


if __name__ == "__main__":
    main()