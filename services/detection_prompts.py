"""Shared prompt text, importable without loading detection model dependencies."""

MEDICINE_BOX_PHRASES = (
    "rectangular medicine carton",
    "pharmaceutical paper box",
    "medicine package with printed text",
    "drug package box",
    "medicine bag",
    "medicine package",
)


def single_detection_prompt(prompts):
    """Combine legacy prompt lists into one text input, expanding medicine_box."""
    clean = []
    for prompt in prompts:
        prompt = str(prompt).strip().lower().strip(". ")
        for phrase in MEDICINE_BOX_PHRASES if prompt == "medicine_box" else (prompt,):
            if phrase and phrase not in clean:
                clean.append(phrase)
    if not clean:
        raise ValueError("target_prompts must contain a nonempty prompt")
    return ". ".join(clean) + "."
