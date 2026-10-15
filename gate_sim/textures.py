"""Small deterministic package textures for the simulation-only scene.

The files are generated locally with Pillow so the environment remains
self-contained and does not need a large object-asset download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont


TEXTURE_VERSION = "procedural-daily-products-v1"


_PALETTES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int], str]] = {
    "oat_box": ((190, 60, 128), (245, 164, 189), "OATS"),
    "orange_can": ((232, 145, 34), (255, 175, 110), "SODA"),
    "purple_cup": ((155, 80, 180), (232, 215, 247), "CUP"),
    "cyan_bottle": ((27, 172, 185), (220, 248, 244), "WATER"),
}


def ensure_texture(texture_dir: Path, name: str, size: int = 256) -> Path:
    """Create one repeatable, trademark-free RGB texture and return its path."""

    if name not in _PALETTES:
        raise KeyError("unknown procedural texture {!r}".format(name))
    texture_dir.mkdir(parents=True, exist_ok=True)
    path = texture_dir / "{}_{}.png".format(name, TEXTURE_VERSION)
    if path.exists():
        return path

    base, accent, label = _PALETTES[name]
    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image)
    # Broad asymmetric bands make viewpoint changes visible even at 192x144.
    draw.rectangle((0, 0, size - 1, size // 7), fill=accent)
    draw.rectangle((0, 5 * size // 7, size - 1, size - 1), fill=accent)
    for x in range(-size, 2 * size, size // 4):
        draw.polygon(
            [(x, size // 7), (x + size // 10, size // 7),
             (x + size // 2, 5 * size // 7), (x + 2 * size // 5, 5 * size // 7)],
            fill=tuple(max(0, channel - 32) for channel in base),
        )
    draw.rounded_rectangle(
        (size // 5, size // 3, 4 * size // 5, 2 * size // 3),
        radius=size // 24,
        fill=(245, 245, 238),
        outline=(38, 38, 42),
        width=max(2, size // 64),
    )
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        ((size - text_width) // 2, (size - text_height) // 2),
        label,
        font=font,
        fill=(24, 24, 28),
    )
    image.save(path)
    return path
