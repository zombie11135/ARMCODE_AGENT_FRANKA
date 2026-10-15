"""Pure container-name, colour and placement-slot helpers."""
from __future__ import annotations

from typing import Dict, Optional


def _looks_like_container(text: str) -> bool:
    lower = str(text or "").lower()
    # Medicine packaging is the object to identify/grasp, not a destination.
    # Keep explicit storage/tray targets in the container path.
    storage = ('收纳盒', '整理盒', '储物盒', '托盘', '盘子', '碗', '容器',
               'tray', 'plate', 'bowl', 'container', 'storage box', 'organizer')
    if any(token in lower for token in storage):
        return True
    if any(token in lower for token in ('药盒', '药品', '药物', '药包装',
                                        'medicine', 'medication', 'drug', 'pill', 'tablet')):
        return False
    return any(token in lower for token in ["盘", "托盘", "盘子", "碗", "盒", "容器", "tray", "plate", "bowl", "container", "box"])


def parse_container_spec(text: str) -> Dict[str, Optional[str]]:
    raw = str(text or "").strip()
    color = parse_color(raw)
    name = parse_container_name(raw)
    return {"raw": raw, "name": name, "color": color}


def parse_color(text: str) -> Optional[str]:
    if any(token in str(text or "").lower() for token in ["透明", "半透明", "transparent", "translucent", "clear"]):
        return None
    aliases = {
        "pink": ["粉色", "粉红色", "粉红", "pink"],
        "blue": ["蓝色", "blue"],
        "green": ["绿色", "green"],
        "red": ["红色", "red"],
        "yellow": ["黄色", "yellow"],
        "orange": ["橙色", "orange"],
        "purple": ["紫色", "purple"],
        "white": ["白色", "white"],
        "black": ["黑色", "black"],
        "gray": ["灰色", "grey", "gray"],
        "brown": ["棕色", "褐色", "brown"],
    }
    lower = str(text or "").lower()
    for canonical, names in aliases.items():
        if any(name in lower for name in names):
            return canonical
    return None


def parse_container_name(text: str) -> str:
    lower = str(text or "").lower()
    if any(token in lower for token in ["碗", "bowl"]):
        return "bowl"
    if any(token in lower for token in ["收纳盒", "整理盒", "透明盒", "半透明盒", "盒", "容器", "box", "container", "bin", "organizer"]):
        return "container"
    if any(token in lower for token in ["盘", "托盘", "plate", "tray", "dish"]):
        return "tray"
    return "tray"


def container_key(spec: Dict[str, Optional[str]]) -> str:
    name = spec.get("name") or "tray"
    color = spec.get("color")
    return f"{name}__{color}" if color else name


def placement_slot_offset_mm(slot_index: int, spacing_mm: float) -> float:
    if slot_index <= 0:
        return 0.0
    magnitude = (slot_index + 1) // 2
    direction = -1.0 if slot_index % 2 == 1 else 1.0
    return direction * magnitude * float(spacing_mm)
