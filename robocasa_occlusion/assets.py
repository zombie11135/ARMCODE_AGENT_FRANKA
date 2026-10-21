"""Generate exact Chinese packaging and simple dynamic MJCF objects locally."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


LABELS = ("布洛芬", "维生素", "复方片")
DEFAULT_OBJECT_PROFILE = "graspable"
# Full dimensions, metres. "graspable" names a size-screened scene profile,
# not a claim that a grasp controller has already executed these grasps.
OBJECT_PROFILES = {
    "graspable": {
        "carton": (0.09, 0.045, 0.025),
        "blockers": {"small": (0.08, 0.03, 0.025), "large": (0.12, 0.06, 0.04)},
    },
    "legacy_wide": {
        "carton": (0.12, 0.075, 0.035),
        "blockers": {"small": (0.105, 0.045, 0.025), "large": (0.19, 0.16, 0.05)},
    },
}


def top_grasp_size_screen(assets, nominal_opening_m=.08, clearance_each_side_m=.01):
    """Evaluator-only size eligibility; not IK, collision or grasp-stability validation."""
    if not 0 <= clearance_each_side_m < nominal_opening_m / 2:
        raise ValueError("clearance must leave a positive opening budget")
    rows = {}
    objects = [(f"carton_{i}", asset) for i, asset in enumerate(assets["cartons"])]
    objects += [(f"blocker_{name}", asset) for name, asset in assets["blockers"].items()]
    for name, asset in objects:
        x, y, _ = asset["size_m"]
        width = min(x, y)  # overhead jaw closure, not the box's vertical thickness
        required = width + 2 * clearance_each_side_m
        rows[name] = {
            "size_m": list(asset["size_m"]), "closure_axis_object": "x" if x < y else "y",
            "grasp_width_m": width, "required_opening_m": required,
            "within_aperture_budget": required <= nominal_opening_m + 1e-9,
        }
    return {
        "nominal_opening_m": nominal_opening_m, "clearance_each_side_m": clearance_each_side_m,
        "maximum_object_width_m": nominal_opening_m - 2 * clearance_each_side_m,
        "objects": rows,
        "all_objects_within_aperture_budget": all(row["within_aperture_budget"] for row in rows.values()),
        "executed_grasp_verified": False,
        "scope": "top_down_parallel_jaw_width_only_not_ik_collision_or_stability",
    }


def chinese_font(path: str | None = None, size: int = 140):
    candidates = [Path(path)] if path else [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            # Noto's TTC face 2 is Simplified Chinese; Droid has a single face.
            index = 2 if candidate.name.startswith("NotoSansCJK") else 0
            return ImageFont.truetype(str(candidate), size, index=index), candidate
    raise FileNotFoundError("Install fonts-noto-cjk or pass --font /path/to/Chinese.ttf")


def packaging(path: Path, label: str, font_path: str | None = None) -> dict:
    image = Image.new("RGB", (1024, 640), (248, 246, 236))
    draw = ImageDraw.Draw(image)
    font, used_font = chinese_font(font_path)
    small, _ = chinese_font(font_path, 42)
    draw.rectangle((0, 0, 1023, 88), fill=(31, 110, 119))
    draw.text((512, 44), "遮挡识别测试", font=small, anchor="mm", fill="white")
    draw.rounded_rectangle((28, 112, 995, 560), radius=12, outline=(31, 110, 119), width=5)
    draw.text((512, 330), label, font=font, anchor="mm", fill=(24, 36, 43))
    draw.text((512, 594), "仿真样品 · 非真实药品", font=small, anchor="mm", fill=(75, 83, 87))
    image.save(path)
    bbox = list(draw.textbbox((512, 330), label, font=font, anchor="mm"))
    return {"text": label, "text_bbox_px": bbox, "image_size": [1024, 640],
            "font": str(used_font), "texture_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_box(path: Path, size: tuple, texture: Path | None = None,
              color: str = "0.18 0.25 0.32 1") -> None:
    """Use a box for contact and a thin closed UV mesh for the printed top face."""
    x, y, z = [v / 2 for v in size]
    root = ET.Element("mujoco", model="occlusion_object")
    asset = ET.SubElement(root, "asset")
    if texture:
        ET.SubElement(asset, "texture", name="print", type="2d", file=str(texture.resolve()))
        ET.SubElement(asset, "material", name="paper", texture="print", specular="0", shininess="0")
        # Thin cuboid avoids zero-volume mesh errors. Each face has its own vertices
        # and UVs. Only the top carries useful text; the sides are 0.1 mm thick.
        mesh = path.with_name("label.obj")
        verts = [(-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0),
                 (-x, -y, -0.0001), (x, -y, -0.0001),
                 (x, y, -0.0001), (-x, y, -0.0001)]
        lines = [f"v {a} {b} {c}" for a, b, c in verts]
        lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
        lines += ["f 1/1 2/2 3/3", "f 1/1 3/3 4/4", "f 8/1 7/2 6/3", "f 8/1 6/3 5/4",
                  "f 5/1 6/2 2/3", "f 5/1 2/3 1/4", "f 6/1 7/2 3/3", "f 6/1 3/3 2/4",
                  "f 7/1 8/2 4/3", "f 7/1 4/3 3/4", "f 8/1 5/2 1/3", "f 8/1 1/3 4/4"]
        mesh.write_text("\n".join(lines) + "\n")
        ET.SubElement(asset, "mesh", name="label_mesh", file=str(mesh.resolve()))
    world = ET.SubElement(root, "worldbody")
    wrapper = ET.SubElement(world, "body")
    body = ET.SubElement(wrapper, "body", name="object")
    size_text = f"{x} {y} {z}"
    ET.SubElement(body, "geom", name="collision", type="box", size=size_text,
                  group="0", density="350", friction="0.8 0.005 0.0001",
                  solref="0.01 1", solimp="0.95 0.99 0.001", rgba="0.5 0.5 0.5 0")
    ET.SubElement(body, "geom", name="shell", type="box", size=size_text,
                  group="1", contype="0", conaffinity="0", mass="0",
                  rgba="0.97 0.96 0.92 1" if texture else color)
    if texture:
        ET.SubElement(body, "geom", name="label", type="mesh", mesh="label_mesh", material="paper",
                      pos=f"0 0 {z + 0.00015}", group="1", contype="0", conaffinity="0", mass="0")
    for name, pos, site_size in [
        ("bottom_site", f"0 0 {-z}", "0.002"),
        ("top_site", f"0 0 {z}", "0.002"),
        ("horizontal_radius_site", f"{x} {y} 0", "0.002"),
    ]:
        ET.SubElement(wrapper, "site", name=name, pos=pos, size=site_size, rgba="0 0 0 0")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def generate_assets(output: Path, font_path: str | None = None,
                    object_profile: str = DEFAULT_OBJECT_PROFILE) -> dict:
    if object_profile not in OBJECT_PROFILES:
        raise ValueError(f"unknown object profile: {object_profile}")
    previous = output / "assets.json"
    if previous.is_file():
        old_profile = json.loads(previous.read_text()).get("object_profile", "legacy_wide")
        if old_profile != object_profile:
            raise ValueError("use a separate asset directory when changing object profile")
    profile = OBJECT_PROFILES[object_profile]
    carton_size = profile["carton"]
    output.mkdir(parents=True, exist_ok=True)
    result = {"object_profile": object_profile, "cartons": [], "blockers": {}}
    for index, label in enumerate(LABELS):
        folder = output / f"carton_{index}"
        folder.mkdir(exist_ok=True)
        info = packaging(folder / "packaging.png", label, font_path)
        write_box(folder / "model.xml", carton_size, folder / "packaging.png")
        result["cartons"].append({**info, "path": str((folder / "model.xml").resolve()), "size_m": carton_size})
    for name, size in profile["blockers"].items():
        folder = output / f"blocker_{name}"
        folder.mkdir(exist_ok=True)
        write_box(folder / "model.xml", size)
        result["blockers"][name] = {"path": str((folder / "model.xml").resolve()), "size_m": size}
    (output / "assets.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
