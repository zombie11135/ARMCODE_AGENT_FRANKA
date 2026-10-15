import numpy as np

from perception_v2.geometry import build_instance_geometry, mask_iou
from perception_v2.retrieval import snapshot_identity


def test_depth_cleaning_removes_disconnected_background_leak() -> None:
    depth = np.zeros((12, 14), np.float32)
    depth[3:9, 3:9] = 0.40
    depth[1:3, 11:13] = 0.75
    mask = depth > 0
    color = np.zeros((12, 14, 3), np.uint8)
    result = build_instance_geometry(
        color,
        depth,
        mask,
        {"fx": 100.0, "fy": 100.0, "cx": 7.0, "cy": 6.0},
        erosion_px=0,
    )
    assert result["quality"]["clean_mask_pixels"] == 36
    assert result["camera_cloud_stats"]["point_count"] == 36
    assert not result["visible_surface_mask"][1, 11]


def test_mask_iou_resizes_nearest_and_is_bounded() -> None:
    first = np.zeros((8, 8), np.uint8)
    first[2:6, 2:6] = 1
    second = first.copy()
    assert mask_iou(first, second) == 1.0
    assert 0.0 <= mask_iou(first, np.zeros((4, 4), np.uint8)) <= 1.0


def test_snapshot_identity_prefers_recorded_query_then_legacy_directory_name() -> None:
    assert snapshot_identity("20260520_131624_西瓜霜", {"query": "蒙脱石散"}) == "蒙脱石散"
    assert snapshot_identity("20260520_131624_西瓜霜", {"query": None}) == "西瓜霜"
