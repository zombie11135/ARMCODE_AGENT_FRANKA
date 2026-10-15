from skill_lib.manipulation.clear_occlusion_to_buffer import (
    ClearOcclusionToBufferSkill,
    _build_foreground_depth_mask,
    _depth_valid_ratio,
    _select_blocker_candidate,
)


def test_detector_geometry_selects_overlapping_blocker_not_target() -> None:
    observation = {
        "selected_instance": {
            "object_id": "target",
            "bbox": [100, 100, 200, 200],
        },
        "grasp_input": {"bbox": [100, 100, 200, 200]},
        "result": {},
    }
    candidates = [
        {"object_id": "target", "bbox": [100, 100, 200, 200]},
        {
            "object_id": "blocker",
            "bbox": [120, 70, 240, 170],
            "source": "blocker_detector:medicine_box",
        },
        {"object_id": "far", "bbox": [400, 400, 500, 500]},
    ]

    blocker = _select_blocker_candidate(
        observation=observation,
        candidates=candidates,
        depth=None,
        manual_bbox=None,
        search_expand_ratio=1.8,
        min_area_ratio=0.5,
        depth_closer_margin_m=0.01,
        allow_unlocalized_target=False,
    )

    assert blocker is not None
    assert blocker["object_id"] == "blocker"
    assert blocker["source"] == "blocker_detector:medicine_box"


def test_unlocalized_target_does_not_clear_by_default() -> None:
    blocker = _select_blocker_candidate(
        observation={"result": {}},
        candidates=[{"object_id": "box", "bbox": [10, 10, 100, 100]}],
        depth=None,
        manual_bbox=None,
        search_expand_ratio=1.8,
        min_area_ratio=0.5,
        depth_closer_margin_m=0.01,
        allow_unlocalized_target=False,
    )

    assert blocker is None


def test_vlm_description_can_ground_verified_unlocalized_blocker() -> None:
    blocker = _select_blocker_candidate(
        observation={"result": {}},
        candidates=[
            {
                "object_id": "obj_1",
                "bbox": [330, 250, 800, 710],
                "confidence": 0.72,
                "source": "vlm_blocker_description",
            },
            {
                "object_id": "obj_2",
                "bbox": [109, 61, 1169, 656],
                "confidence": 0.85,
                "source": "vlm_blocker_description",
            },
        ],
        depth=None,
        manual_bbox=None,
        search_expand_ratio=1.8,
        min_area_ratio=0.5,
        depth_closer_margin_m=0.01,
        allow_unlocalized_target=False,
        image_size=[1280, 720],
        max_image_area_ratio=0.5,
        vlm_occlusion={
            "target_occluded": True,
            "blocking_object_visible": True,
            "blocking_object_description": "large white and green medicine box",
            "confidence": 0.99,
        },
    )

    assert blocker is not None
    assert blocker["object_id"] == "obj_1"
    assert blocker["selection_mode"] == "vlm_description_grounded_unlocalized"


def test_localized_selection_rejects_whole_scene_bbox_and_ignores_prompt_local_ids() -> None:
    observation = {
        "selected_instance": {"object_id": "obj_1", "bbox": [360, 214, 582, 499]},
        "grasp_input": {"bbox": [360, 214, 582, 499]},
        "result": {},
    }
    candidates = [
        {
            "object_id": "obj_1",
            "bbox": [330, 250, 800, 710],
            "confidence": 0.65,
            "source": "vlm_blocker_description",
        },
        {
            "object_id": "obj_2",
            "bbox": [109, 61, 1169, 656],
            "confidence": 0.90,
            "source": "blocker_detector:package",
        },
    ]

    blocker = _select_blocker_candidate(
        observation=observation,
        candidates=candidates,
        depth=None,
        manual_bbox=None,
        search_expand_ratio=1.8,
        min_area_ratio=0.5,
        depth_closer_margin_m=0.01,
        allow_unlocalized_target=False,
        image_size=[1280, 720],
        max_image_area_ratio=0.5,
        max_target_area_ratio=6.0,
    )

    assert blocker is not None
    assert blocker["bbox"] == [330, 250, 800, 710]


def test_depth_valid_ratio_detects_d435_invalid_roi() -> None:
    import numpy as np

    depth = np.zeros((100, 100), dtype=np.float32)
    depth[20:30, 20:30] = 0.30

    ratio = _depth_valid_ratio(depth, [0, 0, 100, 100])

    assert ratio == 0.01


def test_foreground_depth_mask_keeps_only_nearest_connected_surface() -> None:
    import tempfile

    import numpy as np

    depth = np.full((100, 100), 0.40, dtype=np.float32)
    depth[20:80, 25:75] = 0.30
    depth[5:15, 5:15] = 0.33

    with tempfile.TemporaryDirectory() as output_dir:
        result = _build_foreground_depth_mask(
            depth=depth,
            bbox=[0, 0, 100, 100],
            output_dir=output_dir,
            depth_band_m=0.018,
        )

    assert result is not None
    assert result["support_ratio"] == 0.30
    assert result["near_depth_m"] < 0.31


def test_execute_requires_gripper_callbacks_before_perception() -> None:
    skill = ClearOcclusionToBufferSkill(context=None)

    result = skill.run(
        target_name="西瓜霜",
        observation={},
        execute=True,
        gripper_open=None,
        gripper_close=None,
    )

    assert result.success is False
    assert result.data["failure_type"] == "occlusion_recovery_gripper_unavailable"
