"""Object-profile and size-budget contracts; no robot or rendering required."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from robocasa_occlusion.assets import generate_assets, top_grasp_size_screen


@pytest.mark.parametrize("profile,within_budget", [("graspable", True), ("legacy_wide", False)])
def test_compiled_asset_dimensions_match_profile_and_horizontal_jaw_budget(tmp_path, profile, within_budget):
    assets = generate_assets(tmp_path / profile, object_profile=profile)
    screen = top_grasp_size_screen(assets)
    assert screen["all_objects_within_aperture_budget"] is within_budget
    assert screen["executed_grasp_verified"] is False
    for asset in [*assets["cartons"], *assets["blockers"].values()]:
        geom = ET.parse(asset["path"]).find(".//geom[@name='collision']")
        np.testing.assert_allclose(2 * np.fromstring(geom.attrib["size"], sep=" "), asset["size_m"])
    if profile == "legacy_wide":
        # A 5 cm thickness must not make the 16 cm horizontal width pass a top-grasp screen.
        assert screen["objects"]["blocker_large"]["grasp_width_m"] == .16
        assert screen["objects"]["blocker_large"]["within_aperture_budget"] is False


def test_legacy_asset_directory_cannot_be_silently_resized(tmp_path):
    assets = generate_assets(tmp_path, object_profile="legacy_wide")
    assets.pop("object_profile")  # the format used by the original scene exports
    (tmp_path / "assets.json").write_text(json.dumps(assets))
    model = Path(assets["blockers"]["large"]["path"])
    before = model.read_bytes()
    with pytest.raises(ValueError, match="separate asset directory"):
        generate_assets(tmp_path, object_profile="graspable")
    assert model.read_bytes() == before
