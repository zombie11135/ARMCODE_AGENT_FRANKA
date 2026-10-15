"""Offline pivot-calibration audit; this module never connects to xArm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import (
    fit_pivot_tcp,
    load_json,
    pose_mmrad_to_transform,
    rotation_checks,
    write_json,
)


def run(manifest_path: Path, output_dir: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("example_only", False):
        raise ValueError("example_only=true: replace every teaching pose and set it to false")
    if manifest.get("pose_semantics") != "T_base_flange_mmrad":
        raise ValueError("pose_semantics must be T_base_flange_mmrad")
    teaching = manifest.get("teaching_poses_mmrad") or []
    transforms = [pose_mmrad_to_transform(item["pose"]) for item in teaching]
    fit = fit_pivot_tcp(transforms)

    thresholds = manifest.get("acceptance", {})
    max_rms = float(thresholds.get("maximum_rms_residual_m", 0.002))
    max_error = float(thresholds.get("maximum_single_residual_m", 0.004))
    min_spread = float(thresholds.get("minimum_orientation_spread_deg", 30.0))
    position_passed = bool(
        fit["linear_system_rank"] == 6
        and fit["rms_residual_m"] <= max_rms
        and fit["max_residual_m"] <= max_error
        and fit["maximum_pairwise_orientation_change_deg"] >= min_spread
    )

    rpy = manifest.get("tcp_orientation_rpy_rad")
    transform = None
    if rpy is not None:
        transform = pose_mmrad_to_transform(
            [
                fit["tcp_translation_flange_m"][0] * 1000.0,
                fit["tcp_translation_flange_m"][1] * 1000.0,
                fit["tcp_translation_flange_m"][2] * 1000.0,
                *[float(value) for value in rpy],
            ]
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(output_dir / "T_flange_tcp.npy"), transform)
    else:
        position_only = np.eye(4, dtype=np.float64)
        position_only[:3, 3] = fit["tcp_translation_flange_m"]
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(output_dir / "T_flange_tcp_position_only.npy"), position_only)

    report = {
        "status": "complete" if position_passed and transform is not None else "position_only_or_failed",
        "offline_only": True,
        "robot_connection_attempted": False,
        "controller_configuration_written": False,
        "manifest": str(manifest_path),
        "teaching_pose_count": len(teaching),
        "fit": {
            "tcp_translation_flange_m": fit["tcp_translation_flange_m"].tolist(),
            "tcp_translation_flange_mm": (1000.0 * fit["tcp_translation_flange_m"]).tolist(),
            "pivot_point_base_m": fit["pivot_point_base_m"].tolist(),
            "residuals_m": fit["residuals_m"].tolist(),
            "rms_residual_m": fit["rms_residual_m"],
            "max_residual_m": fit["max_residual_m"],
            "linear_system_rank": fit["linear_system_rank"],
            "linear_system_condition": fit["linear_system_condition"],
            "maximum_pairwise_orientation_change_deg": fit[
                "maximum_pairwise_orientation_change_deg"
            ],
        },
        "acceptance": {
            "thresholds": {
                "maximum_rms_residual_m": max_rms,
                "maximum_single_residual_m": max_error,
                "minimum_orientation_spread_deg": min_spread,
            },
            "position_passed": position_passed,
            "orientation_provided": transform is not None,
            "ready_for_planning": bool(position_passed and transform is not None),
        },
        "transform": None if transform is None else transform.tolist(),
        "rotation_check": None if transform is None else rotation_checks(transform),
        "warning": (
            "Pivot fitting estimates the TCP origin only. TCP orientation must come from the GT200 "
            "mechanical axes or a separate orientation calibration. Review the report before any "
            "set_tcp_offset call; this tool never writes the controller."
        ),
    }
    write_json(output_dir / "tcp_calibration_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit xArm/GT200 pivot teaching poses offline")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
