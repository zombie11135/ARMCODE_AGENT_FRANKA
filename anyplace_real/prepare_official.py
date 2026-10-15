from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from .geometry import cloud_stats, write_json, write_ply


def read_cloud(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if not len(points):
        raise ValueError("empty point cloud: {}".format(path))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage one official AnyPlace RGB-D dataset pair")
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = read_cloud(case_dir / "base_obj_pointcloud.ply")
    target = read_cloud(case_dir / "target_obj_pointcloud.ply")
    np.save(str(output_dir / "base_obj_pcd.npy"), base)
    np.save(str(output_dir / "target_obj_pcd.npy"), target)
    write_ply(output_dir / "base_obj_pcd.ply", base)
    write_ply(output_dir / "target_obj_pcd.ply", target)
    write_ply(output_dir / "local_crop_pcd.ply", base)
    write_json(output_dir / "preprocess_report.json", {
        "status": "prepared_official_sample",
        "source_case": str(case_dir),
        "base_obj_semantics": "AnyPlace parent/support object, not robot base",
        "crop_profile": "full base_obj; no Molmo/task-specific crop applied",
        "coordinate_unit": "meter (confirmed from official point-cloud extents)",
        "base_obj": cloud_stats(base),
        "target": cloud_stats(target),
    })
    print(output_dir / "preprocess_report.json")


if __name__ == "__main__":
    main()

