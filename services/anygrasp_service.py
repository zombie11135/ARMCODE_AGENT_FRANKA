# services/anygrasp_service.py

import os
import sys
import json
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
from PIL import Image
import open3d as o3d

@contextmanager
def pushd(path: str):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class AnyGraspService:
    """
    简化版 AnyGrasp service。

    重点：
    - AnyGrasp SDK 保持在原始 grasp_detection 目录
    - 初始化时只加载一次权重
    - 推理时直接复用 self.anygrasp
    """

    def __init__(
        self,
        grasp_detection_dir: str,
        checkpoint_path: str,
        max_gripper_width: float = 0.10,
        gripper_height: float = 0.03,
        top_down_grasp: bool = False,
        debug: bool = False,
    ):
        self.grasp_detection_dir = str(Path(grasp_detection_dir).resolve())
        self.checkpoint_path = checkpoint_path

        if self.grasp_detection_dir not in sys.path:
            sys.path.insert(0, self.grasp_detection_dir)

        cfgs = SimpleNamespace(
            checkpoint_path=checkpoint_path,
            max_gripper_width=max(0.0, min(0.1, max_gripper_width)),
            gripper_height=gripper_height,
            top_down_grasp=top_down_grasp,
            debug=debug,
        )

        print("[AnyGrasp] loading in original demo directory:")
        print("  grasp_detection_dir:", self.grasp_detection_dir)
        print("  checkpoint_path:", checkpoint_path)

        with pushd(self.grasp_detection_dir):
            from gsnet import AnyGrasp
            self.anygrasp = AnyGrasp(cfgs)
            self.anygrasp.load_net()

        print("[AnyGrasp] loaded successfully.")

    def plan_grasp(
        self,
        color_path: str,
        depth_path: Optional[str] = None,
        depth_npy_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        mask_npy_path: Optional[str] = None,
        bbox: Optional[List[int]] = None,
        intrinsics: Optional[Dict[str, float]] = None,
        depth_scale: float = 1000.0,
        lims: Optional[List[float]] = None,
        auto_lims_from_mask: bool = True,
        auto_lims_from_bbox: bool = True,
        lims_margin: float = 0.03,
        bbox_margin_px: int = 12,
        top_k: int = 20,
        apply_object_mask: bool = True,
        dense_grasp: bool = False,
        collision_detection: bool = True,
        output_json_path: str = "runs/current_capture/anygrasp_result.json",
        use_mask_points: bool = True,
        use_roi_points_for_inference: bool = True,
        min_mask_points: int = 200,
        filter_grasps_by_bbox: bool = False,
        bbox_filter_margin_px: int = 0,
        bbox_filter_inner_margin_ratio: float = 0.15,
        fallback_to_roi_points_when_bbox_empty: bool = True,
        visualize_bbox_filtered_grasps: bool = False,
        visualize_bbox_filtered_top_k: int = 100,
        visualize_grasps: bool = False,
        visualize_top_k: int = 20,
        visualize_best_only: bool = False,
        
    ) -> Dict[str, Any]:
        """
        color_path:
            wrist_color.png

        depth_path:
            wrist_depth.png，单位通常是 uint16 mm

        depth_npy_path:
            可选，推荐。如果保存的是米制 depth，直接读取。

        mask_path / mask_npy_path:
            可选。用于自动计算 tight lims 和限制点云。

        bbox:
            可选。当 SAM mask 不稳定时，直接用检测 bbox 生成矩形 ROI，
            用于自动计算 tight lims 和限制点云。
        """
        if intrinsics is None:
            raise ValueError("AnyGrasp 需要相机内参 intrinsics: fx, fy, cx, cy")

        colors = np.array(Image.open(color_path).convert("RGB"), dtype=np.float32) / 255.0
        depths = self._load_depth(depth_path, depth_npy_path, depth_scale)

        if colors.shape[:2] != depths.shape[:2]:
            colors = cv2.resize(
                colors,
                (depths.shape[1], depths.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # 先根据深度反投影得到全图 points_map
        points_map = self._depth_to_points(
            depths=depths,
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
        )

        valid_mask = np.isfinite(depths) & (depths > 0.02) & (depths < 1.5)

        target_mask = self._load_mask(mask_path, mask_npy_path)

        if target_mask is not None and target_mask.shape[:2] != depths.shape[:2]:
            target_mask = cv2.resize(
                target_mask.astype(np.uint8),
                (depths.shape[1], depths.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        bbox_mask = None
        if bbox is not None:
            bbox_mask = self._make_bbox_mask(
                bbox=bbox,
                image_width=depths.shape[1],
                image_height=depths.shape[0],
                margin_px=bbox_margin_px,
            )

        roi_mask = target_mask if target_mask is not None else bbox_mask

        # 1. 根据 mask / bbox 自动算 tight lims
        if lims is None:
            if auto_lims_from_mask and target_mask is not None:
                lims = self._compute_lims_from_mask(
                    points_map=points_map,
                    target_mask=target_mask,
                    valid_mask=valid_mask,
                    margin=lims_margin,
                )
            elif auto_lims_from_bbox and bbox_mask is not None:
                lims = self._compute_lims_from_mask(
                    points_map=points_map,
                    target_mask=bbox_mask,
                    valid_mask=valid_mask,
                    margin=lims_margin,
                )
            else:
                lims = [-1, 1, -1, 1, 0.0, 1.0]

        # 2. 是否直接只使用目标 ROI 内点云。ROI 可以来自 SAM mask，也可以来自 bbox。
        #    如果 use_roi_points_for_inference=False，则 AnyGrasp 在全局有效点云上推理，
        #    后续再用 bbox 筛选抓取中心。
        if use_mask_points and use_roi_points_for_inference and roi_mask is not None:
            final_mask = valid_mask & roi_mask
        else:
            final_mask = valid_mask

        points = points_map[final_mask].astype(np.float32)
        point_colors = colors[final_mask].astype(np.float32)

        if len(points) < min_mask_points:
            raise RuntimeError(
                f"目标 ROI 内有效点云太少: {len(points)}，"
                f"请检查 depth 是否 align 到 color，或增大 mask/bbox。"
            )

        print("[AnyGrasp] points:", points.shape)
        print("[AnyGrasp] lims:", lims)

        with pushd(self.grasp_detection_dir):
            gg, cloud = self.anygrasp.get_grasp(
                points,
                point_colors,
                lims=lims,
                apply_object_mask=apply_object_mask,
                dense_grasp=dense_grasp,
                collision_detection=collision_detection,
            )

        if len(gg) == 0:
            return {
                "success": False,
                "error": "No grasp detected.",
                "lims": lims,
            }

        if hasattr(gg, "grasp_group_array"):
            print("[AnyGrasp] grasp_group_array dtype before:", gg.grasp_group_array.dtype)
            gg.grasp_group_array = gg.grasp_group_array.astype(np.float32, copy=False)
            print("[AnyGrasp] grasp_group_array dtype after:", gg.grasp_group_array.dtype)

        try:
            gg = gg.nms().sort_by_score()
        except RuntimeError as e:
            print(f"[AnyGrasp][WARN] nms failed: {e}")
            print("[AnyGrasp][WARN] fallback to sort_by_score only")
            gg = gg.sort_by_score()

        self._last_visualization_data = {
            "gg": gg,
            "points": points,
            "point_colors": point_colors,
        }

        candidate_indices = None
        fallback_to_roi_points_used = False
        if filter_grasps_by_bbox and bbox is not None:
            candidate_indices = self._filter_grasp_indices_by_bbox(
                gg=gg,
                bbox=bbox,
                intrinsics=intrinsics,
                image_width=depths.shape[1],
                image_height=depths.shape[0],
                margin_px=bbox_filter_margin_px,
                inner_margin_ratio=bbox_filter_inner_margin_ratio,
            )

            if not candidate_indices:
                if fallback_to_roi_points_when_bbox_empty and roi_mask is not None:
                    print(
                        "[AnyGrasp][WARN] no grasp center inside bbox; "
                        "fallback to ROI point cloud inference."
                    )

                    fallback_mask = valid_mask & roi_mask
                    fallback_points = points_map[fallback_mask].astype(np.float32)
                    fallback_colors = colors[fallback_mask].astype(np.float32)

                    if len(fallback_points) < min_mask_points:
                        return {
                            "success": False,
                            "error": (
                                "No grasp center projected inside bbox filter, "
                                f"and ROI fallback points too few: {len(fallback_points)}."
                            ),
                            "lims": lims,
                            "bbox": bbox,
                            "bbox_filter_margin_px": bbox_filter_margin_px,
                            "roi_source": "bbox_filter",
                        }

                    fallback_lims = self._compute_lims_from_mask(
                        points_map=points_map,
                        target_mask=roi_mask,
                        valid_mask=valid_mask,
                        margin=lims_margin,
                    )

                    with pushd(self.grasp_detection_dir):
                        gg, cloud = self.anygrasp.get_grasp(
                            fallback_points,
                            fallback_colors,
                            lims=fallback_lims,
                            apply_object_mask=apply_object_mask,
                            dense_grasp=dense_grasp,
                            collision_detection=collision_detection,
                        )

                    if len(gg) == 0:
                        return {
                            "success": False,
                            "error": "No grasp detected after ROI point cloud fallback.",
                            "lims": fallback_lims,
                            "bbox": bbox,
                            "bbox_filter_margin_px": bbox_filter_margin_px,
                            "roi_source": "bbox_fallback_roi",
                        }

                    if hasattr(gg, "grasp_group_array"):
                        gg.grasp_group_array = gg.grasp_group_array.astype(
                            np.float32,
                            copy=False,
                        )

                    try:
                        gg = gg.nms().sort_by_score()
                    except RuntimeError as e:
                        print(f"[AnyGrasp][WARN] fallback nms failed: {e}")
                        print("[AnyGrasp][WARN] fallback to sort_by_score only")
                        gg = gg.sort_by_score()

                    points = fallback_points
                    point_colors = fallback_colors
                    lims = fallback_lims
                    fallback_to_roi_points_used = True

                    candidate_indices = self._filter_grasp_indices_by_bbox(
                        gg=gg,
                        bbox=bbox,
                        intrinsics=intrinsics,
                        image_width=depths.shape[1],
                        image_height=depths.shape[0],
                        margin_px=bbox_filter_margin_px,
                        inner_margin_ratio=bbox_filter_inner_margin_ratio,
                    )

                    if not candidate_indices:
                        return {
                            "success": False,
                            "error": (
                                "ROI point cloud fallback produced grasps, "
                                "but none projected inside the inner bbox filter."
                            ),
                            "lims": lims,
                            "bbox": bbox,
                            "bbox_filter_margin_px": bbox_filter_margin_px,
                            "bbox_filter_inner_margin_ratio": bbox_filter_inner_margin_ratio,
                            "roi_source": "bbox_fallback_roi",
                        }

                    self._last_visualization_data = {
                        "gg": gg,
                        "points": points,
                        "point_colors": point_colors,
                    }
                else:
                    return {
                        "success": False,
                        "error": "No grasp center projected inside bbox filter.",
                        "lims": lims,
                        "bbox": bbox,
                    "bbox_filter_margin_px": bbox_filter_margin_px,
                    "bbox_filter_inner_margin_ratio": bbox_filter_inner_margin_ratio,
                    "roi_source": "bbox_filter",
                }

        bbox_filtered_visualization_error = None
        if visualize_bbox_filtered_grasps and candidate_indices is not None:
            try:
                self._visualize_grasps(
                    gg=gg,
                    points=points,
                    point_colors=point_colors,
                    indices=candidate_indices[:max(1, int(visualize_bbox_filtered_top_k))],
                    window_name="BBox-filtered AnyGrasp grasps",
                )
            except Exception as e:
                bbox_filtered_visualization_error = str(e)
                print(f"[AnyGrasp][WARN] bbox-filtered visualization failed: {e}")

        candidates = self._extract_candidates(
            gg,
            top_k=top_k,
            intrinsics=intrinsics,
            indices=candidate_indices,
        )

        visualization_error = None
        if visualize_grasps:
            try:
                self._visualize_grasps(
                    gg=gg,
                    points=points,
                    point_colors=point_colors,
                    top_k=visualize_top_k,
                    best_only=visualize_best_only,
                )
            except Exception as e:
                visualization_error = str(e)
                print(f"[AnyGrasp][WARN] visualization failed: {e}")

        result = {
            "success": True,
            "best_grasp_camera": candidates[0],
            "candidate_grasps_camera": candidates,
            "num_candidates": len(candidates),
            "lims": lims,
            "color_path": color_path,
            "depth_path": depth_path,
            "depth_npy_path": depth_npy_path,
            "mask_path": mask_path,
            "mask_npy_path": mask_npy_path,
            "bbox": bbox,
            "bbox_margin_px": bbox_margin_px,
            "roi_source": "mask" if target_mask is not None else ("bbox" if bbox_mask is not None else "full_image"),
            "use_roi_points_for_inference": use_roi_points_for_inference,
            "filter_grasps_by_bbox": filter_grasps_by_bbox,
            "bbox_filter_margin_px": bbox_filter_margin_px,
            "bbox_filter_inner_margin_ratio": bbox_filter_inner_margin_ratio,
            "fallback_to_roi_points_when_bbox_empty": fallback_to_roi_points_when_bbox_empty,
            "fallback_to_roi_points_used": fallback_to_roi_points_used,
            "num_bbox_filtered_grasps": len(candidate_indices) if candidate_indices is not None else None,
            "visualize_bbox_filtered_grasps": visualize_bbox_filtered_grasps,
            "visualize_bbox_filtered_top_k": visualize_bbox_filtered_top_k,
            "bbox_filtered_visualization_error": bbox_filtered_visualization_error,
            "visualize_grasps": visualize_grasps,
            "visualize_top_k": visualize_top_k,
            "visualize_best_only": visualize_best_only,
            "visualization_error": visualization_error,
        }

        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        result["result_json_path"] = output_json_path
        return result

    @staticmethod
    def _load_depth(
        depth_path: Optional[str],
        depth_npy_path: Optional[str],
        depth_scale: float,
    ) -> np.ndarray:
        if depth_npy_path:
            depth = np.load(depth_npy_path).astype(np.float32)

            # 如果看起来是 mm，自动转米
            if np.nanmax(depth) > 10:
                depth = depth / 1000.0

            return depth

        if depth_path is None:
            raise ValueError("需要 depth_path 或 depth_npy_path")

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(depth_path)

        depth_raw = depth_raw.astype(np.float32)
        depth_scale = float(depth_scale)

        # 兼容两类约定：
        # - AnyGrasp demo 常用 divisor=1000.0，depth_raw / 1000 得到米
        # - RealSense SDK 返回 scale=0.001，depth_raw * 0.001 得到米
        if 0 < depth_scale < 1:
            return depth_raw * depth_scale

        return depth_raw / depth_scale

    @staticmethod
    def _depth_to_points(depths, fx, fy, cx, cy):
        h, w = depths.shape[:2]
        xmap, ymap = np.arange(w), np.arange(h)
        xmap, ymap = np.meshgrid(xmap, ymap)

        z = depths
        x = (xmap - cx) / fx * z
        y = (ymap - cy) / fy * z

        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def _load_mask(mask_path, mask_npy_path):
        if mask_npy_path:
            mask = np.load(mask_npy_path)
            if mask.ndim == 3:
                mask = mask[..., 0]
            return mask > 0

        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            return mask > 0

        return None

    @staticmethod
    def _make_bbox_mask(
        bbox: List[int],
        image_width: int,
        image_height: int,
        margin_px: int = 0,
    ) -> np.ndarray:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 -= int(margin_px)
        y1 -= int(margin_px)
        x2 += int(margin_px)
        y2 += int(margin_px)

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox for AnyGrasp ROI: {[x1, y1, x2, y2]}")

        mask = np.zeros((image_height, image_width), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask

    @staticmethod
    def _compute_lims_from_mask(points_map, target_mask, valid_mask, margin=0.03):
        mask = target_mask & valid_mask

        pts = points_map[mask]

        if len(pts) < 50:
            raise RuntimeError("mask 内有效点太少，无法计算 AnyGrasp lims")

        xmin, ymin, zmin = np.percentile(pts, 2, axis=0)
        xmax, ymax, zmax = np.percentile(pts, 98, axis=0)

        lims = [
            float(xmin - margin),
            float(xmax + margin),
            float(ymin - margin),
            float(ymax + margin),
            float(max(0.0, zmin - margin)),
            float(zmax + margin),
        ]

        return lims

    @staticmethod
    def _visualize_grasps(
        gg,
        points: np.ndarray,
        point_colors: np.ndarray,
        top_k: int = 20,
        best_only: bool = False,
        indices: Optional[List[int]] = None,
        window_name: str = "AnyGrasp candidates",
    ) -> None:
        if indices is not None:
            valid_indices = [
                int(i) for i in indices
                if i is not None and 0 <= int(i) < len(gg)
            ]
            if not valid_indices:
                raise ValueError(f"No valid grasp indices for visualization: {indices}")
            grippers = []
            for i in valid_indices:
                grippers.extend(gg[i:i + 1].to_open3d_geometry_list())
        else:
            n = max(1, min(int(top_k), len(gg)))
            gg_pick = gg[0:1] if best_only else gg[0:n]
            grippers = gg_pick.to_open3d_geometry_list()

        trans_mat = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        vis_cloud = o3d.geometry.PointCloud()
        vis_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        vis_cloud.colors = o3d.utility.Vector3dVector(point_colors.astype(np.float64))
        vis_cloud.transform(trans_mat)

        for gripper in grippers:
            gripper.transform(trans_mat)

        o3d.visualization.draw_geometries(
            [*grippers, vis_cloud],
            window_name=window_name,
        )

    def visualize_last_grasps_by_indices(
        self,
        indices: List[int],
        window_name: str = "Selected AnyGrasp grasp",
    ) -> None:
        data = getattr(self, "_last_visualization_data", None)
        if not data:
            raise RuntimeError("没有可视化缓存，请先调用 plan_grasp。")

        self._visualize_grasps(
            gg=data["gg"],
            points=data["points"],
            point_colors=data["point_colors"],
            indices=indices,
            window_name=window_name,
        )







    def _filter_grasp_indices_by_bbox(
        self,
        gg,
        bbox: List[int],
        intrinsics: Dict[str, float],
        image_width: int,
        image_height: int,
        margin_px: int = 0,
        inner_margin_ratio: float = 0.0,
    ) -> List[int]:
        x1, y1, x2, y2 = self._expand_and_clip_bbox(
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
            margin_px=margin_px,
        )
        x1, y1, x2, y2 = self._shrink_bbox(
            bbox=[x1, y1, x2, y2],
            image_width=image_width,
            image_height=image_height,
            inner_margin_ratio=inner_margin_ratio,
        )

        indices = []
        translations = getattr(gg, "translations", None)

        for i in range(len(gg)):
            if translations is not None:
                t = np.asarray(translations[i], dtype=float)
            else:
                t = np.asarray(gg[i].translation, dtype=float)

            if t[2] <= 1e-6:
                continue

            u = int(round(float(intrinsics["fx"]) * t[0] / t[2] + float(intrinsics["cx"])))
            v = int(round(float(intrinsics["fy"]) * t[1] / t[2] + float(intrinsics["cy"])))

            if x1 <= u <= x2 and y1 <= v <= y2:
                indices.append(i)

        return indices

    @staticmethod
    def _shrink_bbox(
        bbox: List[int],
        image_width: int,
        image_height: int,
        inner_margin_ratio: float = 0.0,
    ) -> List[int]:
        if inner_margin_ratio <= 0:
            return bbox

        ratio = max(0.0, min(float(inner_margin_ratio), 0.45))
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        w = x2 - x1
        h = y2 - y1

        if w <= 4 or h <= 4:
            return bbox

        dx = int(round(w * ratio))
        dy = int(round(h * ratio))

        nx1 = max(0, min(x1 + dx, image_width - 1))
        ny1 = max(0, min(y1 + dy, image_height - 1))
        nx2 = max(0, min(x2 - dx, image_width - 1))
        ny2 = max(0, min(y2 - dy, image_height - 1))

        if nx2 <= nx1 or ny2 <= ny1:
            return bbox

        return [nx1, ny1, nx2, ny2]

    @staticmethod
    def _expand_and_clip_bbox(
        bbox: List[int],
        image_width: int,
        image_height: int,
        margin_px: int = 0,
    ) -> List[int]:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 -= int(margin_px)
        y1 -= int(margin_px)
        x2 += int(margin_px)
        y2 += int(margin_px)

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox after clipping: {[x1, y1, x2, y2]}")

        return [x1, y1, x2, y2]

    @staticmethod
    def _extract_candidates(gg, top_k, intrinsics, indices=None):
        candidates = []
        if indices is None:
            indices = list(range(len(gg)))
        indices = indices[:min(len(indices), top_k)]

        translations = getattr(gg, "translations", None)
        rotations = getattr(gg, "rotation_matrices", None)
        scores = getattr(gg, "scores", None)
        widths = getattr(gg, "widths", None)
        heights = getattr(gg, "heights", None)
        depths = getattr(gg, "depths", None)

        for i in indices:
            if translations is not None:
                t = np.asarray(translations[i], dtype=float)
                r = np.asarray(rotations[i], dtype=float)
                score = float(scores[i]) if scores is not None else None
                width = float(widths[i]) if widths is not None else None
                height = float(heights[i]) if heights is not None else None
                depth = float(depths[i]) if depths is not None else None
            else:
                g = gg[i]
                t = np.asarray(g.translation, dtype=float)
                r = np.asarray(g.rotation_matrix, dtype=float)
                score = float(getattr(g, "score", 0.0))
                width = float(getattr(g, "width", 0.0))
                height = float(getattr(g, "height", 0.0))
                depth = float(getattr(g, "depth", 0.0))

            u = None
            v = None
            if t[2] > 1e-6:
                u = int(round(intrinsics["fx"] * t[0] / t[2] + intrinsics["cx"]))
                v = int(round(intrinsics["fy"] * t[1] / t[2] + intrinsics["cy"]))

            candidates.append({
                "index": i,
                "translation": t.tolist(),
                "rotation_matrix": r.tolist(),
                "score": score,
                "width": width,
                "height": height,
                "depth": depth,
                "projected_pixel": [u, v] if u is not None else None,
            })

        return candidates
