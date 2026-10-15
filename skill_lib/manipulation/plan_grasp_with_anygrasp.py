# skill_lib/manipulation/plan_grasp_with_anygrasp.py

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import math
import json
import random
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from core.skill_base import BaseSkill, SkillResult
import traceback

class PlanGraspWithAnyGraspSkill(BaseSkill):
    name = "plan_grasp_with_anygrasp"
    description = "使用 AnyGrasp 根据 RGB-D 和目标 mask 生成抓取，并直接转换为 xArm base 坐标系抓取位姿。"

    def run(
        self,
        grasp_input: Dict[str, Any],

        # AnyGrasp 参数
        lims: Optional[List[float]] = None,
        auto_lims_from_mask: bool = True,
        auto_lims_from_bbox: bool = True,
        lims_margin: float = 0.03,
        bbox_margin_px: int = 12,
        use_mask_points: bool = True,
        use_roi_points_for_inference: bool = True,
        min_mask_points: int = 200,
        filter_grasps_by_bbox: bool = False,
        bbox_filter_margin_px: int = 0,
        bbox_filter_inner_margin_ratio: float = 0.15,
        fallback_to_roi_points_when_bbox_empty: bool = True,
        visualize_bbox_filtered_grasps: bool = False,
        visualize_bbox_filtered_top_k: int = 100,
        top_k: int = 50,
        apply_object_mask: bool = True,
        dense_grasp: bool = False,
        collision_detection: bool = True,
        visualize_grasps: bool = False,
        visualize_top_k: int = 20,
        visualize_best_only: bool = False,
        visualize_selected_grasp: bool = False,

        # xArm / 手眼参数
        xarm_ip: str = "192.168.1.237",
        current_ee_pose_mrad: Optional[List[float]] = None,
        handeye_config_path: str = "/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
        handeye_rot: Optional[List[List[float]]] = None,
        handeye_trans: Optional[List[float]] = None,

        # 坐标转换 / 选择策略
        gripper_length: float = 0.12,
        target_rpy: Tuple[float, float, float] = (3.14, 0.0, -1.57),
        target_rx: float = math.pi,
        target_ry: float = 0.0,
        max_rx_tilt_deg: float = 65.0,
        max_ry_tilt_deg: float = 65.0,
        lam_rx: float = 1.0,
        lam_ry: float = 1.0,
        enable_pose_fix: bool = True,
        grasp_selection_mode: str = "random_top_k_good",
        random_select_top_k: int = 5,
        random_seed: Optional[int] = None,
        fallback_to_raw_when_no_good: bool = False,

        # 预抓取 / 抬升
        approach_clearance_mm: float = 80.0,
        lift_after_grasp_mm: float = 120.0,

        **kwargs,
    ) -> SkillResult:
        try:
            anygrasp = self.context.require("anygrasp")

            color_path = grasp_input.get("color_path") or grasp_input.get("image_path")
            depth_path = grasp_input.get("depth_path")
            depth_npy_path = grasp_input.get("depth_npy_path")
            mask_path = grasp_input.get("mask_path")
            mask_npy_path = grasp_input.get("mask_npy_path")
            bbox = grasp_input.get("bbox")

            intrinsics = (
                grasp_input.get("depth_intrinsics")
                or grasp_input.get("color_intrinsics")
                or grasp_input.get("camera_intrinsics")
            )

            intrinsics = self._normalize_intrinsics(intrinsics)

            if intrinsics is None:
                return SkillResult(
                    success=False,
                    data=grasp_input,
                    error="grasp_input 缺少相机内参 depth_intrinsics / color_intrinsics / camera_intrinsics",
                )

            depth_scale = grasp_input.get("depth_scale", 1000.0)

            backend = getattr(anygrasp, 'backend', None)
            backend_label = 'GSNet' if backend == 'gsnet' else 'AnyGrasp'
            self.context.emit_text(f"正在调用 {backend_label} 生成相机系抓取候选...")

            fr3_args = {}
            if self.context.transforms is not None:
                fr3_args = {key: grasp_input.get(key) for key in ("capture_geometry", "points_path")}
                fr3_args['retain_all_candidates'] = True
                selection_options = dict(target_rx=target_rx, target_ry=target_ry,
                    max_rx_tilt_deg=max_rx_tilt_deg, max_ry_tilt_deg=max_ry_tilt_deg,
                    lam_rx=lam_rx, lam_ry=lam_ry, grasp_selection_mode=grasp_selection_mode,
                    random_select_top_k=random_select_top_k, random_seed=random_seed,
                    fallback_to_raw_when_no_good=fallback_to_raw_when_no_good)
                from hardware.agent_geometry import validate_pose_selection
                validate_pose_selection(**selection_options)
            anygrasp_result = anygrasp.plan_grasp(
                **fr3_args,
                color_path=color_path,
                depth_path=depth_path,
                depth_npy_path=depth_npy_path,
                mask_path=mask_path,
                mask_npy_path=mask_npy_path,
                bbox=bbox,
                intrinsics=intrinsics,
                depth_scale=depth_scale,
                lims=lims,
                auto_lims_from_mask=auto_lims_from_mask,
                auto_lims_from_bbox=auto_lims_from_bbox,
                lims_margin=lims_margin,
                bbox_margin_px=bbox_margin_px,
                use_mask_points=use_mask_points,
                use_roi_points_for_inference=use_roi_points_for_inference,
                min_mask_points=min_mask_points,
                filter_grasps_by_bbox=filter_grasps_by_bbox,
                bbox_filter_margin_px=bbox_filter_margin_px,
                bbox_filter_inner_margin_ratio=bbox_filter_inner_margin_ratio,
                fallback_to_roi_points_when_bbox_empty=fallback_to_roi_points_when_bbox_empty,
                visualize_bbox_filtered_grasps=visualize_bbox_filtered_grasps,
                visualize_bbox_filtered_top_k=visualize_bbox_filtered_top_k,
                top_k=top_k,
                apply_object_mask=apply_object_mask,
                dense_grasp=dense_grasp,
                collision_detection=collision_detection,
                visualize_grasps=visualize_grasps,
                visualize_top_k=visualize_top_k,
                visualize_best_only=visualize_best_only,
                output_json_path="runs/current_capture/anygrasp_result.json",
            )

            if not anygrasp_result.get("success", False):
                return SkillResult(
                    success=False,
                    data=anygrasp_result,
                    error=anygrasp_result.get("error", "AnyGrasp failed."),
                )

            if self.context.transforms is not None:
                from hardware.agent_geometry import grasp_plan, GraspSelectionError
                try:
                    result = grasp_plan(grasp_input, anygrasp_result, self.context.transforms,
                                        anygrasp_result["workcell"], **selection_options)
                except GraspSelectionError as error:
                    if anygrasp_result.get('run_dir'):
                        (Path(anygrasp_result['run_dir']) / 'pose_selection.json').write_text(
                            json.dumps(error.report, indent=2))
                    raise
                anygrasp.save_geometry(result)
                self.context.emit_text(result["choose_reason"])
                if visualize_selected_grasp or visualize_grasps or visualize_bbox_filtered_grasps:
                    self.context.emit_text(
                        "[抓取预览] 显示本次最终选中的一个抓取位姿；关闭 Open3D 窗口后继续流程。"
                        "若已启用 execute_robot，随后会执行抓取。")
                    self.context.emit_text(
                        "[抓取目标] fr3_link0 中的 fr3_link8 法兰 [x,y,z,rx,ry,rz]（m/rad）："
                        + json.dumps(result['best_grasp_base_mrad']))
                    anygrasp.visualize_last_grasps_by_indices([0])
                return SkillResult(success=True, data=result, message=f"{backend_label} -> FR3 标定抓取位姿已生成")

            # 1. 获取 xArm 当前末端位姿，单位 m/rad
            if current_ee_pose_mrad is None:
                current_ee_pose_mrad = self._get_current_xarm_pose_mrad(
                    xarm_ip=xarm_ip,
                )

            if current_ee_pose_mrad is None:
                return SkillResult(
                    success=False,
                    data=anygrasp_result,
                    error="无法获取 xArm 当前末端位姿 current_ee_pose_mrad。",
                )

            # 2. 获取手眼标定矩阵：相机 -> 末端
            if handeye_rot is None or handeye_trans is None:
                handeye_rot, handeye_trans = self._load_handeye_cam2ee(
                    handeye_config_path,
                )

            handeye_rot = np.asarray(handeye_rot, dtype=float)
            handeye_trans = np.asarray(handeye_trans, dtype=float).reshape(3)

            # 3. 对每个 AnyGrasp candidate 转到 base，按姿态约束重新选择
            candidates_camera = anygrasp_result.get("candidate_grasps_camera", [])
            if not candidates_camera:
                return SkillResult(
                    success=False,
                    data=anygrasp_result,
                    error="AnyGrasp 没有返回 candidate_grasps_camera。",
                )

            converted_candidates = []

            max_rx_tilt = np.deg2rad(max_rx_tilt_deg)
            max_ry_tilt = np.deg2rad(max_ry_tilt_deg)

            for cand in candidates_camera:
                base_pose_mrad = self._camera_grasp_to_base_pose(
                    grasp_translation=cand["translation"],
                    grasp_rotation_mat=cand["rotation_matrix"],
                    current_ee_pose=current_ee_pose_mrad,
                    handeye_rot=handeye_rot,
                    handeye_trans=handeye_trans,
                    gripper_length=gripper_length,
                )

                rx = base_pose_mrad[3]
                ry = base_pose_mrad[4]

                tilt_rx = self._angle_abs_diff(rx, target_rx)
                tilt_ry = self._angle_abs_diff(ry, target_ry)

                raw_score = cand.get("score")
                raw_score = 0.0 if raw_score is None else float(raw_score)

                effective_score = raw_score - lam_rx * tilt_rx - lam_ry * tilt_ry

                xarm_pose_mmrad = self._base_mrad_to_xarm_mmrad(
                    base_pose_mrad,
                    target_rpy=target_rpy,
                    enable_pose_fix=enable_pose_fix,
                )

                converted_candidates.append(
                    {
                        "camera_grasp": cand,
                        "base_pose_mrad": base_pose_mrad,
                        "xarm_pose_mmrad": xarm_pose_mmrad,
                        "raw_score": raw_score,
                        "effective_score": effective_score,
                        "rx": rx,
                        "ry": ry,
                        "tilt_rx": tilt_rx,
                        "tilt_ry": tilt_ry,
                        "tilt_rx_deg": float(np.rad2deg(tilt_rx)),
                        "tilt_ry_deg": float(np.rad2deg(tilt_ry)),
                    }
                )

            good = [
                c for c in converted_candidates
                if c["tilt_rx"] <= max_rx_tilt and c["tilt_ry"] <= max_ry_tilt
            ]

            if good:
                good.sort(key=lambda c: c["effective_score"], reverse=True)
                if grasp_selection_mode == "best_good":
                    choose_pool = good[:1]
                    chosen = good[0]
                    choose_reason = (
                        f"选择满足姿态约束的最高分抓取："
                        f"{len(good)}/{len(converted_candidates)} 个候选满足 "
                        f"rx≤{max_rx_tilt_deg}°, ry≤{max_ry_tilt_deg}°"
                    )
                elif grasp_selection_mode == "random_top_k_good":
                    choose_pool = good[:max(1, min(int(random_select_top_k), len(good)))]
                    chosen = self._random_pick(choose_pool, random_seed=random_seed)
                    choose_reason = (
                        f"在满足姿态约束的前 {len(choose_pool)} 个抓取中随机选择："
                        f"{len(good)}/{len(converted_candidates)} 个候选满足 "
                        f"rx≤{max_rx_tilt_deg}°, ry≤{max_ry_tilt_deg}°"
                    )
                elif grasp_selection_mode == "random_good":
                    choose_pool = good
                    chosen = self._random_pick(choose_pool, random_seed=random_seed)
                    choose_reason = (
                        f"在所有满足姿态约束的 {len(choose_pool)} 个抓取中随机选择："
                        f"{len(good)}/{len(converted_candidates)} 个候选满足 "
                        f"rx≤{max_rx_tilt_deg}°, ry≤{max_ry_tilt_deg}°"
                    )
                else:
                    return SkillResult(
                        success=False,
                        data={
                            **anygrasp_result,
                            "current_ee_pose_mrad": [float(x) for x in current_ee_pose_mrad],
                            "converted_candidates": converted_candidates,
                            "valid_grasp_selection_modes": [
                                "random_good",
                                "random_top_k_good",
                                "best_good",
                            ],
                        },
                        error=f"未知 grasp_selection_mode: {grasp_selection_mode}",
                    )
            else:
                if not fallback_to_raw_when_no_good:
                    return SkillResult(
                        success=False,
                        data={
                            **anygrasp_result,
                            "current_ee_pose_mrad": [float(x) for x in current_ee_pose_mrad],
                            "handeye_rot": handeye_rot.tolist(),
                            "handeye_trans": handeye_trans.tolist(),
                            "converted_candidates": converted_candidates,
                            "max_rx_tilt_deg": max_rx_tilt_deg,
                            "max_ry_tilt_deg": max_ry_tilt_deg,
                        },
                        error=(
                            "没有候选同时满足姿态约束，未进行 raw_score 退化选择。"
                            "如需退化，请设置 fallback_to_raw_when_no_good=True。"
                        ),
                    )

                converted_candidates.sort(key=lambda c: c["raw_score"], reverse=True)
                choose_pool = converted_candidates[:max(
                    1,
                    min(int(random_select_top_k), len(converted_candidates)),
                )]
                chosen = self._random_pick(choose_pool, random_seed=random_seed)
                choose_reason = (
                    f"没有候选同时满足姿态约束，按参数退化为在 raw_score 前 "
                    f"{len(choose_pool)} 个抓取中随机选择。"
                )

            best_xarm_pose = chosen["xarm_pose_mmrad"]

            selected_visualization_error = None
            if visualize_selected_grasp:
                try:
                    chosen_index = chosen["camera_grasp"].get("index")
                    anygrasp.visualize_last_grasps_by_indices(
                        indices=[chosen_index],
                        window_name="Selected AnyGrasp grasp",
                    )
                except Exception as e:
                    selected_visualization_error = str(e)
                    self.context.emit_warning(
                        f"最终选中抓取可视化失败: {selected_visualization_error}"
                    )

            approach_pose = list(best_xarm_pose)
            approach_pose[2] = approach_pose[2] + approach_clearance_mm

            lift_pose = list(best_xarm_pose)
            lift_pose[2] = lift_pose[2] + lift_after_grasp_mm

            result = {
                **anygrasp_result,

                "current_ee_pose_mrad": [float(x) for x in current_ee_pose_mrad],
                "handeye_rot": handeye_rot.tolist(),
                "handeye_trans": handeye_trans.tolist(),

                "best_grasp_camera": chosen["camera_grasp"],
                "best_grasp_base_mrad": chosen["base_pose_mrad"],
                "best_grasp_xarm_mmrad": best_xarm_pose,

                "approach_pose_mmrad": approach_pose,
                "lift_pose_mmrad": lift_pose,

                "chosen_candidate": chosen,
                "grasp_selection_mode": grasp_selection_mode,
                "random_select_top_k": random_select_top_k,
                "random_seed": random_seed,
                "fallback_to_raw_when_no_good": fallback_to_raw_when_no_good,
                "choose_pool_size": len(choose_pool),
                "visualize_selected_grasp": visualize_selected_grasp,
                "selected_visualization_error": selected_visualization_error,
                "converted_candidates": converted_candidates,
                "choose_reason": choose_reason,

                "target_rpy": list(target_rpy),
                "gripper_length": gripper_length,
                "grasp_input": grasp_input,
            }

            self.context.emit_text(choose_reason)
            self.context.emit_text(
                f"AnyGrasp base 位姿生成完成：xArm pose={best_xarm_pose}"
            )

            return SkillResult(
                success=True,
                data=result,
                message="AnyGrasp 已生成 xArm base 坐标系抓取位姿。",
            )

        except Exception as e:
            tb = traceback.format_exc()
            error = f"AnyGrasp base 抓取规划失败: {e}\n{tb}"

            try:
                self.context.emit_error(error)
            except Exception:
                pass

            return SkillResult(
                success=False,
                data={"traceback": tb},
                error=error,
            )

    # ----------------------------------------------------------------------
    # xArm
    # ----------------------------------------------------------------------

    def _get_current_xarm_pose_mrad(self, xarm_ip: str) -> Optional[List[float]]:
        """
        返回:
            [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]
        """
        arm = getattr(self.context, "xarm", None)

        if arm is None:
            from xarm.wrapper import XArmAPI
            arm = XArmAPI(xarm_ip, is_radian=True)
            # 不在这里强制 motion_enable / set_mode，因为这里只读取状态
            self.context.xarm = arm

        # xArm SDK 常见返回格式：code, pose
        # pose = [x_mm, y_mm, z_mm, roll, pitch, yaw]
        try:
            code, pose = arm.get_position(is_radian=True)
        except TypeError:
            code, pose = arm.get_position()

        if code != 0 or pose is None or len(pose) < 6:
            return None

        x_mm, y_mm, z_mm, rx, ry, rz = pose[:6]

        return [
            float(x_mm) / 1000.0,
            float(y_mm) / 1000.0,
            float(z_mm) / 1000.0,
            float(rx),
            float(ry),
            float(rz),
        ]

    # ----------------------------------------------------------------------
    # hand-eye
    # ----------------------------------------------------------------------

    @staticmethod
    def _load_handeye_cam2ee(config_path: str):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到手眼标定文件: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if cfg.get("schema_version") == 1 and "cameras" in cfg:
            from hardware.transforms import CalibratedTransforms
            matrix = CalibratedTransforms(path).cameras["wrist"]["matrix"]
            return matrix[:3, :3], matrix[:3, 3]

        rot = np.asarray(cfg["rotation"], dtype=float)
        trans = np.asarray(cfg["translation"], dtype=float).reshape(3)

        if cfg.get("unit", "meter") in ["millimeter", "mm"]:
            trans = trans / 1000.0

        T = np.eye(4, dtype=float)
        T[:3, :3] = rot
        T[:3, 3] = trans

        frame_type = cfg.get("frame_type", "camera_to_ee")

        if frame_type in ["camera_to_ee", "cam2ee", "T_cam2ee"]:
            T_cam2ee = T
        elif frame_type in ["ee_to_camera", "ee2cam", "T_ee2cam"]:
            T_cam2ee = np.linalg.inv(T)
        else:
            raise ValueError(f"未知 handeye frame_type: {frame_type}")

        return T_cam2ee[:3, :3], T_cam2ee[:3, 3]

    # ----------------------------------------------------------------------
    # coordinate transform: camera grasp -> base pose
    # ----------------------------------------------------------------------

    @staticmethod
    def _camera_grasp_to_base_pose(
        grasp_translation,
        grasp_rotation_mat,
        current_ee_pose,
        handeye_rot,
        handeye_trans,
        gripper_length: float = 0.12,
    ) -> List[float]:
        """
        等价于你之前的 convert_new()：
            AnyGrasp/GraspNet camera grasp
                -> 轴对齐 + gripper_length 补偿
                -> camera_to_ee
                -> ee_to_base
                -> base pose [x_m, y_m, z_m, rx, ry, rz]
        """
        grasp_translation = np.asarray(grasp_translation, dtype=float)
        grasp_rotation_mat = np.asarray(grasp_rotation_mat, dtype=float)

        T_grasp2cam = np.eye(4, dtype=float)
        T_grasp2cam[:3, :3] = grasp_rotation_mat
        T_grasp2cam[:3, 3] = grasp_translation

        # 保留你之前的轴对齐逻辑：
        # GraspNet/AnyGrasp.x -> Robot.z
        R_align = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ], dtype=float)

        R_z_90 = R.from_euler("Z", -np.pi / 2).as_matrix()
        R_align = R_align @ R_z_90

        T_align = np.eye(4, dtype=float)
        T_align[:3, :3] = R_align

        # 保留你之前的 gripper_length 补偿方向
        T_align[:3, 3] = [-gripper_length, 0, 0]

        T_gripper2cam = T_grasp2cam @ T_align

        T_cam2ee = np.eye(4, dtype=float)
        T_cam2ee[:3, :3] = np.asarray(handeye_rot, dtype=float)
        T_cam2ee[:3, 3] = np.asarray(handeye_trans, dtype=float).reshape(3)

        x_ee, y_ee, z_ee, rx_ee, ry_ee, rz_ee = current_ee_pose

        R_ee2base = R.from_euler(
            "ZYX",
            [rz_ee, ry_ee, rx_ee],
            degrees=False,
        ).as_matrix()

        T_ee2base = np.eye(4, dtype=float)
        T_ee2base[:3, :3] = R_ee2base
        T_ee2base[:3, 3] = [x_ee, y_ee, z_ee]

        T_gripper2base = T_ee2base @ (T_cam2ee @ T_gripper2cam)

        final_rot_mat = T_gripper2base[:3, :3]
        final_trans = T_gripper2base[:3, 3]

        final_euler = R.from_matrix(final_rot_mat).as_euler("ZYX", degrees=False)
        base_rz, base_ry, base_rx = final_euler

        return [
            float(final_trans[0]),
            float(final_trans[1]),
            float(final_trans[2]),
            float(base_rx),
            float(base_ry),
            float(base_rz),
        ]

    # ----------------------------------------------------------------------
    # pose post-process: m/rad -> xArm mm/rad
    # ----------------------------------------------------------------------

    @classmethod
    def _base_mrad_to_xarm_mmrad(
        cls,
        base_pose_mrad: List[float],
        target_rpy: Tuple[float, float, float] = (3.14, 0.0, -1.57),
        enable_pose_fix: bool = True,
    ) -> List[float]:
        pose = list(base_pose_mrad)

        pose[0] *= 1000.0
        pose[1] *= 1000.0
        pose[2] *= 1000.0

        if not enable_pose_fix:
            return [float(x) for x in pose]

        pose[3], pose[4], pose[5] = cls._snap_rpy_to_target_keep_geometry(
            pose[3],
            pose[4],
            pose[5],
            target=target_rpy,
            use_flip=True,
        )

        # 保留你之前 fix_result 中对低高度/姿态的保护逻辑
        if(pose[2]<160):
            pose[2] = 158
        return [float(x) for x in pose]

    @staticmethod
    def _normalize_intrinsics(intrinsics):
        if intrinsics is None:
            return None

        def get(keys):
            for k in keys:
                if isinstance(intrinsics, dict) and k in intrinsics:
                    return intrinsics[k]
                if hasattr(intrinsics, k):
                    return getattr(intrinsics, k)
            return None

        fx = get(["fx"])
        fy = get(["fy"])
        cx = get(["cx", "ppx"])
        cy = get(["cy", "ppy"])

        if fx is None or fy is None or cx is None or cy is None:
            return intrinsics

        return {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }

    @staticmethod
    def _angle_abs_diff(a: float, b: float) -> float:
        """
        返回角度 a 和 b 的最小绝对差，考虑 2π 周期。
        """
        d = (a - b + np.pi) % (2 * np.pi) - np.pi
        return float(abs(d))

    @staticmethod
    def _random_pick(candidates: List[Dict[str, Any]], random_seed: Optional[int] = None):
        if not candidates:
            raise ValueError("random pick candidates is empty")

        rng = random.Random(random_seed) if random_seed is not None else random
        return rng.choice(candidates)

    @staticmethod
    def _rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def _ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def _rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    @classmethod
    def _euler_to_R_zyx(cls, rx, ry, rz):
        return cls._rz(rz) @ cls._ry(ry) @ cls._rx(rx)

    @staticmethod
    def _R_to_euler_zyx(Rm):
        if abs(Rm[2, 0]) < 1.0:
            ry = np.arcsin(-Rm[2, 0])
            rx = np.arctan2(Rm[2, 1], Rm[2, 2])
            rz = np.arctan2(Rm[1, 0], Rm[0, 0])
        else:
            ry = np.pi / 2 if Rm[2, 0] <= -1 else -np.pi / 2
            rx = 0.0
            rz = np.arctan2(-Rm[0, 1], -Rm[0, 2])
        return float(rx), float(ry), float(rz)

    @staticmethod
    def _wrap_to_target(a, t):
        return a + 2 * np.pi * np.round((t - a) / (2 * np.pi))

    @classmethod
    def _snap_rpy_to_target_keep_geometry(
        cls,
        rx,
        ry,
        rz,
        target=(3.14, 0.0, -1.57),
        use_flip=True,
    ):
        Rm = cls._euler_to_R_zyx(rx, ry, rz)

        cands = [Rm]
        if use_flip:
            cands.append(Rm @ np.diag([-1.0, -1.0, 1.0]))

        best = None
        best_err = 1e9
        tx, ty, tz = target

        for C in cands:
            rxi, ryi, rzi = cls._R_to_euler_zyx(C)

            rxi = cls._wrap_to_target(rxi, tx)
            ryi = cls._wrap_to_target(ryi, ty)
            rzi = cls._wrap_to_target(rzi, tz)

            err = abs(rxi - tx) + abs(ryi - ty) + abs(rzi - tz)

            if err < best_err:
                best_err = err
                best = (float(rxi), float(ryi), float(rzi))

        return best
