from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import math
import time
import copy
import uuid

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from core.skill_base import BaseSkill, SkillResult
from runtime.observation_memory import fresh_skill_camera, memory_for, StaleObservationError
from runtime.perception_pipeline import ordered_bounded_map
from skill_lib.combound.select_obj_by_text import SelectObjectByTextSkill
from skill_lib.evaluation.verify_medicine_ocr import VerifyMedicineOCRSkill
from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill
from skill_lib.perception.seg_obj import SegmentObjectSkill


class FindAndGraspMedicineSkill(BaseSkill):
    name = "find_and_grasp_medicine"
    description = "高级复合技能：稳定识别指定药盒，生成 AnyGrasp 抓取输入，并可选规划/执行抓取。"

    input_schema = {
        "query": "药品名称或包装文字，例如 布洛芬 / 西瓜霜 / 蒙脱石散。",
        "camera_name": "默认 wrist。",
        "target_prompts": "可选。图库分支将描述合并为一次检测；旧 OCR 分支仍逐提示词调用。",
        "min_score": "目标文本/视觉融合最低分数。",
        "use_sam_for_selection": "是否用 SAM 生成候选 mask/crop，默认 False。",
        "use_mask_for_grasp": "是否把 SAM mask 传给 AnyGrasp，默认 False，推荐 bbox ROI。",
        "plan_grasp": "是否调用 AnyGrasp 规划抓取。",
        "execute": "是否执行 approach -> grasp -> close -> lift。",
        "gripper_open": "可选夹爪打开函数。",
        "gripper_close": "可选夹爪闭合函数。",
    }

    output_schema = {
        "selected_instance": "最终选中的药盒实例。",
        "grasp_input": "可直接传给 plan_grasp_with_anygrasp 的输入。",
        "grasp_plan": "可选 AnyGrasp 规划结果。",
        "ranked_candidates": "候选药盒排序明细。",
    }

    DEFAULT_TARGET_PROMPTS = [
        "medicine_box",
        "rectangular medicine carton",
        "pharmaceutical paper box",
        "medicine package with printed text",
        "drug package box",
    ]

    def run(
        self,
        query: str,
        camera_name: str = "wrist",
        target_prompts: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        min_score: float = 0.50,
        max_detections_per_prompt: int = 12,
        max_instances: int = 12,
        detection_threshold: Optional[float] = 0.22,
        text_threshold: Optional[float] = 0.18,
        merge_iou: float = 0.62,
        expand_bbox_px: int = 12,
        ocr_padding_px: int = 24,
        use_sam_for_selection: bool = False,
        use_mask_for_grasp: bool = False,
        bbox_margin_px: int = 16,
        infer_grasps_on_full_cloud: bool = True,
        filter_grasps_by_bbox: bool = True,
        bbox_filter_margin_px: int = 0,
        bbox_filter_inner_margin_ratio: float = 0.15,
        fallback_to_roi_points_when_bbox_empty: bool = True,
        fallback_intrinsics: Optional[Dict[str, float]] = None,
        plan_grasp: bool = True,
        execute: bool = False,
        xarm_ip: str = "192.168.1.237",
        handeye_config_path: str = "/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
        top_k: int = 50,
        lims_margin: float = 0.03,
        min_mask_points: int = 200,
        grasp_selection_mode: str = "random_top_k_good",
        random_select_top_k: int = 5,
        random_seed: Optional[int] = None,
        fallback_to_raw_when_no_good: bool = False,
        grasp_pose_options: Optional[Dict[str, float]] = None,
        visualize_grasps: bool = False,
        visualize_top_k: int = 20,
        visualize_best_only: bool = False,
        visualize_selected_grasp: bool = False,
        visualize_bbox_filtered_grasps: bool = False,
        visualize_bbox_filtered_top_k: int = 100,
        collision_detection: bool = True,
        approach_clearance_mm: float = 80.0,
        lift_after_grasp_mm: float = 120.0,
        move_speed: float = 50,
        move_acc: float = 500,
        gripper_open: Optional[Callable[[], bool]] = None,
        gripper_close: Optional[Callable[[], bool]] = None,
        gripper_close_repeat_count: int = 2,
        gripper_close_repeat_interval_s: float = 0.25,
        gripper_close_wait_s: float = 1.5,
        verify_ocr: bool = True,
        ocr_verify_strict_mode: bool = False,
        ocr_verify_fuzzy_threshold: float = 0.72,
        reobserve_on_grasp_failure: bool = True,
        reobserve_only_when_execute: bool = True,
        reobserve_z_mm: float = 400.0,
        reobserve_wait_s: float = 0.8,
        reobserve_depth_window_px: int = 25,
        prepared_capture: Optional[Dict[str, Any]] = None,
        prepared_observation: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SkillResult:
        try:
            self.context.emit_text(f"开始高级药盒识别抓取：query={query}")

            prepared_node = None
            if prepared_observation is not None:
                from skill_lib.actions.geometry_occlusion import validate_prepared
                prepared_node = validate_prepared(self.context, prepared_observation, query)
                capture = copy.deepcopy(prepared_observation['geometry_capture'])
                reobserve_on_grasp_failure = False
                fallback_to_roi_points_when_bbox_empty = False
            else:
                if prepared_capture is not None and (execute or plan_grasp):
                    raise ValueError('prepared_capture alone is only allowed for recognition, not grasp planning')
                capture = prepared_capture if prepared_capture is not None else self._capture(camera_name)
            image_path = capture["color_path"]
            self.context.emit_image(image_path=image_path, caption="高级药盒识别输入图像")

            recognition = (SkillResult(True, data=copy.deepcopy({k: v for k, v in prepared_observation.items()
                           if k not in ('geometry_capture', 'geometry_occlusion', 'geometry_auxiliary')}))
                           if prepared_observation is not None else self._recognize_image(
                image_path=image_path, query=query,
                target_prompts=target_prompts, aliases=aliases,
                min_score=min_score, max_detections_per_prompt=max_detections_per_prompt,
                max_instances=max_instances, detection_threshold=detection_threshold,
                text_threshold=text_threshold, merge_iou=merge_iou,
                expand_bbox_px=expand_bbox_px, ocr_padding_px=ocr_padding_px,
                use_sam_for_selection=use_sam_for_selection, use_mask_for_grasp=use_mask_for_grasp,
                verify_ocr=verify_ocr, ocr_verify_strict_mode=ocr_verify_strict_mode,
                ocr_verify_fuzzy_threshold=ocr_verify_fuzzy_threshold, **kwargs))
            if not recognition.success:
                return recognition
            selected = recognition.data["selected_instance"]

            grasp_input = self._build_grasp_input(
                capture=capture,
                image_path=image_path,
                selected=selected,
                query=query,
                use_mask_for_grasp=use_mask_for_grasp,
                fallback_intrinsics=fallback_intrinsics,
            )
            if prepared_node is not None:
                grasp_input.update(mask_npy_path=prepared_node['visible_mask_path'],
                                   mask_path=None, bbox=prepared_node['bbox_xyxy'])

            result_data = dict(recognition.data, grasp_input=grasp_input)

            if plan_grasp:
                plan_kwargs = dict(
                    grasp_input=grasp_input,
                    xarm_ip=xarm_ip,
                    handeye_config_path=handeye_config_path,
                    approach_clearance_mm=approach_clearance_mm,
                    lift_after_grasp_mm=lift_after_grasp_mm,
                    auto_lims_from_mask=use_mask_for_grasp,
                    auto_lims_from_bbox=not infer_grasps_on_full_cloud,
                    lims_margin=lims_margin,
                    bbox_margin_px=bbox_margin_px,
                    use_mask_points=True,
                    use_roi_points_for_inference=not infer_grasps_on_full_cloud,
                    filter_grasps_by_bbox=filter_grasps_by_bbox,
                    bbox_filter_margin_px=bbox_filter_margin_px,
                    bbox_filter_inner_margin_ratio=bbox_filter_inner_margin_ratio,
                    fallback_to_roi_points_when_bbox_empty=fallback_to_roi_points_when_bbox_empty,
                    min_mask_points=min_mask_points,
                    top_k=top_k,
                    grasp_selection_mode=grasp_selection_mode,
                    random_select_top_k=random_select_top_k,
                    random_seed=random_seed,
                    fallback_to_raw_when_no_good=fallback_to_raw_when_no_good,
                    apply_object_mask=True,
                    visualize_grasps=visualize_grasps,
                    visualize_top_k=visualize_top_k,
                    visualize_best_only=visualize_best_only,
                    visualize_selected_grasp=visualize_selected_grasp,
                    visualize_bbox_filtered_grasps=visualize_bbox_filtered_grasps,
                    visualize_bbox_filtered_top_k=visualize_bbox_filtered_top_k,
                    collision_detection=collision_detection,
                )
                pose_options = dict(grasp_pose_options or {})
                allowed = {'target_rx', 'target_ry', 'target_rz', 'max_rx_tilt_deg', 'max_ry_tilt_deg',
                           'lam_rx', 'lam_ry', 'lam_rz', 'enable_pose_fix'}
                if set(pose_options) - allowed:
                    raise ValueError(f"Unknown grasp pose options: {sorted(set(pose_options) - allowed)}")
                plan_kwargs.update(pose_options)
                plan_result = PlanGraspWithAnyGraspSkill(self.context).run(**plan_kwargs)
                result_data["grasp_plan"] = plan_result.data

                if not plan_result.success:
                    retry_result = None
                    if reobserve_on_grasp_failure and (
                        execute or not reobserve_only_when_execute
                    ):
                        retry_result = self._reobserve_target_and_retry_grasp(
                            query=query,
                            camera_name=camera_name,
                            selected=selected,
                            capture=capture,
                            target_prompts=target_prompts,
                            aliases=aliases,
                            min_score=min_score,
                            max_detections_per_prompt=max_detections_per_prompt,
                            max_instances=max_instances,
                            detection_threshold=detection_threshold,
                            text_threshold=text_threshold,
                            merge_iou=merge_iou,
                            expand_bbox_px=expand_bbox_px,
                            ocr_padding_px=ocr_padding_px,
                            use_sam_for_selection=use_sam_for_selection,
                            use_mask_for_grasp=use_mask_for_grasp,
                            fallback_intrinsics=fallback_intrinsics,
                            plan_kwargs=plan_kwargs,
                            xarm_ip=xarm_ip,
                            handeye_config_path=handeye_config_path,
                            observe_z_mm=reobserve_z_mm,
                            observe_wait_s=reobserve_wait_s,
                            depth_window_px=reobserve_depth_window_px,
                            move_speed=move_speed,
                            move_acc=move_acc,
                            verify_ocr=verify_ocr,
                            ocr_verify_strict_mode=ocr_verify_strict_mode,
                            ocr_verify_fuzzy_threshold=ocr_verify_fuzzy_threshold,
                            kwargs=kwargs,
                        )
                        result_data["reobserve_retry"] = (
                            retry_result.data if retry_result is not None else None
                        )

                        if retry_result is not None and retry_result.success:
                            result_data.update(retry_result.data)
                            selected = result_data.get("selected_instance", selected)
                            plan_result = SkillResult(
                                success=True,
                                data=result_data["grasp_plan"],
                            )
                            self.context.emit_text("二次观察后成功生成抓取位姿。")
                        else:
                            retry_error = retry_result.error if retry_result else "未执行二次观察重试。"
                            result_data["reobserve_retry_error"] = retry_error

                    if not plan_result.success:
                        result_data.setdefault("failure_type", "grasp_pose_not_found")
                        return SkillResult(
                            success=False,
                            data=result_data,
                            error=f"药盒已识别，但抓取规划失败: {plan_result.error}",
                        )

                if execute:
                    exec_result = self._execute_grasp(
                        plan=plan_result.data,
                        move_speed=move_speed,
                        move_acc=move_acc,
                        gripper_open=gripper_open,
                        gripper_close=gripper_close,
                        gripper_close_repeat_count=gripper_close_repeat_count,
                        gripper_close_repeat_interval_s=gripper_close_repeat_interval_s,
                        gripper_close_wait_s=gripper_close_wait_s,
                    )
                    result_data["execution"] = exec_result

                    if not exec_result.get("success", False):
                        return SkillResult(
                            success=False,
                            data=result_data,
                            error=exec_result.get("error", "抓取执行失败。"),
                        )

            self.context.emit_text(
                "高级药盒识别完成："
                f"object_id={selected.get('object_id')}, "
                f"score={result_data['selection_result'].get('score'):.3f}, "
                f"identity={result_data.get('identity_evidence', {}).get('identity') if result_data.get('identity_evidence') else selected.get('ocr_text')}"
            )

            if selected.get("overlay_path"):
                self.context.emit_image(
                    image_path=selected["overlay_path"],
                    caption=f"最终选择药盒：{query}",
                )

            return SkillResult(
                success=True,
                data=result_data,
                message=f"已稳定识别目标药盒：{query}",
            )

        except Exception as e:
            from hardware.fr3 import MotionError
            from hardware.robotiq import GripperError
            if isinstance(e, (MotionError, GripperError)):
                raise
            error = f"find_and_grasp_medicine 执行失败: {e}"
            try:
                self.context.emit_error(error)
            except Exception:
                pass
            return SkillResult(success=False, data={}, error=error)

    @fresh_skill_camera('grasp_planning_retry')
    def _reobserve_target_and_retry_grasp(
        self,
        query: str,
        camera_name: str,
        selected: Dict[str, Any],
        capture: Dict[str, Any],
        target_prompts: Optional[List[str]],
        aliases: Optional[List[str]],
        min_score: float,
        max_detections_per_prompt: int,
        max_instances: int,
        detection_threshold: Optional[float],
        text_threshold: Optional[float],
        merge_iou: float,
        expand_bbox_px: int,
        ocr_padding_px: int,
        use_sam_for_selection: bool,
        use_mask_for_grasp: bool,
        fallback_intrinsics: Optional[Dict[str, float]],
        plan_kwargs: Dict[str, Any],
        xarm_ip: str,
        handeye_config_path: str,
        observe_z_mm: float,
        observe_wait_s: float,
        depth_window_px: int,
        move_speed: float,
        move_acc: float,
        verify_ocr: bool,
        ocr_verify_strict_mode: bool,
        ocr_verify_fuzzy_threshold: float,
        kwargs: Dict[str, Any],
    ) -> SkillResult:
        try:
            self.context.emit_warning(
                "目标药盒已识别但未生成可用抓取位姿，开始移动到目标上方二次观察。"
            )

            observe_pose = self._build_target_observe_pose(
                capture=capture,
                selected=selected,
                fallback_intrinsics=fallback_intrinsics,
                xarm_ip=xarm_ip,
                handeye_config_path=handeye_config_path,
                observe_z_mm=observe_z_mm,
                depth_window_px=depth_window_px,
            )
            arm = self.context.require("xarm")
            self.context.emit_text(f"移动到目标二次观察位姿：{observe_pose}")

            if not self._move_xarm(arm, observe_pose, move_speed, move_acc):
                return SkillResult(
                    success=False,
                    data={"stage": "reobserve_move", "observe_pose_mmrad": observe_pose},
                    error="移动到目标二次观察位姿失败。",
                )

            if observe_wait_s > 0:
                time.sleep(float(observe_wait_s))

            refined_capture = self._capture(camera_name)
            refined_image_path = refined_capture["color_path"]
            self.context.emit_image(
                image_path=refined_image_path,
                caption="目标二次观察输入图像",
            )

            recognition = self._recognize_image(
                image_path=refined_image_path, query=query,
                target_prompts=target_prompts, aliases=aliases,
                min_score=min_score, max_detections_per_prompt=max_detections_per_prompt,
                max_instances=max_instances, detection_threshold=detection_threshold,
                text_threshold=text_threshold, merge_iou=merge_iou,
                expand_bbox_px=expand_bbox_px, ocr_padding_px=ocr_padding_px,
                use_sam_for_selection=use_sam_for_selection, use_mask_for_grasp=use_mask_for_grasp,
                verify_ocr=verify_ocr, ocr_verify_strict_mode=ocr_verify_strict_mode,
                ocr_verify_fuzzy_threshold=ocr_verify_fuzzy_threshold, **kwargs)
            if not recognition.success:
                recognition.data["observe_pose_mmrad"] = observe_pose
                return recognition
            refined_selected = recognition.data["selected_instance"]

            refined_grasp_input = self._build_grasp_input(
                capture=refined_capture,
                image_path=refined_image_path,
                selected=refined_selected,
                query=query,
                use_mask_for_grasp=use_mask_for_grasp,
                fallback_intrinsics=fallback_intrinsics,
            )
            refined_plan_kwargs = dict(plan_kwargs)
            refined_plan_kwargs["grasp_input"] = refined_grasp_input
            refined_plan = PlanGraspWithAnyGraspSkill(self.context).run(**refined_plan_kwargs)

            data = dict(recognition.data, stage="reobserve_retry",
                        observe_pose_mmrad=observe_pose, grasp_input=refined_grasp_input,
                        grasp_plan=refined_plan.data)

            if not refined_plan.success:
                return SkillResult(
                    success=False,
                    data=data,
                    error=f"二次观察后抓取规划仍失败: {refined_plan.error}",
                )

            return SkillResult(
                success=True,
                data=data,
                message="二次观察后成功生成抓取位姿。",
            )
        except Exception as e:
            from hardware.fr3 import MotionError
            from hardware.robotiq import GripperError
            if isinstance(e, (MotionError, GripperError)):
                raise
            return SkillResult(
                success=False,
                data={"stage": "reobserve_retry_exception"},
                error=f"二次观察抓取恢复失败: {e}",
            )

    def _recognize_image(self, **options):
        from skill_lib.perception.reference_recognition import RecognizeObjectSkill
        from services.object_names import ObjectNameResolver
        started = time.perf_counter()
        fast = None
        references = getattr(self.context, 'object_references', None)
        requested = options['query']
        fast_options = dict(
            segment=bool(options.get("use_sam_for_selection") or options.get("use_mask_for_grasp")),
            detection_threshold=options.get("detection_threshold", 0.22),
            text_threshold=options.get("text_threshold", 0.18), max_instances=options.get("max_instances", 12),
            expand_bbox_px=options.get("expand_bbox_px", 12), target_prompts=options.get("target_prompts"))
        resolution = None
        if references is not None:
            resolver = ObjectNameResolver(references)
            resolution = resolver.lookup(requested)
            if resolution['status'] == 'needs_scene':
                resolution = resolver.resolve_in_image(requested, self.context, options['image_path'],
                                                       **dict(fast_options, segment=False))
                if resolution['status'] != 'resolved':
                    return SkillResult(False, {'failure_type': 'target_name_ambiguous', 'name_resolution': resolution},
                                       error='目标名称仍有歧义，请提供完整名称或临时绑定')
                options['query'] = resolution['object_id']
            if resolution['status'] == 'unknown' and str(requested).startswith('ref_'):
                return SkillResult(False, {'failure_type': 'unknown_object_id', 'name_resolution': resolution},
                                   error=f'图库没有这个物品编号: {requested}')
        # Explicit strict OCR requests retain their original semantics.
        if references is not None and not options.get("ocr_verify_strict_mode") and not options.get('skip_reference_matching'):
            fast = RecognizeObjectSkill(self.context).run(
                image_path=options["image_path"], query=options["query"],
                **fast_options)
            fast.data['name_resolution'] = resolution
            if fast.success:
                # Gate compares evidence.query with the caller's exact target.
                # Keep a shared-brand/alias request attached to its resolved ID.
                evidence = fast.data['identity_evidence']
                evidence['resolved_query'] = evidence['query']
                evidence['query'] = requested
                fast.data['target_identity'] = dict(object_id=evidence['identity_id'],
                    name=evidence['identity'], requested_name=requested)
                return fast
            if references.settings.recognition_policy == "reference_only":
                fast.data.setdefault('failure_type', fast.data.get('stage', 'reference_uncertain'))
                return fast
        legacy_options = dict(options)
        if resolution and resolution['status'] == 'resolved':
            obj = references.resolve(resolution['object_id'])
            legacy_options['query'] = obj['name']
            legacy_options['aliases'] = list(dict.fromkeys([*obj.get('aliases', []), *(options.get('aliases') or [])]))
        result = self._recognize_legacy(**legacy_options)
        result.data['name_resolution'] = resolution
        if resolution and resolution['status'] == 'resolved':
            result.data['target_identity'] = {'object_id': resolution['object_id'],
                                              'name': resolution['canonical_name'], 'requested_name': requested}
        schedule = {"source": "legacy_ocr", "elapsed_s": time.perf_counter() - started,
                    "reference_fallback_reason": fast.data.get("stage") if fast else "disabled_or_strict_ocr"}
        result.data["recognition_schedule"] = schedule
        if fast:
            result.data["reference_attempt"] = fast.data
        self.context.emit_text(f"[recognition] 原识别流程: {schedule['elapsed_s']:.2f}s；"
                               f"原因={schedule['reference_fallback_reason']}")
        return result

    def _recognize_legacy(self, *, image_path, query, target_prompts=None, aliases=None,
                          min_score=0.50, max_detections_per_prompt=12, max_instances=12,
                          detection_threshold=0.22, text_threshold=0.18, merge_iou=0.62,
                          expand_bbox_px=12, ocr_padding_px=24, use_sam_for_selection=False,
                          use_mask_for_grasp=False, verify_ocr=True, ocr_verify_strict_mode=False,
                          ocr_verify_fuzzy_threshold=0.72, **kwargs):
        prompts = self._build_prompts(target_prompts)
        detections = self._detect_with_prompt_fusion(
            image_path=image_path,
            prompts=prompts,
            threshold=detection_threshold,
            text_threshold=text_threshold,
            max_detections_per_prompt=max_detections_per_prompt,
            merge_iou=merge_iou,
        )

        if not detections:
            return SkillResult(
                success=False,
                data={"stage": "detect", "image_path": image_path, "prompts": prompts},
                error=f"没有检测到候选药盒：{query}",
            )

        detections = detections[:max_instances]
        self.context.emit_text(f"多提示词融合后得到 {len(detections)} 个候选 bbox。")

        workers = getattr(self.context, 'perception_workers', 1)
        pipelined = use_sam_for_selection and workers > 1 and len(detections) > 1
        if pipelined:
            instances = self._candidate_pipeline(image_path, detections, query,
                                                  expand_bbox_px, ocr_padding_px, workers, **kwargs)
        elif use_sam_for_selection:
            instances = self._segment_candidates(
                image_path=image_path,
                detections=detections,
                expand_bbox_px=expand_bbox_px,
                **kwargs,
            )
            instances = [inst for inst in instances if inst.get("success")]
        else:
            instances = self._instances_from_detections(
                image_path=image_path,
                detections=detections,
            )

        if not instances:
            return SkillResult(
                success=False,
                data={"stage": "segment", "image_path": image_path, "detections": detections},
                error="没有可用于 OCR/抓取的候选药盒。",
            )

        if not pipelined:
            instances = self._augment_ocr(
                image_path=image_path, instances=instances, query=query, padding=ocr_padding_px)

        select_result = self._select_best_instance(
            instances=instances,
            query=query,
            aliases=aliases,
            min_score=min_score,
        )

        if not select_result.success:
            return SkillResult(
                success=False,
                data={
                    "stage": "select",
                    "image_path": image_path,
                    "detections": detections,
                    "instances": instances,
                    "selection_result": select_result.data,
                },
                error=select_result.error,
            )

        selected = select_result.data["selected_instance"]

        ocr_verify_result = None
        if verify_ocr:
            ocr_verify_result = VerifyMedicineOCRSkill(self.context).run(
                medicine_query=query,
                image_path=image_path,
                bbox=selected.get("bbox"),
                texts=selected.get("ocr_texts", []),
                raw_text=selected.get("ocr_text", ""),
                strict_mode=ocr_verify_strict_mode,
                fuzzy_threshold=ocr_verify_fuzzy_threshold,
                alias_map={query: aliases or []} if aliases else None,
            )
            if not ocr_verify_result.success:
                return SkillResult(
                    success=False,
                    data={
                        "stage": "verify_ocr",
                        "image_path": image_path,
                        "detections": detections,
                        "instances": instances,
                        "selected_instance": selected,
                        "selection_result": select_result.data,
                        "ocr_verification": ocr_verify_result.data,
                    },
                    error=(
                        "OCR 判别器认为目标文字不可靠: "
                        f"{ocr_verify_result.data.get('failure_type')}"
                    ),
                )

        return SkillResult(True, dict(image_path=image_path, prompts=prompts, detections=detections,
            instances=instances, selected_instance=selected, selection_result=select_result.data,
            ocr_verification=ocr_verify_result.data if ocr_verify_result is not None else None,
            ranked_candidates=select_result.data["ranked_candidates"]))

    def _capture(self, camera_name: str) -> Dict[str, Any]:
        camera = self.context.require("camera")
        if camera_name == "wrist":
            return camera.capture_wrist()
        if camera_name == "global":
            return camera.capture_global()
        raise ValueError(f"Unknown camera_name: {camera_name}")

    @classmethod
    def _build_prompts(cls, target_prompts: Optional[List[str]]) -> List[str]:
        prompts = list(target_prompts or cls.DEFAULT_TARGET_PROMPTS)
        clean = []
        for prompt in prompts:
            prompt = str(prompt).strip()
            if prompt and prompt not in clean:
                clean.append(prompt)
        return clean

    def _detect_with_prompt_fusion(
        self,
        image_path: str,
        prompts: List[str],
        threshold: Optional[float],
        text_threshold: Optional[float],
        max_detections_per_prompt: int,
        merge_iou: float,
    ) -> List[Dict[str, Any]]:
        detector = self.context.require("detector")
        fused: List[Dict[str, Any]] = []

        for prompt in prompts:
            self.context.emit_text(f"GroundingDINO 检测候选药盒 prompt={prompt}")
            result = detector.detect(
                image_path=image_path,
                target_name=prompt,
                threshold=threshold,
                text_threshold=text_threshold,
                max_objects=max_detections_per_prompt,
                enable_filter=True,
            )

            for det in result.get("detections", []):
                det = dict(det)
                det["source_prompt"] = prompt
                self._merge_detection(fused, det, merge_iou=merge_iou)

        fused.sort(
            key=lambda d: (
                len(d.get("prompt_hits", [])),
                float(d.get("confidence", 0.0) or 0.0),
                self._bbox_area(d.get("bbox")),
            ),
            reverse=True,
        )

        for idx, det in enumerate(fused):
            det["object_id"] = f"obj_{idx + 1}"

        return fused

    @classmethod
    def _merge_detection(
        cls,
        fused: List[Dict[str, Any]],
        det: Dict[str, Any],
        merge_iou: float,
    ) -> None:
        for old in fused:
            if cls._bbox_iou(old["bbox"], det["bbox"]) < merge_iou:
                continue

            old_conf = float(old.get("confidence", 0.0) or 0.0)
            new_conf = float(det.get("confidence", 0.0) or 0.0)
            if new_conf > old_conf:
                old["bbox"] = det["bbox"]
                old["confidence"] = new_conf
                old["phrase"] = det.get("phrase")
                old["source_prompt"] = det.get("source_prompt")

            hits = old.setdefault("prompt_hits", [])
            prompt = det.get("source_prompt")
            if prompt and prompt not in hits:
                hits.append(prompt)
            old["fused_confidence"] = max(old_conf, new_conf)
            return

        new_det = dict(det)
        new_det["prompt_hits"] = [det.get("source_prompt")] if det.get("source_prompt") else []
        new_det["fused_confidence"] = float(det.get("confidence", 0.0) or 0.0)
        fused.append(new_det)

    def _candidate_pipeline(self, image_path, detections, query, expand_bbox_px, padding, workers, **kwargs):
        root = Path('runs/current_capture/perception') / uuid.uuid4().hex
        started = time.perf_counter()
        memory = memory_for(self.context.camera)
        source = getattr(self.context.camera, 'last_wrist_capture', None) if memory is not None else None
        epoch = (source.get('observation_cache', {}).get('epoch')
                 if isinstance(source, dict) and source.get('color_path') == image_path else None)

        def require_current():
            if memory is not None and epoch is not None and not memory.commit(epoch, lambda: None):
                raise StaleObservationError('Scene changed; discard pending candidate inference')

        def process(indexed):
            require_current()
            index, detection = indexed
            # Share backend instances, but buffer UI events and keep output
            # files and intermediate dictionaries local to this candidate.
            context = copy.copy(self.context)
            events = []
            context.event_callback = events.append
            finder = type(self)(context)
            folder = root / f'candidate_{index:02d}'
            instances = finder._segment_candidates(image_path, [copy.deepcopy(detection)],
                        expand_bbox_px, output_root=folder, **kwargs)
            instances = [inst for inst in instances if inst.get('success')]
            if instances:
                require_current()
                instances = finder._augment_ocr(image_path, instances, query, padding, output_root=folder)
            require_current()
            return instances, events

        instances = []
        for candidate_instances, events in ordered_bounded_map(process, enumerate(detections), workers):
            instances.extend(candidate_instances)
            for event in events:
                self.context.emit(event)
        self.context.emit_text(f'[perception_pipeline] {len(detections)} candidates, workers={workers}, '
                               f'elapsed={time.perf_counter()-started:.3f}s; GPU=1, OCR=1')
        return instances

    def _segment_candidates(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        expand_bbox_px: int,
        output_root=None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        segment_skill = SegmentObjectSkill(self.context)
        instances = []
        root = Path(output_root) if output_root is not None else Path('runs/current_capture/perception') / uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=True)

        for idx, det in enumerate(detections):
            obj_id = det.get("object_id", f"obj_{idx + 1}")
            prefix = str(root / f'segment_{idx + 1:02d}')
            seg_result = segment_skill.run(
                image_path=image_path,
                bbox=det["bbox"],
                target_name="medicine_box",
                segment_all=False,
                output_prefix=prefix,
                expand_bbox_px=expand_bbox_px,
                select_by="iou",
                **kwargs,
            )

            if not seg_result.success:
                self.context.emit_warning(f"{obj_id} 分割失败: {seg_result.error}")
                continue

            seg_instances = seg_result.data.get("instances", [])
            if not seg_instances:
                continue

            inst = dict(seg_instances[0])
            inst["object_id"] = obj_id
            inst["detection"] = det
            inst["confidence"] = det.get("confidence")
            inst["prompt_hits"] = det.get("prompt_hits", [])
            inst["source_prompt"] = det.get("source_prompt")
            instances.append(inst)

        return instances

    @staticmethod
    def _instances_from_detections(
        image_path: str,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        instances = []
        for idx, det in enumerate(detections):
            instances.append(
                {
                    "object_id": det.get("object_id", f"obj_{idx + 1}"),
                    "name": det.get("name", "medicine_box"),
                    "target_name": "medicine_box",
                    "detection": det,
                    "bbox": det.get("bbox"),
                    "confidence": det.get("confidence"),
                    "phrase": det.get("phrase"),
                    "image_path": image_path,
                    "prompt_hits": det.get("prompt_hits", []),
                    "source_prompt": det.get("source_prompt"),
                    "success": True,
                    "segmenter": "bbox_only",
                    "mask_path": None,
                    "mask_npy_path": None,
                    "mask_area": None,
                }
            )
        return instances

    def _augment_ocr(
        self,
        image_path: str,
        instances: List[Dict[str, Any]],
        query: str,
        padding: int,
        output_root=None,
    ) -> List[Dict[str, Any]]:
        ocr = self.context.require("ocr")
        root = Path(output_root) if output_root is not None else Path('runs/current_capture/perception') / uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=True)

        for idx, inst in enumerate(instances):
            texts = []
            confidences = []
            items = []
            vis_paths = []

            sources = []
            if inst.get("crop_path"):
                sources.append(("crop", inst["crop_path"]))
            if inst.get("masked_image_path"):
                sources.append(("masked", inst["masked_image_path"]))

            for source_name, source_path in sources:
                try:
                    result = ocr.read_image(
                        image_path=source_path,
                        output_prefix=str(root / f'ocr_{idx + 1:02d}_{source_name}'),
                        min_confidence=0.0,
                    )
                    texts.extend(result.get("texts", []))
                    confidences.extend(result.get("confidences", []))
                    items.extend(result.get("items", []))
                    if result.get("vis_path"):
                        vis_paths.append(result["vis_path"])
                except Exception as e:
                    from hardware.fr3 import MotionError
                    from hardware.robotiq import GripperError
                    if isinstance(e, (MotionError, GripperError)):
                        raise
                    inst.setdefault("ocr_errors", []).append(f"{source_name}: {e}")

            if inst.get("bbox"):
                try:
                    result = ocr.read_region(
                        image_path=image_path,
                        bbox=inst["bbox"],
                        output_prefix=str(root / f'ocr_{idx + 1:02d}_region'),
                        padding=padding,
                        min_confidence=0.0,
                    )
                    texts.extend(result.get("texts", []))
                    confidences.extend(result.get("confidences", []))
                    items.extend(result.get("items", []))
                    inst["ocr_region_crop_path"] = result.get("crop_path")
                    if result.get("vis_path"):
                        vis_paths.append(result["vis_path"])
                except Exception as e:
                    from hardware.fr3 import MotionError
                    from hardware.robotiq import GripperError
                    if isinstance(e, (MotionError, GripperError)):
                        raise
                    inst.setdefault("ocr_errors", []).append(f"region: {e}")

            inst["ocr_success"] = bool(texts)
            inst["ocr_texts"] = self._dedupe_texts(texts)
            inst["ocr_text"] = " ".join(inst["ocr_texts"])
            inst["ocr_confidences"] = confidences
            inst["ocr_avg_confidence"] = (
                sum(float(x) for x in confidences) / len(confidences)
                if confidences else 0.0
            )
            inst["ocr_items"] = items
            inst["ocr_vis_paths"] = vis_paths

            self.context.emit_text(
                f"候选 {inst.get('object_id')} OCR: {inst['ocr_text'] or '<empty>'}"
            )
            if vis_paths:
                self.context.emit_image(
                    image_path=vis_paths[0],
                    caption=f"候选 {inst.get('object_id')} OCR 增强结果：{query}",
                )

        return instances

    def _select_best_instance(
        self,
        instances: List[Dict[str, Any]],
        query: str,
        aliases: Optional[List[str]],
        min_score: float,
    ) -> SkillResult:
        keywords = SelectObjectByTextSkill._build_keywords(query, aliases)
        ranked = []

        for idx, inst in enumerate(instances):
            text = SelectObjectByTextSkill._collect_instance_text(inst)
            text_score, matched_keyword, match_type = SelectObjectByTextSkill._score_text(
                text=text,
                keywords=keywords,
            )
            ocr_conf = self._clamp(float(inst.get("ocr_avg_confidence", 0.0) or 0.0))
            det_conf = self._clamp(float(inst.get("confidence", 0.0) or 0.0))
            prompt_score = min(len(inst.get("prompt_hits", [])) / 3.0, 1.0)
            mask_score = self._mask_quality_score(inst)

            final_score = (
                0.68 * text_score
                + 0.10 * ocr_conf
                + 0.10 * det_conf
                + 0.08 * prompt_score
                + 0.04 * mask_score
            )

            if not text.strip():
                final_score *= 0.62

            ranked.append(
                {
                    "index": idx,
                    "object_id": inst.get("object_id"),
                    "bbox": inst.get("bbox"),
                    "crop_path": inst.get("crop_path"),
                    "overlay_path": inst.get("overlay_path"),
                    "ocr_text": text,
                    "ocr_avg_confidence": ocr_conf,
                    "detection_confidence": det_conf,
                    "prompt_hits": inst.get("prompt_hits", []),
                    "score": final_score,
                    "raw_text_score": text_score,
                    "matched_keyword": matched_keyword,
                    "match_type": match_type,
                    "mask_score": mask_score,
                    "instance": inst,
                }
            )

        ranked.sort(key=lambda x: x["score"], reverse=True)
        best = ranked[0] if ranked else None

        if best is None or best["score"] < min_score:
            highest = 0.0 if best is None else best["score"]
            return SkillResult(
                success=False,
                data={"query": query, "keywords": keywords, "ranked_candidates": ranked},
                error=f"未找到稳定匹配的药盒。最高分={highest:.3f}, min_score={min_score:.3f}",
            )

        return SkillResult(
            success=True,
            data={
                "query": query,
                "keywords": keywords,
                "selected_instance": best["instance"],
                "selected_index": best["index"],
                "score": best["score"],
                "matched_keyword": best["matched_keyword"],
                "match_type": best["match_type"],
                "ranked_candidates": ranked,
            },
            message=f"成功选择药盒：{query}",
        )

    def _build_grasp_input(
        self,
        capture: Dict[str, Any],
        image_path: str,
        selected: Dict[str, Any],
        query: str,
        use_mask_for_grasp: bool,
        fallback_intrinsics: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        depth_path = self._existing_or_none(
            capture.get("depth_path")
            or capture.get("wrist_depth_path")
            or "runs/current_capture/wrist_depth.png"
        )
        depth_npy_path = self._existing_or_none(
            capture.get("depth_npy_path")
            or capture.get("depth_meter_path")
        )

        if depth_path is None and depth_npy_path is None:
            raise FileNotFoundError("缺少同一帧 depth_path/depth_npy_path，无法抓取。")

        intrinsics = (
            fallback_intrinsics
            or capture.get("depth_intrinsics")
            or capture.get("color_intrinsics")
            or capture.get("camera_intrinsics")
        )
        intrinsics = self._normalize_intrinsics(intrinsics)

        if intrinsics is None:
            raise RuntimeError("缺少相机内参，无法构造 AnyGrasp 输入。")

        mask_path = None
        mask_npy_path = None
        if use_mask_for_grasp:
            mask_path = self._existing_or_none(selected.get("mask_path"))
            mask_npy_path = self._existing_or_none(selected.get("mask_npy_path"))
            if mask_path is None and mask_npy_path is None:
                raise FileNotFoundError("use_mask_for_grasp=True，但缺少目标 mask。")

        bbox = selected.get("bbox")
        if bbox is None and mask_path is None and mask_npy_path is None:
            raise FileNotFoundError("缺少目标 bbox/mask，无法构造 AnyGrasp 输入。")

        return {
            "capture_geometry": capture.get("capture_geometry"),
            "observation_provenance": capture.get("observation_cache"),
            "capture_timestamp": capture.get("timestamp"),
            "camera_name": capture.get("camera_name"),
            "device_serial": capture.get("device_serial"),
            "points_path": capture.get("points_path"),
            "object_id": selected.get("object_id"),
            "target_name": selected.get("target_name", "medicine_box"),
            "identity_evidence": selected.get("identity_evidence"),
            "query": query,
            "color_path": image_path,
            "depth_path": depth_path,
            "depth_npy_path": depth_npy_path,
            "mask_path": mask_path,
            "mask_npy_path": mask_npy_path,
            "bbox": bbox,
            "crop_path": selected.get("crop_path"),
            "overlay_path": selected.get("overlay_path"),
            "depth_intrinsics": intrinsics,
            "color_intrinsics": intrinsics,
            "camera_intrinsics": intrinsics,
            "depth_scale": capture.get("depth_scale", 1000.0),
            "ocr_text": selected.get("ocr_text"),
            "ocr_texts": selected.get("ocr_texts"),
        }

    def _execute_grasp(
        self,
        plan: Dict[str, Any],
        move_speed: float,
        move_acc: float,
        gripper_open: Optional[Callable[[], bool]],
        gripper_close: Optional[Callable[[], bool]],
        gripper_close_repeat_count: int,
        gripper_close_repeat_interval_s: float,
        gripper_close_wait_s: float,
    ) -> Dict[str, Any]:
        if self.context.transforms is not None:
            from hardware.agent_geometry import execute_grasp
            return execute_grasp(self.context, plan, gripper_open, gripper_close)
        arm = self.context.require("xarm")
        approach = plan.get("approach_pose_mmrad")
        grasp = plan.get("best_grasp_xarm_mmrad")
        lift = plan.get("lift_pose_mmrad")

        for name, pose in [("approach", approach), ("grasp", grasp), ("lift", lift)]:
            if pose is None or len(pose) != 6:
                return {"success": False, "error": f"抓取规划缺少 {name} pose。"}

        if gripper_open and not gripper_open():
            return {"success": False, "error": "夹爪打开失败。"}

        for label, pose in [("approach", approach), ("grasp", grasp)]:
            ok = self._move_xarm(arm, pose, move_speed, move_acc)
            if not ok:
                return {"success": False, "error": f"xArm 移动到 {label} 失败。"}

        close_count = max(1, int(gripper_close_repeat_count or 1))
        if gripper_close:
            for close_index in range(close_count):
                self.context.emit_text(f"夹爪闭合命令 {close_index + 1}/{close_count}")
                if not gripper_close():
                    return {
                        "success": False,
                        "error": f"夹爪闭合失败，第 {close_index + 1}/{close_count} 次。",
                    }
                if close_index < close_count - 1 and gripper_close_repeat_interval_s > 0:
                    time.sleep(float(gripper_close_repeat_interval_s))

        if gripper_close_wait_s > 0:
            self.context.emit_text(f"等待夹爪完全闭合：{gripper_close_wait_s:.2f}s")
            time.sleep(float(gripper_close_wait_s))

        if not self._move_xarm(arm, lift, move_speed, move_acc):
            return {"success": False, "error": "xArm 抬升失败。"}

        return {"success": True, "message": "抓取执行完成。"}

    def _build_target_observe_pose(
        self,
        capture: Dict[str, Any],
        selected: Dict[str, Any],
        fallback_intrinsics: Optional[Dict[str, float]],
        xarm_ip: str,
        handeye_config_path: str,
        observe_z_mm: float,
        depth_window_px: int,
    ) -> List[float]:
        bbox = selected.get("bbox")
        if not bbox:
            raise RuntimeError("目标二次观察需要 selected bbox。")

        depth_path = self._existing_or_none(
            capture.get("depth_path")
            or capture.get("wrist_depth_path")
            or "runs/current_capture/wrist_depth.png"
        )
        if depth_path is None:
            raise RuntimeError("目标二次观察需要 wrist depth_path。")

        intrinsics = self._normalize_intrinsics(
            fallback_intrinsics
            or capture.get("depth_intrinsics")
            or capture.get("color_intrinsics")
            or capture.get("camera_intrinsics")
        )
        if intrinsics is None:
            raise RuntimeError("目标二次观察缺少相机内参。")

        depth = self._load_depth(
            depth_path=depth_path,
            depth_scale=capture.get("depth_scale", 1000.0),
        )
        center_pixel = self._bbox_center_pixel(
            bbox=bbox,
            image_width=depth.shape[1],
            image_height=depth.shape[0],
        )
        center_depth = self._median_depth_near_pixel(
            depth=depth,
            pixel=center_pixel,
            window_px=depth_window_px,
        )
        if center_depth is None:
            raise RuntimeError("目标 bbox 中心附近没有有效深度，无法移动二次观察。")

        center_camera_xyz = self._pixel_to_camera_xyz(
            pixel=center_pixel,
            depth_m=center_depth,
            intrinsics=intrinsics,
        )
        current_ee_pose = PlanGraspWithAnyGraspSkill(
            self.context
        )._get_current_xarm_pose_mrad(xarm_ip=xarm_ip)
        if current_ee_pose is None:
            raise RuntimeError("无法通过 xArm SDK 获取当前末端位姿。")

        handeye_rot, handeye_trans = PlanGraspWithAnyGraspSkill._load_handeye_cam2ee(
            handeye_config_path,
        )
        center_base_xyz = self._camera_point_to_base(
            point_camera=center_camera_xyz,
            current_ee_pose=current_ee_pose,
            handeye_rot=handeye_rot,
            handeye_trans=handeye_trans,
        )
        if self.context.transforms is not None:
            from hardware.agent_geometry import point_to_base
            center_base_xyz = point_to_base(capture, center_camera_xyz, self.context.transforms)


        return [
            float(center_base_xyz[0]) * 1000.0,
            float(center_base_xyz[1]) * 1000.0,
            float(observe_z_mm),
            3.14,
            0.0,
            -1.57,
        ]

    @staticmethod
    def _move_xarm(arm, pose: List[float], speed: float, acc: float) -> bool:
        x, y, z, rx, ry, rz = [float(v) for v in pose]
        code = arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=rx,
            pitch=ry,
            yaw=rz,
            speed=speed,
            mvacc=acc,
            wait=True,
        )
        return code == 0

    @staticmethod
    def _load_depth(depth_path: str, depth_scale: float) -> np.ndarray:
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(depth_path)

        depth_raw = depth_raw.astype(np.float32)
        depth_scale = float(depth_scale)
        if 0 < depth_scale < 1:
            return depth_raw * depth_scale
        return depth_raw / depth_scale

    @staticmethod
    def _bbox_center_pixel(
        bbox: List[int],
        image_width: int,
        image_height: int,
    ) -> List[int]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        u = int(round(max(0, min((x1 + x2) * 0.5, image_width - 1))))
        v = int(round(max(0, min((y1 + y2) * 0.5, image_height - 1))))
        return [u, v]

    @staticmethod
    def _median_depth_near_pixel(
        depth: np.ndarray,
        pixel: List[int],
        window_px: int,
    ) -> Optional[float]:
        u, v = [int(x) for x in pixel]
        half = max(1, int(window_px) // 2)
        h, w = depth.shape[:2]
        x1 = max(0, u - half)
        x2 = min(w, u + half + 1)
        y1 = max(0, v - half)
        y2 = min(h, v + half + 1)
        patch = depth[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0.02) & (patch < 1.5)]
        if len(valid) == 0:
            return None
        return float(np.median(valid))

    @staticmethod
    def _pixel_to_camera_xyz(
        pixel: List[int],
        depth_m: float,
        intrinsics: Dict[str, float],
    ) -> List[float]:
        u, v = [float(x) for x in pixel]
        z = float(depth_m)
        x = (u - float(intrinsics["cx"])) / float(intrinsics["fx"]) * z
        y = (v - float(intrinsics["cy"])) / float(intrinsics["fy"]) * z
        return [float(x), float(y), float(z)]

    @staticmethod
    def _camera_point_to_base(
        point_camera: List[float],
        current_ee_pose: List[float],
        handeye_rot,
        handeye_trans,
    ) -> List[float]:
        point_camera = np.asarray(point_camera, dtype=float).reshape(3)
        handeye_rot = np.asarray(handeye_rot, dtype=float)
        handeye_trans = np.asarray(handeye_trans, dtype=float).reshape(3)

        x_ee, y_ee, z_ee, rx_ee, ry_ee, rz_ee = current_ee_pose
        r_ee2base = R.from_euler(
            "ZYX",
            [rz_ee, ry_ee, rx_ee],
            degrees=False,
        ).as_matrix()
        point_ee = handeye_rot @ point_camera + handeye_trans
        point_base = r_ee2base @ point_ee + np.asarray([x_ee, y_ee, z_ee], dtype=float)
        return [float(x) for x in point_base.tolist()]

    @staticmethod
    def _dedupe_texts(texts: List[str]) -> List[str]:
        clean = []
        for text in texts:
            text = str(text).strip()
            if text and text not in clean:
                clean.append(text)
        return clean

    @staticmethod
    def _existing_or_none(path) -> Optional[str]:
        if not path:
            return None
        p = Path(path)
        return str(p) if p.exists() else None

    @staticmethod
    def _normalize_intrinsics(intrinsics):
        if intrinsics is None:
            return None
        if isinstance(intrinsics, dict):
            fx = intrinsics.get("fx")
            fy = intrinsics.get("fy")
            cx = intrinsics.get("cx", intrinsics.get("ppx"))
            cy = intrinsics.get("cy", intrinsics.get("ppy"))
            if fx is not None and fy is not None and cx is not None and cy is not None:
                return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}
        return intrinsics

    @staticmethod
    def _bbox_area(bbox) -> float:
        if not bbox:
            return 0.0
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

    @classmethod
    def _bbox_iou(cls, box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union = cls._bbox_area(box_a) + cls._bbox_area(box_b) - inter
        return 0.0 if union <= 0 else float(inter / union)

    @staticmethod
    def _clamp(value: float) -> float:
        if math.isnan(value):
            return 0.0
        return max(0.0, min(1.0, value))

    @classmethod
    def _mask_quality_score(cls, inst: Dict[str, Any]) -> float:
        bbox_area = cls._bbox_area(inst.get("bbox"))
        mask_area = float(inst.get("mask_area", 0.0) or 0.0)
        if bbox_area <= 0 or mask_area <= 0:
            return 0.0
        fill = mask_area / bbox_area
        if fill < 0.05 or fill > 1.20:
            return 0.15
        if 0.20 <= fill <= 0.95:
            return 1.0
        return 0.55
