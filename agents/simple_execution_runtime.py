"""Execute an action sequence, preserve task state and report failures.

Service assembly is delegated to runtime.services; device implementations live
in hardware/ and skill behavior lives in skill_lib/.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.actions import SimpleAction
# Keep the old camera-wrapper import path available to callers.
from runtime.camera import TaskCachedCamera as TaskCachedCamera
from runtime.services import build_camera, build_execution_context
from skill_lib.evaluation.verify_medicine_placed_vdm import VerifyMedicinePlacedVDMSkill
from skill_lib.config import SimpleRobotConfig
from skill_lib.simple_actions import SimpleRobotActions


class SimpleExecutionRuntime:
    """
    Execute concise actions with shared robot state.

    Responsibilities:
    - obtain shared services from runtime.services;
    - pre-observe all containers before grasping;
    - execute observe / grasp / place actions through SimpleRobotActions;
    - optionally verify placement after each place;
    - write a compact failure report for replanning.
    """

    def __init__(self, config):
        self.config = config
        self.replan_attempts = 0
        self.replan_paths = {}

    def run(self, actions: List[SimpleAction]) -> bool:
        if not actions:
            raise ValueError('规划没有可执行动作，未启动相机或机械臂执行会话。')
        from hardware.runtime import require_task_geometry
        require_task_geometry(self.config)
        print("\n[executing] 使用简短技能执行")
        context, camera_service = self._build_execution_context()
        arm = None
        hardware = None
        perception_pipeline = None

        try:
            camera_service.start()
            time.sleep(self.config.camera_warmup_s)

            # Resolve all known names before opening an execution session or moving home.
            actions = self._resolve_action_targets(context, actions)

            if getattr(self.config, "hardware_config_path", None):
                from hardware.runtime import build_hardware
                hardware = build_hardware(self.config.hardware_config_path,
                    execute=self.config.execute_robot, agent=True,
                    speed_scaling=getattr(self.config, "arm_speed_scaling", None))
                hardware.attach(context, monitor_gripper=(self.config.execute_robot and not self.config.disable_grip))
                if self.config.execute_robot and not self.config.disable_grip:
                    session = hardware.gripper.session_info()
                    print('[FR3] 夹爪通信:', session.get('communication_mode', 'thread'),
                          'worker_pid='+str(session.get('worker_pid')),
                          'state_timeout_s='+str(session.get('state_timeout_s')))
            elif self.config.execute_robot:
                arm = self._init_arm(self.config.xarm_ip)
                context.xarm = arm

            robot = self._build_robot(context)
            self._load_memory(robot)

            if self.config.execute_robot:
                print("[executing] 回到 home 并拍摄任务首帧。")
                home_result = robot.home()
                if not home_result.success:
                    self._write_failure_report(
                        failed_action=SimpleAction(op="home", target="before_task_snapshot"),
                        medicine_query=None,
                        failure_type="return_home_failed",
                        error=home_result.error,
                        evidence=home_result.data,
                    )
                    return False

            captures = self._capture_task_home_view(context, camera_service)
            from runtime.task_perception_pipeline import start_task_perception_pipeline
            perception_pipeline = start_task_perception_pipeline(robot, actions, captures, self.config)

            if not self._preobserve_containers(robot, context.camera, actions):
                self._save_memory(robot, completed_actions=[])
                return False

            # preobserve_containers already returns home after each container,
            # including the last. Avoid another identical MoveIt request here.
            if perception_pipeline is not None:
                perception_pipeline.publish()

            return self._run_actions(
                robot=robot,
                context=context,
                camera_service=camera_service,
                actions=actions,
            )
        finally:
            if perception_pipeline is not None:
                perception_pipeline.close()
            if hardware is not None:
                try:
                    if (self.config.execute_robot and not self.config.disable_grip
                            and hardware.gripper.session_info().get("active")):
                        try:
                            state = hardware.gripper.assert_healthy()
                        except Exception as error:
                            print(f"[FR3] 夹爪通信异常，结束会话：{error}")
                            state = None
                        if state is not None and hardware.gripper.may_be_holding_object():
                            from hardware.gsnet_pick import hold_console
                            print("[FR3] 夹爪有接触记录，可能仍夹持物体；保持通信，使用 open 释放或 quit 退出。")
                            hold_console(hardware.gripper, lambda event, **data: print(event, data))
                finally:
                    hardware.disconnect()
            if arm is not None:
                try:
                    arm.disconnect()
                except Exception:
                    pass

    def _resolve_action_targets(self, context, actions):
        from services.object_names import ObjectNameResolver, NameResolutionRequired, normalize_name
        from services.object_reference_service import atomic_json
        service = getattr(context, 'object_references', None)
        if service is None:
            return actions
        resolver = ObjectNameResolver(service, getattr(self.config, 'target_name_bindings', None))
        started = time.perf_counter()
        decisions, resolved, records = {}, [], []
        image_path = None
        report = Path(getattr(self.config, 'plan_json', 'runs/current_plan/plan_graph.json')).with_name('name_resolution.json')
        for action in actions:
            if action.op not in ('grasp', 'observe'):
                resolved.append(action)
                continue
            key = normalize_name(action.target)
            result = decisions.get(key) or resolver.lookup(action.target)
            if result['status'] == 'needs_scene':
                if image_path is None:
                    # A new task gets a new image; no previous-scene name binding is reused.
                    if isinstance(context.camera, TaskCachedCamera):
                        context.camera.disable_cache()
                    from runtime.observation_memory import fresh_camera
                    with fresh_camera(context.camera, 'new_task_name_resolution'):
                        image_path = context.require('camera').capture_wrist()['color_path']
                result = resolver.resolve_in_image(action.target, context, image_path)
            decisions[key] = result
            records.append(dict(op=action.op, **result))
            atomic_json(report, {'scope': 'this_task_only', 'records': records,
                                 'elapsed_s': time.perf_counter()-started})
            if (result['status'] == 'clarification_required' or
                    (result['status'] == 'unknown' and str(action.target).startswith('ref_'))):
                raise NameResolutionRequired(result)
            if result['status'] == 'resolved':
                print(f"[name-resolution] {action.target} → {result['canonical_name']} [{result['object_id']}]")
                resolved.append(SimpleAction(op=action.op, target=result['object_id']))
            else:
                resolved.append(action)  # Undeclared names retain the original OCR path.
        return resolved

    def _build_robot(self, context) -> SimpleRobotActions:
        context.task_goal = getattr(self.config, 'goal', '')
        pose_options = {}
        startup_pose = None
        buffers = [list(pose) for pose in getattr(self.config, 'geometry_occlusion_buffer_poses_mmrad', [])]
        buffer_positions = getattr(self.config, 'geometry_occlusion_buffer_positions_mm', [])
        if buffer_positions:
            import math
            if not getattr(self.config, 'hardware_config_path', None):
                raise ValueError('缓冲位 XYZ 自动补齐朝向仅支持 FR3 硬件配置')
            if any(len(xyz) != 3 or not all(math.isfinite(float(v)) for v in xyz)
                   for xyz in buffer_positions):
                raise ValueError('缓冲位法兰 XYZ 必须是三个有限的毫米数值')
        if getattr(self.config, "hardware_config_path", None):
            from hardware.config import load_config
            from runtime.home_pose import home_pose_session
            configured_home = load_config(self.config.hardware_config_path).arm.home_pose
            def read_startup_pose():
                nonlocal startup_pose
                startup_pose = context.require('arm').get_pose().values
                return startup_pose
            session = home_pose_session(self.config)
            home, _ = session.resolve(self.config.hardware_config_path, configured_home, read_startup_pose)
            print('[FR3] 会话 home（基座系法兰 m/rad）:', home, 'source='+session.source)
            pose_options["home_pose_mmrad"] = [v * 1000 for v in home[:3]] + list(home[3:])
            if buffer_positions:
                if startup_pose is None:
                    startup_pose = context.require('arm').get_pose().values
                for xyz in buffer_positions:
                    buffers.append([float(v) for v in xyz] + list(startup_pose[3:]))
                print('[FR3] 清障缓冲法兰位姿（毫米/弧度，启动时朝向）:', buffers)
        gripper_open = None
        gripper_close = None
        if self.config.execute_robot and not self.config.disable_grip:
            if getattr(self.config, "hardware_config_path", None):
                def gripper_open():
                    grip = context.require("gripper")
                    result = grip.open()
                    context.arm.motion_guard = lambda: grip.assert_healthy(require_ready=True)
                    return result.ready and result.object_state == 3
                def gripper_close():
                    grip = context.require("gripper")
                    result = grip.close()
                    # A contact-limited close deliberately finishes with gGTO=0.
                    context.arm.motion_guard = lambda: grip.assert_healthy(require_ready=True)
                    return grip.close_completed(result)
            else:
                gripper_open = lambda: self._call_grip_service_repeated(
                    value=self.config.grip_open_value,
                    repeat_count=self.config.gripper_open_repeat_count,
                    interval_s=self.config.gripper_open_repeat_interval_s,
                    label="夹爪打开",
                )
                gripper_close = lambda: self._call_grip_service(self.config.grip_close_value)

        return SimpleRobotActions(
            context=context,
            config=SimpleRobotConfig(
                **pose_options,
                use_sam_for_grasp=getattr(self.config, "use_sam_for_grasp", False),
                enable_geometry_occlusion=getattr(self.config, 'enable_geometry_occlusion', False),
                geometry_occlusion_buffer_poses_mmrad=buffers,
                enable_llm_occlusion_recovery=getattr(self.config, 'enable_llm_occlusion_recovery', False),
                geometry_occlusion_observation_budget=getattr(self.config, 'geometry_occlusion_observation_budget', 2),
                geometry_occlusion_recovery_view_poses_mmrad=getattr(self.config, 'geometry_occlusion_recovery_view_poses_mmrad', []),
                execute=self.config.execute_robot,
                xarm_ip=self.config.xarm_ip,
                handeye_config_path=self.config.handeye_config,
                move_speed=self.config.move_speed,
                move_acc=self.config.move_acc,
                visualize_selected_grasp=self.config.visualize_selected_grasp,
                visualize_bbox_filtered_grasps=self.config.visualize_bbox_filtered_grasps,
                bbox_filter_margin_px=self.config.bbox_filter_margin_px,
                bbox_filter_inner_margin_ratio=self.config.bbox_filter_inner_margin_ratio,
                collision_detection=self.config.collision_detection,
                grasp_selection_mode=getattr(self.config, "grasp_selection_mode", "best_good"),
                grasp_random_select_top_k=getattr(self.config, "grasp_random_select_top_k", 5),
                grasp_random_seed=getattr(self.config, "grasp_random_seed", None),
                grasp_pose_options=getattr(self.config, "grasp_pose_options", {}),
                gripper_close_repeat_count=self.config.gripper_close_repeat_count,
                gripper_close_repeat_interval_s=self.config.gripper_close_repeat_interval_s,
                gripper_close_wait_s=self.config.gripper_close_wait_s,
                place_slot_spacing_mm=self.config.place_slot_spacing_mm,
                enable_place_slot_offsets=not self.config.disable_place_slot_offsets,
                enable_nesyro_gate=getattr(self.config, "enable_nesyro_gate", True),
                nesyro_gate_threshold=getattr(self.config, "nesyro_gate_threshold", 0.7),
                nesyro_gate_max_probe=getattr(self.config, "nesyro_gate_max_probe", 3),
                nesyro_gate_trace_path=getattr(
                    self.config,
                    "nesyro_gate_trace_path",
                    "runs/current_plan/gate_trace.jsonl",
                ),
                gate_mode=getattr(self.config, "gate_mode", "legacy"),
                gate_experience_log_path=getattr(
                    self.config,
                    "gate_experience_log_path",
                    "logs/gate_experience/gate_experience.jsonl",
                ),
                learned_gate_checkpoint_path=getattr(
                    self.config,
                    "learned_gate_checkpoint_path",
                    "",
                ),
                enable_vlm_occlusion_gate=getattr(self.config, "enable_vlm_occlusion_gate", False),
                vlm_occlusion_base_url=getattr(self.config, "vlm_occlusion_base_url", ""),
                vlm_occlusion_api_key=getattr(self.config, "vlm_occlusion_api_key", ""),
                vlm_occlusion_model=getattr(self.config, "vlm_occlusion_model", ""),
                vlm_occlusion_timeout=getattr(self.config, "vlm_occlusion_timeout", 120),
                vlm_occlusion_max_tokens=getattr(self.config, "vlm_occlusion_max_tokens", 500),
                vlm_occlusion_temperature=getattr(self.config, "vlm_occlusion_temperature", 0.0),
                enable_occlusion_recovery=getattr(
                    self.config,
                    "enable_occlusion_recovery",
                    False,
                ),
                occlusion_recovery_max_attempts=getattr(
                    self.config,
                    "occlusion_recovery_max_attempts",
                    1,
                ),
                occlusion_recovery_capture_dir=getattr(
                    self.config,
                    "capture_dir",
                    "runs/current_capture",
                ),
                occlusion_buffer_pose_mmrad=list(
                    getattr(
                        self.config,
                        "occlusion_buffer_pose_mmrad",
                        [280.0, -260.0, 170.0, 3.14, 0.0, -1.57],
                    )
                ),
                occlusion_buffer_approach_clearance_mm=getattr(
                    self.config,
                    "occlusion_buffer_approach_clearance_mm",
                    120.0,
                ),
                occlusion_blocker_bbox_shrink_ratio=getattr(
                    self.config,
                    "occlusion_blocker_bbox_shrink_ratio",
                    0.08,
                ),
                occlusion_blocker_search_expand_ratio=getattr(
                    self.config,
                    "occlusion_blocker_search_expand_ratio",
                    1.8,
                ),
                occlusion_blocker_min_area_ratio=getattr(
                    self.config,
                    "occlusion_blocker_min_area_ratio",
                    0.50,
                ),
                occlusion_blocker_depth_closer_margin_m=getattr(
                    self.config,
                    "occlusion_blocker_depth_closer_margin_m",
                    0.01,
                ),
                occlusion_blocker_detector_prompts=list(
                    getattr(
                        self.config,
                        "occlusion_blocker_detector_prompts",
                        [
                            "medicine_box",
                            "snack",
                            "box",
                            "package",
                            "object on top of another object",
                        ],
                    )
                ),
                occlusion_blocker_detector_threshold=getattr(
                    self.config,
                    "occlusion_blocker_detector_threshold",
                    0.20,
                ),
                occlusion_blocker_detector_text_threshold=getattr(
                    self.config,
                    "occlusion_blocker_detector_text_threshold",
                    0.16,
                ),
                occlusion_blocker_detector_max_objects=getattr(
                    self.config,
                    "occlusion_blocker_detector_max_objects",
                    12,
                ),
                occlusion_allow_unlocalized_target=getattr(
                    self.config,
                    "occlusion_allow_unlocalized_target",
                    False,
                ),
                occlusion_blocker_max_image_area_ratio=getattr(
                    self.config,
                    "occlusion_blocker_max_image_area_ratio",
                    0.50,
                ),
                occlusion_blocker_max_target_area_ratio=getattr(
                    self.config,
                    "occlusion_blocker_max_target_area_ratio",
                    6.0,
                ),
                occlusion_unlocalized_vlm_min_confidence=getattr(
                    self.config,
                    "occlusion_unlocalized_vlm_min_confidence",
                    0.85,
                ),
                occlusion_unlocalized_detector_min_confidence=getattr(
                    self.config,
                    "occlusion_unlocalized_detector_min_confidence",
                    0.25,
                ),
                occlusion_min_depth_valid_ratio=getattr(
                    self.config,
                    "occlusion_min_depth_valid_ratio",
                    0.35,
                ),
                occlusion_foreground_depth_band_m=getattr(
                    self.config,
                    "occlusion_foreground_depth_band_m",
                    0.018,
                ),
                occlusion_min_foreground_support_ratio=getattr(
                    self.config,
                    "occlusion_min_foreground_support_ratio",
                    0.15,
                ),
                occlusion_active_view_enabled=getattr(
                    self.config,
                    "occlusion_active_view_enabled",
                    True,
                ),
                occlusion_active_view_z_mm=getattr(
                    self.config,
                    "occlusion_active_view_z_mm",
                    550.0,
                ),
                occlusion_active_view_y_offsets_mm=list(
                    getattr(
                        self.config,
                        "occlusion_active_view_y_offsets_mm",
                        [0.0, -120.0, 120.0],
                    )
                ),
                occlusion_active_view_wait_s=getattr(
                    self.config,
                    "occlusion_active_view_wait_s",
                    0.8,
                ),
            ),
            gripper_open=gripper_open,
            gripper_close=gripper_close,
        )

    def _capture_task_home_view(self, context, camera_service):
        from runtime.observation_memory import invalidate_camera
        invalidate_camera(camera_service, 'new_task')
        task_global = (None if getattr(camera_service, 'active_camera_names', None) == ('wrist',)
                       else camera_service.capture_global())
        task_wrist = camera_service.capture_wrist()
        if isinstance(context.camera, TaskCachedCamera):
            context.camera.set_task_captures(
                global_capture=task_global,
                wrist_capture=task_wrist,
            )
        return {'wrist': task_wrist, 'global': task_global}

    def _preobserve_containers(self, robot, camera, actions: List[SimpleAction]) -> bool:
        print("[executing] 预观测所有目标容器。")
        if isinstance(camera, TaskCachedCamera):
            # Container localization performs a second live capture after moving
            # above the first estimate. Reusing the task home-frame cache here
            # makes the refined observation silently use the wrong viewpoint.
            camera.disable_cache()

        result = robot.preobserve_containers(actions)

        if not result.success and isinstance(camera, TaskCachedCamera):
            print("[executing] 容器预观测失败，保持实时拍摄再试一次。")
            camera.disable_cache()
            from runtime.observation_memory import fresh_camera
            with fresh_camera(camera, 'container_preobserve_retry'):
                result = robot.preobserve_containers(actions)

        if isinstance(camera, TaskCachedCamera):
            camera.disable_cache()

        if result.success:
            self._print_container_observation_summary(result.data)
            return True

        self._write_failure_report(
            failed_action=SimpleAction(op="preobserve_containers", target="all containers"),
            medicine_query=None,
            failure_type="container_observation_failed",
            error=result.error,
            evidence=result.data,
        )
        return False

    def _print_container_observation_summary(self, data: Dict[str, Any]) -> None:
        if not getattr(self.config, "show_container_observation", True):
            return

        items = list((data or {}).get("preobserved_containers") or [])
        if not items:
            return

        print("[executing] 容器观测结果：")
        for item in items:
            plan = item.get("data") or {}
            key = item.get("key") or item.get("container")
            final_center = plan.get("center_base_xyz")
            final_pixel = plan.get("center_pixel")
            depth = plan.get("center_depth_m")
            place_pose = plan.get("place_pose_mmrad")
            depth_method = (plan.get("depth_estimation") or {}).get("method")
            refined = plan.get("refined_observation") or {}
            first = plan.get("first_observation") or {}

            print(f"  - {key}")
            print(f"    center_pixel: {final_pixel}")
            print(f"    center_base_xyz_m: {self._round_list(final_center, 4)}")
            print(f"    center_depth_m: {self._round_value(depth, 4)}  method: {depth_method}")
            print(f"    place_pose_mmrad: {self._round_list(place_pose, 2)}")
            print(f"    used_refined_observation: {bool(refined)}")
            if first and refined:
                print(f"    first_bbox: {(first.get('selected_detection') or {}).get('bbox')}")
                print(f"    refined_bbox: {(refined.get('selected_detection') or {}).get('bbox')}")
                print(f"    first_center_base_xyz_m: {self._round_list(first.get('center_base_xyz'), 4)}")
                print(f"    refined_center_base_xyz_m: {self._round_list(refined.get('center_base_xyz'), 4)}")

    @staticmethod
    def _round_list(values, ndigits: int):
        if values is None:
            return None
        return [round(float(v), ndigits) for v in values]

    @staticmethod
    def _round_value(value, ndigits: int):
        if value is None:
            return None
        return round(float(value), ndigits)

    def _run_actions(self, robot, context, camera_service, actions: List[SimpleAction],
                     *, completed_actions=None, start_index=0) -> bool:
        from services.object_names import display_name
        current_grasp: Optional[str] = None
        completed_actions = list(completed_actions or [])
        last_gated_result: Optional[float] = None
        total_steps = max(1, len(actions))
        for index, action in enumerate(actions[start_index:], start=start_index + 1):
            print(f"[executing] {index}/{len(actions)} {action.op}({display_name(context, action.target)})")
            gate_blocked = False

            if action.op in {"observe", "grasp", "place"}:
                gate_result = robot.gate_before_action(
                    action=action,
                    current_holding=current_grasp,
                    target_plate=SimpleRobotActions._next_place_target(actions, index - 1),
                    previous_action_result=last_gated_result,
                    step_index=index - 1,
                    total_steps=total_steps,
                )
                if not gate_result.success:
                    result = gate_result
                    gate_blocked = True
                else:
                    result = None
            else:
                result = None

            if result is not None:
                pass
            elif action.op == "observe":
                result = robot.observe(action.target)
                if not result.success:
                    print(f"[executing][WARN] observe({action.target}) 失败，继续执行后续抓取：{result.error}")
                    result = self._skill_success(
                        action,
                        data={
                            "warning": "observe_failed_but_continue",
                            "original_error": result.error,
                            "original_data": result.data,
                        },
                    )
            elif action.op == "grasp":
                result = self._grasp_with_cache_retry(robot, context.camera, action.target)
                if result.success:
                    current_grasp = action.target
            elif action.op == "place":
                result = self._place_and_verify(
                    robot=robot,
                    context=context,
                    camera_service=camera_service,
                    action=action,
                    current_grasp=current_grasp,
                    step_index=index,
                )
                if result.success:
                    current_grasp = None
            else:
                result = self._skill_failure(action, f"unknown action: {action.op}")

            if not result.success:
                failure_medicine_query = current_grasp
                if action.op == "grasp":
                    failure_medicine_query = action.target
                failure_type = (result.data or {}).get("failure_type") or self._failure_type_for_action(action)
                if action.op == "place" and not str(failure_type).startswith("nesyro_gate"):
                    home_result = self._home_after_place(robot)
                    if not home_result.success:
                        self._log_gate_outcome_for_action(
                            robot=robot,
                            actions=actions,
                            action=action,
                            completed_actions=completed_actions,
                            primitive_success=False,
                            execution_failure_type="return_home_failed",
                        )
                        self._write_failure_report(
                            failed_action=SimpleAction(op="home", target="after_failed_place"),
                            medicine_query=failure_medicine_query,
                            failure_type="return_home_failed",
                            error=home_result.error,
                            failed_action_index=index - 1,
                            completed_actions=completed_actions,
                            evidence={
                                "home": home_result.data,
                                "original_place_error": result.error,
                                "original_place_data": result.data,
                            },
                        )
                        return False
                self._log_gate_outcome_for_action(
                    robot=robot,
                    actions=actions,
                    action=action,
                    completed_actions=completed_actions,
                    primitive_success=False,
                    execution_failure_type=failure_type,
                )
                report = self._write_failure_report(
                    failed_action=action,
                    medicine_query=failure_medicine_query,
                    failure_type=failure_type,
                    error=result.error,
                    failed_action_index=index - 1,
                    completed_actions=completed_actions,
                    evidence=result.data,
                )
                self._save_memory(robot, completed_actions=completed_actions)
                if gate_blocked and action.op == 'grasp' and current_grasp is None:
                    from agents.same_session_replanning import repair_gate_before_grasp
                    try:
                        repaired = repair_gate_before_grasp(
                            self, robot, context, completed_actions, report)
                    except (OSError, ValueError) as error:
                        print(f'[replanning] stopped: {error}')
                        repaired = None
                    if repaired is not None:
                        return self._run_actions(robot, context, camera_service, repaired,
                            completed_actions=completed_actions, start_index=len(completed_actions))
                return False

            if action.op in {"observe", "grasp", "place"}:
                last_gated_result = 1.0
            self._print_action_bbox_summary(action, result.data or {})
            completed_actions.append(self._action_to_dict(action))
            self._log_gate_outcome_for_action(
                robot=robot,
                actions=actions,
                action=action,
                completed_actions=completed_actions,
                primitive_success=True,
                execution_failure_type=None,
            )
            self._save_memory(robot, completed_actions=completed_actions)

        print("[executing] done")
        return True

    def _log_gate_outcome_for_action(
        self,
        robot,
        actions: List[SimpleAction],
        action: SimpleAction,
        completed_actions: List[Dict[str, Any]],
        primitive_success: bool,
        execution_failure_type: Optional[str],
    ) -> None:
        """Record one gate_outcome experience line for a gated action.

        Only runs for observe/grasp/place actions (the only gated ops).
        ``SimpleRobotActions.log_gate_outcome`` never raises, so this cannot
        affect the robot main flow.
        """

        if action.op not in {"observe", "grasp", "place"}:
            return
        total = max(1, len(actions))
        progress_before = len(completed_actions) / total
        progress_after = (len(completed_actions) + 1) / total if primitive_success else progress_before
        robot.log_gate_outcome(
            primitive_success=primitive_success,
            skill_success=primitive_success,
            execution_failure_type=execution_failure_type,
            task_progress_before=progress_before,
            task_progress_after=progress_after,
            trace_reference=str(getattr(self.config, "nesyro_gate_trace_path", "") or ""),
        )

    def _print_action_bbox_summary(self, action: SimpleAction, data: Dict[str, Any]) -> None:
        bbox_items = self._extract_bbox_items(data)
        if not bbox_items:
            return

        print(f"[bbox] {action.op}({action.target})")
        for item in bbox_items:
            print(f"  - {item['label']}: bbox={item.get('bbox')}")
            if item.get("object_id"):
                print(f"    object_id: {item.get('object_id')}")
            if item.get("score") is not None:
                print(f"    score: {item.get('score')}")
            if item.get("image_path"):
                print(f"    image_path: {item.get('image_path')}")

        debug_paths = self._save_bbox_debug_images(action, bbox_items)
        for path in debug_paths:
            print(f"    bbox_debug_image: {path}")

    def _extract_bbox_items(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        def add(label: str, source: Any):
            if not isinstance(source, dict):
                return
            bbox = source.get("bbox")
            if not bbox:
                return
            items.append(
                {
                    "label": label,
                    "bbox": bbox,
                    "object_id": source.get("object_id"),
                    "score": source.get("score")
                    or source.get("confidence")
                    or source.get("fused_confidence")
                    or source.get("color_score"),
                    "image_path": source.get("image_path") or source.get("color_path"),
                }
            )

        add("selected_instance", data.get("selected_instance"))
        add("grasp_input", data.get("grasp_input"))
        add("selected_detection", data.get("selected_detection"))

        place_plan = data.get("place_plan") if isinstance(data.get("place_plan"), dict) else {}
        add("place_plan.selected_detection", place_plan.get("selected_detection"))

        grasp_plan = data.get("grasp_plan") if isinstance(data.get("grasp_plan"), dict) else {}
        add("grasp_plan.grasp_input", grasp_plan.get("grasp_input"))

        original_data = data.get("original_data") if isinstance(data.get("original_data"), dict) else {}
        add("original.selected_instance", original_data.get("selected_instance"))
        add("original.grasp_input", original_data.get("grasp_input"))

        detections = original_data.get("detections") or data.get("detections")
        if isinstance(detections, list):
            for index, detection in enumerate(detections[:3], start=1):
                add(f"detection_top{index}", detection)

        return items

    def _save_bbox_debug_images(self, action: SimpleAction, bbox_items: List[Dict[str, Any]]) -> List[str]:
        try:
            import cv2
        except Exception as e:
            print(f"[bbox][WARN] cv2 不可用，无法保存 bbox 调试图：{e}")
            return []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in bbox_items:
            image_path = item.get("image_path")
            if not image_path:
                continue
            grouped.setdefault(str(image_path), []).append(item)

        output_paths = []
        for image_path, items in grouped.items():
            image = cv2.imread(image_path)
            if image is None:
                continue

            for index, item in enumerate(items, start=1):
                bbox = item.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                color = self._bbox_color(index)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

                label = str(item.get("label") or f"bbox_{index}")
                object_id = item.get("object_id")
                score = item.get("score")
                if object_id:
                    label = f"{label}:{object_id}"
                if score is not None:
                    try:
                        label = f"{label} {float(score):.2f}"
                    except Exception:
                        label = f"{label} {score}"

                y_text = max(20, y1 - 8)
                cv2.putText(
                    image,
                    label,
                    (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )

            safe_target = self._safe_filename(action.target)
            stem = Path(image_path).stem
            timestamp = time.strftime("%H%M%S")
            output_path = Path("runs/current_capture") / (
                f"bbox_debug_{timestamp}_{action.op}_{safe_target}_{stem}.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            output_paths.append(str(output_path))

        return output_paths

    @staticmethod
    def _bbox_color(index: int):
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]
        return colors[(index - 1) % len(colors)]

    @staticmethod
    def _safe_filename(text: str) -> str:
        clean = []
        for char in str(text or "target"):
            if char.isalnum() or char in ["_", "-"]:
                clean.append(char)
            else:
                clean.append("_")
        return "".join(clean).strip("_")[:40] or "target"

    def _load_memory(self, robot: SimpleRobotActions) -> None:
        path = Path(getattr(self.config, "simple_memory_path", ""))
        if not path.exists():
            return
        try:
            memory = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[memory][WARN] 读取执行记忆失败，忽略：{e}")
            return

        memory_goal = str(memory.get("goal") or "")
        if memory_goal and memory_goal != str(self.config.goal):
            print("[memory] 执行记忆属于旧任务，跳过加载。")
            return

        if getattr(self.config, "hardware_config_path", None):
            print("[memory] FR3 本轮重新拍摄；不复用跨任务的图像或坐标缓存。")
            return
        observations = memory.get("observations") or {}
        place_plans = memory.get("place_plans") or {}
        place_counts = memory.get("place_counts") or {}

        if isinstance(observations, dict):
            robot.observations.update(observations)
        if isinstance(place_plans, dict):
            robot.place_plans.update(place_plans)
        if isinstance(place_counts, dict):
            robot.place_counts.update({str(k): int(v) for k, v in place_counts.items() if str(v).isdigit()})

        medicines = [
            key for key, value in robot.observations.items()
            if isinstance(value, dict) and value.get("kind") == "medicine"
        ]
        if medicines:
            print(f"[memory] 已恢复药盒观察缓存：{', '.join(medicines)}")

    def _save_memory(self, robot: SimpleRobotActions, completed_actions: List[Dict[str, Any]]) -> None:
        path = Path(getattr(self.config, "simple_memory_path", ""))
        if not str(path):
            return

        memory = {
            "goal": self.config.goal,
            "completed_actions": completed_actions,
            "observations": self._compact_observations(robot.observations),
            "place_plans": self._safe_json(robot.place_plans),
            "place_counts": self._safe_json(robot.place_counts),
            "geometry_recovery": self._safe_json(dict(
                clear_counts=robot.occlusion_recovery_counts,
                buffer_slots_used=robot.geometry_buffer_slots_used,
                state=robot.geometry_recovery_state)),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[memory][WARN] 保存执行记忆失败：{e}")

    def _compact_observations(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        compact = {}
        for target, value in (observations or {}).items():
            if not isinstance(value, dict):
                continue
            if value.get("kind") == "medicine":
                compact[str(target)] = {
                    "kind": "medicine",
                    "target": value.get("target") or target,
                    "selected_instance": self._safe_json(value.get("selected_instance")),
                    "grasp_input": self._compact_grasp_input(value.get("grasp_input") or {}),
                }
            elif value.get("kind") == "container":
                compact[str(target)] = {
                    "kind": "container",
                    "target": value.get("target") or target,
                    "container_key": value.get("container_key"),
                    "place_plan": self._safe_json(value.get("place_plan")),
                }
        return compact

    def _compact_grasp_input(self, grasp_input: Dict[str, Any]) -> Dict[str, Any]:
        keep_keys = [
            "capture_geometry",
            "points_path",
            "mask_path",
            "mask_npy_path",
            "bbox",
            "color_path",
            "depth_path",
            "depth_npy_path",
            "depth_scale",
            "camera_intrinsics",
            "color_intrinsics",
            "depth_intrinsics",
            "query",
            "target_name",
            "object_id",
            "ocr_text",
            "ocr_texts",
        ]
        return {
            key: self._safe_json(grasp_input.get(key))
            for key in keep_keys
            if key in grasp_input
        }

    def _safe_json(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._safe_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._safe_json(v) for v in value]
        if hasattr(value, "tolist"):
            return self._safe_json(value.tolist())
        return str(value)

    def _grasp_with_cache_retry(self, robot, camera, target: str):
        if isinstance(camera, TaskCachedCamera):
            camera.disable_cache()

        result = robot.grasp(target)

        if isinstance(camera, TaskCachedCamera):
            camera.disable_cache()
        return result

    def _place_and_verify(self, robot, context, camera_service, action, current_grasp, step_index):
        from services.object_names import display_name
        before = self._capture_verify_pair(camera_service)
        result = robot.place(action.target)
        if not result.success:
            return result

        verify_ok = self._verify_place_if_needed(
            context=context,
            camera_service=camera_service,
            before_pair=before,
            medicine_query=display_name(context, current_grasp) if current_grasp else "target medicine",
            container=action.target,
            step_index=step_index,
        )
        home_result = self._home_after_place(robot)
        if not home_result.success:
            return self._skill_failure(
                SimpleAction(op="home", target="after_place"),
                home_result.error,
                data={"failure_type": "return_home_failed", "home": home_result.data},
            )
        if not verify_ok:
            return self._skill_failure(
                action,
                "放置后判别未确认目标进入容器。",
                data={"failure_type": "pick_place_verification_failed", "medicine_query": current_grasp},
            )
        return result

    def _home_after_place(self, robot: SimpleRobotActions):
        print("[executing] 放置验证完成，回到 home 观察位。")
        result = robot.home()
        if not result.success:
            print(f"[executing][WARN] 回 home 失败：{result.error}")
        return result

    def _build_execution_context(self):
        return build_execution_context(
            self.config, camera=self._build_camera(), event_callback=self._event_callback
        )

    def _build_camera(self):
        return build_camera(self.config)[0]

    def _verify_place_if_needed(self, context, camera_service, before_pair, medicine_query, container, step_index) -> bool:
        if not self.config.verify_place_vdm:
            return True
        if not (self.config.vdm_base_url and self.config.vdm_api_key and self.config.vdm_model):
            print("[verifying] 缺少 VDM API 参数，跳过放置判别。")
            return True

        after = self._capture_verify_pair(camera_service)
        container_name, container_color = self._split_container_label(container)
        result = VerifyMedicinePlacedVDMSkill(context).run(
            medicine_query=medicine_query,
            container_name=container_name,
            container_color=container_color,
            before_global_color_path=before_pair["global"]["color_path"],
            before_wrist_color_path=before_pair["wrist"]["color_path"],
            after_global_color_path=after["global"]["color_path"],
            after_wrist_color_path=after["wrist"]["color_path"],
            wrist_only=bool(after.get('global_is_wrist_fallback', False)),
            base_url=self.config.vdm_base_url,
            api_key=self.config.vdm_api_key,
            model_name=self.config.vdm_model,
            temperature=self.config.vdm_temperature,
            max_tokens=self.config.vdm_max_tokens,
            timeout=self.config.vdm_timeout,
            min_success_confidence=self.config.place_vdm_min_success_confidence,
            min_visible_success_confidence=self.config.place_vdm_min_visible_success_confidence,
        )
        print("[verifying] 放置判别:", result.success, result.data.get("failure_type") if result.data else result.error)
        if result.success:
            return True

        data = result.data or {}
        confidence = float(data.get("confidence") or 0.0)
        if (
            data.get("failure_type") in {"not_in_container", "uncertain", "container_occluded"}
            and not data.get("target_visible_elsewhere")
            and confidence < self.config.place_vdm_min_failure_confidence
        ):
            print("[verifying] 放置判定证据不足，按成功继续。")
            return True
        return not self.config.place_vdm_stop_on_failure

    @staticmethod
    def _capture_verify_pair(camera_service) -> Dict[str, Dict[str, Any]]:
        from runtime.observation_memory import fresh_camera
        with fresh_camera(camera_service, 'placement_verification'):
            if getattr(camera_service, 'active_camera_names', None) == ('wrist',):
                wrist = camera_service.capture_wrist()
                return {'global': wrist, 'wrist': wrist, 'global_is_wrist_fallback': True}
            return {'global': camera_service.capture_global(), 'wrist': camera_service.capture_wrist()}

    def _write_failure_report(
        self,
        failed_action,
        medicine_query,
        failure_type,
        error,
        evidence=None,
        failed_action_index=None,
        completed_actions=None,
    ):
        path = Path(self.config.failure_report_path)
        previous_report = None
        if str(failure_type) == "return_home_failed" and path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded.get("failure_type") != "return_home_failed":
                    previous_report = loaded
            except Exception:
                previous_report = None
        evidence = dict(evidence or {})
        if previous_report is not None:
            evidence["previous_failure_report"] = previous_report
        report = {
            "success": False,
            "goal": self.config.goal,
            "failed_action": failed_action.__dict__,
            "failed_action_index": failed_action_index,
            "completed_actions": completed_actions or [],
            "medicine_query": medicine_query,
            "failure_type": failure_type,
            "error": str(error or ""),
            "evidence": evidence,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "automatic_replan": evidence.get('automatic_replan',
                                              not getattr(self.config, 'enable_geometry_occlusion', False)),
            "same_session_replan_attempts": self.replan_attempts,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[failure] 已生成失败报告。")
        print(f"[failure] 报告路径：{path}")
        return report

    @staticmethod
    def _failure_type_for_action(action: SimpleAction) -> str:
        if action.op == "observe":
            return "observe_failed"
        if action.op == "grasp":
            return "grasp_failed"
        if action.op == "place":
            return "place_failed"
        return "execution_failed"

    @staticmethod
    def _action_to_dict(action: SimpleAction) -> Dict[str, str]:
        return {"op": str(action.op), "target": str(action.target)}

    @staticmethod
    def _split_container_label(label: str):
        text = str(label or "").strip()
        parts = text.split(maxsplit=1)
        if len(parts) == 2 and parts[0] in {"pink", "blue", "green", "red", "yellow", "white", "black"}:
            return parts[1], parts[0]
        return text, None

    def _event_callback(self, event: dict):
        event_type = event.get("type")
        if event_type in {"warning", "error"}:
            print(f"[{event_type}] {event.get('content')}")
        elif self.config.verbose_events and event_type == "text":
            print(f"[Text] {event.get('content')}")

    def _init_arm(self, ip: str):
        from xarm.wrapper import XArmAPI

        arm = XArmAPI(ip, is_radian=True)
        time.sleep(0.5)
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)
        return arm

    def _call_grip_service(self, value: int) -> bool:
        cmd = f"source {self.config.ros_ws}/devel/setup.bash && rosservice call {self.config.grip_service} {int(value)}"
        try:
            result = subprocess.run(
                ["bash", "-lc", cmd],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.config.grip_timeout,
            )
            if result.returncode != 0:
                print(f"[Gripper][WARN] rosservice returned code={result.returncode}")
                return bool(self.config.ignore_grip_error)
            return True
        except Exception as e:
            print(f"[Gripper][WARN] rosservice failed: {e}")
            return bool(self.config.ignore_grip_error)

    def _call_grip_service_repeated(
        self,
        value: int,
        repeat_count: int,
        interval_s: float,
        label: str,
    ) -> bool:
        count = max(1, int(repeat_count or 1))
        for index in range(count):
            print(f"[Gripper] {label}命令 {index + 1}/{count}")
            if not self._call_grip_service(value):
                return False
            if index < count - 1 and interval_s > 0:
                time.sleep(float(interval_s))
        return True

    @staticmethod
    def _skill_failure(action, error, data=None):
        from core.skill_base import SkillResult

        return SkillResult(success=False, data=data or {"action": action.__dict__}, error=error)

    @staticmethod
    def _skill_success(action, data=None):
        from core.skill_base import SkillResult

        return SkillResult(success=True, data=data or {"action": action.__dict__})
