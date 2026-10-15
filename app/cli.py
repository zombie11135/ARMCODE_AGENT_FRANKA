"""Interactive application entry point for FR3 and explicit legacy xArm runs.

Argument parsing and command dispatch live here. Agent orchestration lives in
agents/; shared service assembly and task configuration live in runtime/.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import readline

    readline.parse_and_bind("set editing-mode emacs")
    readline.parse_and_bind('"\\C-?": backward-delete-char')
    readline.parse_and_bind('"\\C-h": backward-delete-char')
except Exception:
    readline = None

try:
    from prompt_toolkit import prompt as prompt_toolkit_prompt
except Exception:
    prompt_toolkit_prompt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent_system import AgentSystemConfig, ArmCodeAgentSystem
from agents.simple_agent_system import SimpleArmCodeAgentSystem
from runtime.config import SimpleAgentSystemConfig


HOME_POSE_MMRAD = [345.0, 0.0, 450.0, 3.14, 0.0, -1.57]


def check_code(arm, code, label: str = "") -> bool:
    if code != 0:
        print(f"[xArm][ERROR] {label} failed, code={code}")
        print(f"[xArm] state={arm.state}, error={arm.error_code}, warn={arm.warn_code}")
        return False
    return True


def init_arm(ip: str):
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=True)
    time.sleep(0.5)

    print("[xArm] connected:", arm.connected)
    print("[xArm] state:", arm.state)
    print("[xArm] error:", arm.error_code)
    print("[xArm] warn:", arm.warn_code)

    arm.clean_warn()
    arm.clean_error()
    check_code(arm, arm.motion_enable(enable=True), "motion_enable")
    check_code(arm, arm.set_mode(0), "set_mode")
    check_code(arm, arm.set_state(0), "set_state")
    time.sleep(0.5)
    return arm


def move_xarm_pose(arm, pose, speed: float, acc: float) -> bool:
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
    return check_code(arm, code, "set_position")


def call_grip_service(
    value: int,
    ros_ws: str,
    service_name: str,
    timeout: float,
    ignore_error: bool,
) -> bool:
    cmd = f"source {ros_ws}/devel/setup.bash && rosservice call {service_name} {int(value)}"
    print(f"[Gripper] {cmd}")

    try:
        result = subprocess.run(
            ["bash", "-lc", cmd],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        if result.stdout.strip():
            print("[Gripper] output:")
            print(result.stdout)
        if result.returncode != 0:
            print(f"[Gripper][WARN] rosservice returned code={result.returncode}")
            return bool(ignore_error)
        return True
    except subprocess.TimeoutExpired:
        print(f"[Gripper][WARN] rosservice timeout: {service_name}")
        return bool(ignore_error)
    except Exception as e:
        print(f"[Gripper][WARN] rosservice failed: {e}")
        return bool(ignore_error)


def print_help():
    print(
        """
可输入自然语言任务，例如：
  将桌面上所有的药品盒子整理到粉色托盘中
  把布洛芬放到托盘里

内置命令：
  /help, help, 帮助              显示帮助
  /quit, quit, exit, 退出        退出系统
  /home, home, 回到初始位置       移动到初始位姿 [345,0,450,3.14,0,-1.57]
  /open, 打开夹爪                rosservice call /grip 1
  /close, 关闭夹爪               rosservice call /grip 0
  /reset, 重置执行状态            清空 execution_state / failure_report
  /clearcache, 清空模型缓存       清空已加载的 GroundingDINO/OCR
  /stopcamera, 关闭相机           停止缓存中的相机服务
  /names                       查看图库名称、别名、品牌和编号
  /bind 简称 = 名称或编号         只为下一次任务绑定一个临时称呼
  /bind-clear                  清空临时名称绑定

安全提示：
  未加 --execute_robot 时，内置运动/夹爪命令只打印，不会真的执行。
  普通任务会走 planning -> coding -> executing；若未加 --execute_robot，生成脚本也只干跑。
  执行失败后会写 failure_report，并默认自动重规划重试 1 次；下次执行默认跳过已完成子目标。
  若不希望失败后自动重试，运行 CLI 时加 --max_replan_attempts 0。
  使用 --use_vdm --execute_robot 时，默认只启用放置后的 VDM 判别器。
  若要查看完整底层日志，运行 CLI 时加 --debug_output。
  默认同进程执行生成脚本，GroundingDINO/OCR 会在交互式会话内复用。
  默认交互模式会预热并显示相机，后续任务只调用拍照接口。
"""
    )


def normalize_command(text: str) -> str:
    return " ".join(str(text or "").strip().split()).lower()


def is_quit_command(command: str) -> bool:
    return command in {"/quit", "quit", "exit", "q", "退出", "结束"}


def is_help_command(command: str) -> bool:
    return command in {"/help", "help", "?", "帮助"}


def is_home_command(command: str) -> bool:
    return command in {"/home", "home", "init", "initial", "回到初始位置", "初始位置", "回初始位"}


def is_open_gripper_command(command: str) -> bool:
    return command in {"/open", "open", "打开夹爪", "夹爪打开"}


def is_close_gripper_command(command: str) -> bool:
    return command in {"/close", "close", "关闭夹爪", "夹爪关闭"}


def is_reset_state_command(command: str) -> bool:
    return command in {"/reset", "reset", "重置执行状态", "清空执行状态"}


def is_clear_cache_command(command: str) -> bool:
    return command in {"/clearcache", "clearcache", "清空模型缓存", "清空缓存"}


def is_stop_camera_command(command: str) -> bool:
    return command in {"/stopcamera", "stopcamera", "关闭相机"}


def warmup_camera_if_needed(args):
    if args.no_camera_warmup:
        return None
    if not (args.execute_robot or args.use_vdm or args.warmup_camera):
        return None

    if getattr(args, "hardware_config", None):
        # The deferred calibrated task should not open devices merely on startup.
        if not (args.use_vdm or args.warmup_camera):
            return None
        from hardware.runtime import camera_service
        camera, _ = camera_service(args.hardware_config, "runs/current_capture")
        camera.start()
        time.sleep(args.camera_warmup_s)
        return camera

    from services.realsense_dual_camera_service import CameraConfig, RealSenseDualCameraService
    from services.runtime_service_cache import get_or_create_service

    camera_params = {
        "global_serial": args.global_serial,
        "wrist_serial": args.wrist_serial,
        "capture_dir": "runs/current_capture",
        "show_window": True,
        "global_color_width": 1280,
        "global_color_height": 720,
        "wrist_color_width": 1280,
        "wrist_color_height": 720,
        "wrist_depth_width": 1280,
        "wrist_depth_height": 720,
    }
    camera = get_or_create_service(
        "dual_realsense_camera",
        camera_params,
        lambda: RealSenseDualCameraService(
            global_config=CameraConfig(
                name="global",
                serial=args.global_serial,
                color_width=1280,
                color_height=720,
                color_fps=30,
                enable_depth=False,
            ),
            wrist_config=CameraConfig(
                name="wrist",
                serial=args.wrist_serial,
                color_width=1280,
                color_height=720,
                color_fps=30,
                enable_depth=True,
                depth_width=1280,
                depth_height=720,
                depth_fps=30,
            ),
            save_dir="runs/current_capture",
            show_window=True,
        ),
    )
    camera.start()
    time.sleep(args.camera_warmup_s)
    print("[agent] camera session ready.")
    return camera


def run_builtin_command(command: str, args) -> bool:
    from services.object_names import handle_name_command
    if handle_name_command(command, args):
        return True
    if getattr(args, "hardware_config", None):
        operation = ("home" if is_home_command(command) else "open" if is_open_gripper_command(command)
                     else "close" if is_close_gripper_command(command) else None)
        if operation:
            from hardware.runtime import run_fr3_builtin
            try:
                run_fr3_builtin(operation, args)
            except Exception as error:
                print(f"[builtin] stopped: {error}")
            return True
    if is_help_command(command):
        print_help()
        return True

    if is_home_command(command):
        print(f"[builtin] home pose: {HOME_POSE_MMRAD}")
        if not args.execute_robot:
            print("[builtin] dry-run: 未加 --execute_robot，不执行机械臂移动。")
            return True
        arm = init_arm(args.xarm_ip)
        try:
            ok = move_xarm_pose(arm, HOME_POSE_MMRAD, args.move_speed, args.move_acc)
            print("[builtin] home result:", ok)
        finally:
            try:
                arm.disconnect()
            except Exception:
                pass
        return True

    if is_open_gripper_command(command):
        print(f"[builtin] open gripper value={args.grip_open_value}")
        if not args.execute_robot:
            print("[builtin] dry-run: 未加 --execute_robot，不执行夹爪服务。")
            return True
        call_grip_service(
            value=args.grip_open_value,
            ros_ws=args.ros_ws,
            service_name=args.grip_service,
            timeout=args.grip_timeout,
            ignore_error=args.ignore_grip_error,
        )
        return True

    if is_close_gripper_command(command):
        print(f"[builtin] close gripper value={args.grip_close_value}")
        if not args.execute_robot:
            print("[builtin] dry-run: 未加 --execute_robot，不执行夹爪服务。")
            return True
        call_grip_service(
            value=args.grip_close_value,
            ros_ws=args.ros_ws,
            service_name=args.grip_service,
            timeout=args.grip_timeout,
            ignore_error=args.ignore_grip_error,
        )
        return True

    if is_reset_state_command(command):
        for path_text in [args.execution_state_path, args.failure_report_path]:
            path = Path(path_text)
            if path.exists():
                path.unlink()
                print(f"[builtin] removed: {path}")
            else:
                print(f"[builtin] already clean: {path}")
        return True

    if is_clear_cache_command(command):
        from services.runtime_service_cache import close_cached_services

        close_cached_services()
        print("[builtin] service cache cleared.")
        return True

    if is_stop_camera_command(command):
        from services.runtime_service_cache import close_cached_services

        close_cached_services()
        print("[builtin] cached camera/services stopped.")
        return True

    return False


def build_config(args, goal: str, executor_args):
    if args.simple_actions:
        return SimpleAgentSystemConfig(
            goal=goal,
            target_name_bindings=dict(getattr(args, 'next_task_name_bindings', {})),
            hardware_config_path=getattr(args, "hardware_config", None),
            arm_speed_scaling=getattr(args, "arm_speed_scaling", None),
            use_sam_for_grasp=getattr(args, "use_sam_for_grasp", False),
            grasp_selection_mode=args.grasp_selection_mode,
            grasp_random_select_top_k=args.grasp_random_select_top_k,
            grasp_random_seed=args.grasp_random_seed,
            grasp_pose_options={name: getattr(args, 'grasp_' + name) for name in
                ('target_rx', 'target_ry', 'max_rx_tilt_deg', 'max_ry_tilt_deg', 'lam_rx', 'lam_ry')},
            project_root=str(PROJECT_ROOT),
            use_vdm=args.use_vdm,
            use_llm_planner=args.use_llm_planner,
            scene_text=args.scene_text,
            plan_json=args.plan_json,
            plan_md=args.plan_md,
            replan_json=args.replan_json,
            replan_md=args.replan_md,
            generated_code=args.generated_code,
            failure_report_path=args.failure_report_path,
            simple_memory_path=args.simple_memory_path,
            execute_robot=args.execute_robot,
            run_executor=not args.no_run_executor,
            enable_replan=not args.no_replan,
            max_replan_attempts=args.max_replan_attempts,
            resume_from_failure=not args.no_resume_from_failure,
            debug_output=args.debug_output,
            verbose_events=args.verbose_events,
            show_container_observation=not args.hide_container_observation,
            enable_nesyro_gate=not args.disable_nesyro_gate,
            nesyro_gate_threshold=args.nesyro_gate_threshold,
            nesyro_gate_max_probe=args.nesyro_gate_max_probe,
            nesyro_gate_trace_path=args.nesyro_gate_trace_path,
            gate_mode=args.gate_mode,
            gate_experience_log_path=args.gate_experience_log_path,
            learned_gate_checkpoint_path=args.learned_gate_checkpoint,
            enable_vlm_occlusion_gate=args.enable_vlm_occlusion_gate,
            vlm_occlusion_base_url=args.vlm_occlusion_base_url or args.api_base_url,
            vlm_occlusion_api_key=args.vlm_occlusion_api_key or args.api_key,
            vlm_occlusion_model=args.vlm_occlusion_model or args.api_model,
            vlm_occlusion_timeout=args.vlm_occlusion_timeout,
            vlm_occlusion_max_tokens=args.vlm_occlusion_max_tokens,
            vlm_occlusion_temperature=args.vlm_occlusion_temperature,
            enable_occlusion_recovery=args.enable_occlusion_recovery,
            occlusion_recovery_max_attempts=args.occlusion_recovery_max_attempts,
            occlusion_buffer_pose_mmrad=args.occlusion_buffer_pose_mmrad,
            occlusion_buffer_approach_clearance_mm=args.occlusion_buffer_approach_clearance_mm,
            occlusion_blocker_bbox_shrink_ratio=args.occlusion_blocker_bbox_shrink_ratio,
            occlusion_blocker_search_expand_ratio=args.occlusion_blocker_search_expand_ratio,
            occlusion_blocker_min_area_ratio=args.occlusion_blocker_min_area_ratio,
            occlusion_blocker_depth_closer_margin_m=args.occlusion_blocker_depth_closer_margin_m,
            occlusion_blocker_detector_prompts=args.occlusion_blocker_detector_prompts,
            occlusion_blocker_detector_threshold=args.occlusion_blocker_detector_threshold,
            occlusion_blocker_detector_text_threshold=args.occlusion_blocker_detector_text_threshold,
            occlusion_blocker_detector_max_objects=args.occlusion_blocker_detector_max_objects,
            occlusion_allow_unlocalized_target=args.occlusion_allow_unlocalized_target,
            occlusion_blocker_max_image_area_ratio=args.occlusion_blocker_max_image_area_ratio,
            occlusion_blocker_max_target_area_ratio=args.occlusion_blocker_max_target_area_ratio,
            occlusion_unlocalized_vlm_min_confidence=args.occlusion_unlocalized_vlm_min_confidence,
            occlusion_unlocalized_detector_min_confidence=args.occlusion_unlocalized_detector_min_confidence,
            occlusion_min_depth_valid_ratio=args.occlusion_min_depth_valid_ratio,
            occlusion_foreground_depth_band_m=args.occlusion_foreground_depth_band_m,
            occlusion_min_foreground_support_ratio=args.occlusion_min_foreground_support_ratio,
            occlusion_active_view_enabled=not args.disable_occlusion_active_view,
            occlusion_active_view_z_mm=args.occlusion_active_view_z_mm,
            occlusion_active_view_y_offsets_mm=args.occlusion_active_view_y_offsets_mm,
            occlusion_active_view_wait_s=args.occlusion_active_view_wait_s,
            global_serial=args.global_serial,
            wrist_serial=args.wrist_serial,
            camera_warmup_s=args.camera_warmup_s,
            xarm_ip=args.xarm_ip,
            handeye_config=args.handeye_config,
            move_speed=args.move_speed,
            move_acc=args.move_acc,
            visualize_selected_grasp=args.visualize_selected_grasp,
            visualize_bbox_filtered_grasps=args.visualize_bbox_filtered_grasps,
            bbox_filter_margin_px=args.bbox_filter_margin_px,
            bbox_filter_inner_margin_ratio=args.bbox_filter_inner_margin_ratio,
            collision_detection=not args.disable_anygrasp_collision_detection,
            place_slot_spacing_mm=args.place_slot_spacing_mm,
            disable_place_slot_offsets=not args.enable_place_slot_offsets,
            grounding_dino_model=args.grounding_dino_model,
            detector_device=args.detector_device,
            anygrasp_dir=args.anygrasp_dir,
            anygrasp_ckpt=args.anygrasp_ckpt,
            ros_ws=args.ros_ws,
            grip_service=args.grip_service,
            grip_timeout=args.grip_timeout,
            grip_open_value=args.grip_open_value,
            grip_close_value=args.grip_close_value,
            gripper_open_repeat_count=args.gripper_open_repeat_count,
            gripper_open_repeat_interval_s=args.gripper_open_repeat_interval_s,
            gripper_close_repeat_count=args.gripper_close_repeat_count,
            gripper_close_repeat_interval_s=args.gripper_close_repeat_interval_s,
            gripper_close_wait_s=args.gripper_close_wait_s,
            ignore_grip_error=args.ignore_grip_error,
            disable_grip=args.disable_grip,
            verify_place_vdm=(
                args.verify_place_vdm
                or (args.use_vdm and args.execute_robot and not args.no_execution_vdm)
            ),
            place_vdm_stop_on_failure=True,
            place_vdm_min_success_confidence=args.place_vdm_min_success_confidence,
            place_vdm_min_visible_success_confidence=args.place_vdm_min_visible_success_confidence,
            place_vdm_min_failure_confidence=args.place_vdm_min_failure_confidence,
            vdm_base_url=args.grasp_vdm_base_url or args.api_base_url or args.vlm_base_url,
            vdm_api_key=args.grasp_vdm_api_key or args.api_key or args.vlm_api_key,
            vdm_model=args.grasp_vdm_model or args.api_model or args.vlm_model,
            vdm_timeout=args.grasp_vdm_timeout,
            vdm_temperature=args.grasp_vdm_temperature,
            vdm_max_tokens=args.grasp_vdm_max_tokens,
            fallback_vlm_base_url=args.fallback_vlm_base_url,
            fallback_vlm_api_key=args.fallback_vlm_api_key,
            fallback_vlm_model=args.fallback_vlm_model,
            planner_llm_base_url=args.planner_llm_base_url,
            planner_llm_api_key=args.planner_llm_api_key,
            planner_llm_model=args.planner_llm_model,
            vlm_timeout=args.vlm_timeout,
        )
    return AgentSystemConfig(
        goal=goal,
        project_root=str(PROJECT_ROOT),
        use_vdm=args.use_vdm,
        use_llm_planner=args.use_llm_planner,
        scene_text=args.scene_text,
        plan_json=args.plan_json,
        plan_md=args.plan_md,
        replan_json=args.replan_json,
        replan_md=args.replan_md,
        execution_state_path=args.execution_state_path,
        failure_report_path=args.failure_report_path,
        generated_code=args.generated_code,
        execute_robot=args.execute_robot,
        run_executor=not args.no_run_executor,
        enable_replan=not args.no_replan,
        max_replan_attempts=args.max_replan_attempts,
        in_process_executor=not args.subprocess_executor,
        executor_args=executor_args,
        verbose_events=args.verbose_events,
        debug_output=args.debug_output,
        vlm_base_url=args.vlm_base_url,
        vlm_api_key=args.vlm_api_key,
        vlm_model=args.vlm_model,
        vlm_timeout=args.vlm_timeout,
        fallback_vlm_base_url=args.fallback_vlm_base_url,
        fallback_vlm_api_key=args.fallback_vlm_api_key,
        fallback_vlm_model=args.fallback_vlm_model,
        planner_llm_base_url=args.planner_llm_base_url,
        planner_llm_api_key=args.planner_llm_api_key,
        planner_llm_model=args.planner_llm_model,
        global_serial=args.global_serial,
        wrist_serial=args.wrist_serial,
    )


def _has_executor_flag(executor_args, flag: str) -> bool:
    return flag in set(executor_args or [])


def _append_executor_option(executor_args, flag: str, value=None):
    if _has_executor_flag(executor_args, flag):
        return
    executor_args.append(flag)
    if value is not None:
        executor_args.append(str(value))


def build_executor_args(args, raw_executor_args):
    executor_args = list(raw_executor_args or [])
    auto_execution_vdm_enabled = args.use_vdm and args.execute_robot and not args.no_execution_vdm
    grasp_vdm_enabled = bool(args.verify_grasp_vdm)
    place_vdm_enabled = bool(args.verify_place_vdm or auto_execution_vdm_enabled)

    if not (grasp_vdm_enabled or place_vdm_enabled):
        return executor_args

    base_url = args.grasp_vdm_base_url or args.api_base_url or args.vlm_base_url
    api_key = args.grasp_vdm_api_key or args.api_key or args.vlm_api_key
    model = args.grasp_vdm_model or args.api_model or args.vlm_model

    if grasp_vdm_enabled:
        _append_executor_option(executor_args, "--verify_grasp_vdm")
        if not args.no_grasp_vdm_stop_on_failure:
            _append_executor_option(executor_args, "--grasp_vdm_stop_on_failure")
    _append_executor_option(executor_args, "--grasp_vdm_base_url", base_url)
    _append_executor_option(executor_args, "--grasp_vdm_api_key", api_key)
    _append_executor_option(executor_args, "--grasp_vdm_model", model)
    _append_executor_option(executor_args, "--grasp_vdm_timeout", args.grasp_vdm_timeout)
    _append_executor_option(executor_args, "--grasp_vdm_temperature", args.grasp_vdm_temperature)
    _append_executor_option(executor_args, "--grasp_vdm_max_tokens", args.grasp_vdm_max_tokens)
    _append_executor_option(
        executor_args,
        "--grasp_vdm_min_success_confidence",
        args.grasp_vdm_min_success_confidence,
    )
    _append_executor_option(
        executor_args,
        "--grasp_vdm_min_success_score",
        args.grasp_vdm_min_success_score,
    )
    _append_executor_option(
        executor_args,
        "--grasp_vdm_failure_policy",
        args.grasp_vdm_failure_policy,
    )
    _append_executor_option(
        executor_args,
        "--grasp_vdm_local_retry_attempts",
        args.grasp_vdm_local_retry_attempts,
    )
    _append_executor_option(
        executor_args,
        "--grasp_vdm_retry_observe_z_mm",
        args.grasp_vdm_retry_observe_z_mm,
    )
    _append_executor_option(
        executor_args,
        "--grasp_vdm_retry_observe_wait_s",
        args.grasp_vdm_retry_observe_wait_s,
    )
    if args.grasp_vdm_no_contact_sheet:
        _append_executor_option(executor_args, "--grasp_vdm_no_contact_sheet")
    if args.grasp_vdm_contact_sheet:
        _append_executor_option(executor_args, "--grasp_vdm_contact_sheet")
    if place_vdm_enabled:
        _append_executor_option(executor_args, "--verify_place_vdm")
    if args.place_vdm_stop_on_failure or place_vdm_enabled:
        _append_executor_option(executor_args, "--place_vdm_stop_on_failure")
    _append_executor_option(
        executor_args,
        "--place_vdm_min_success_confidence",
        args.place_vdm_min_success_confidence,
    )
    _append_executor_option(
        executor_args,
        "--place_vdm_min_visible_success_confidence",
        args.place_vdm_min_visible_success_confidence,
    )
    _append_executor_option(
        executor_args,
        "--place_vdm_min_failure_confidence",
        args.place_vdm_min_failure_confidence,
    )
    if args.enable_place_slot_offsets:
        _append_executor_option(executor_args, "--place_slot_spacing_mm", args.place_slot_spacing_mm)
    else:
        _append_executor_option(executor_args, "--disable_place_slot_offsets")
    return executor_args


def apply_unified_api_defaults(args):
    if args.api_base_url:
        args.vlm_base_url = args.vlm_base_url or args.api_base_url
        args.planner_llm_base_url = args.planner_llm_base_url or args.api_base_url
        args.fallback_vlm_base_url = args.fallback_vlm_base_url or args.api_base_url
    if args.api_key:
        args.vlm_api_key = args.vlm_api_key or args.api_key
        args.planner_llm_api_key = args.planner_llm_api_key or args.api_key
        args.fallback_vlm_api_key = args.fallback_vlm_api_key or args.api_key
    if args.api_model:
        args.vlm_model = args.vlm_model or args.api_model
        args.planner_llm_model = args.planner_llm_model or args.api_model
        args.fallback_vlm_model = args.fallback_vlm_model or args.api_model


def run_goal(args, goal: str, executor_args) -> bool:
    config = build_config(args, goal=goal, executor_args=executor_args)
    # Consume even on failure: an old scene's temporary name must not leak to a later task.
    args.next_task_name_bindings = {}
    if getattr(args, "hardware_config", None) and not args.no_run_executor:
        from hardware.runtime import require_task_geometry
        try:
            require_task_geometry(config)
        except RuntimeError as error:
            print(f"[agent] {error}")
            return False
    if args.simple_actions:
        result = SimpleArmCodeAgentSystem(config).run()
    else:
        result = ArmCodeAgentSystem(config).run()

    if args.debug_output:
        print("\n=== Agent System Summary ===")
        print("success:", result.success)
        print("plan_json:", result.plan_json)
        print("plan_md:", result.plan_md)
        print("replan_json:", result.replan_json)
        print("replan_md:", result.replan_md)
        print("generated_code:", result.generated_code)
        print("failure_report_path:", result.failure_report_path)
        print("executor_returncode:", result.executor_returncode)
        print("replan_attempts:", result.replan_attempts)
        print("error:", result.error)
    else:
        print("\n=== 任务结果 ===")
        if args.no_run_executor and result.success:
            print("状态: 已生成执行脚本")
        else:
            print("状态:", "完成" if result.success else "未完成")
        if result.replan_attempts:
            print("重规划次数:", result.replan_attempts)
        if not result.success:
            print("失败报告:", result.failure_report_path)
            if result.replan_md:
                print("重规划结果:", result.replan_md)
            print("原因:", result.error)
    return bool(result.success)


def interactive_loop(args, executor_args):
    print("\nArmCode Agent CLI")
    print("输入 /help 查看命令；输入 /quit 退出。")
    print("当前执行模式:", "robot" if args.execute_robot else "dry-run")
    warmup_camera_if_needed(args)

    try:
        while True:
            try:
                if prompt_toolkit_prompt is not None and not args.disable_prompt_toolkit:
                    raw = prompt_toolkit_prompt("\narmcode> ")
                else:
                    raw = input("\narmcode> ")
            except (EOFError, KeyboardInterrupt):
                print("\n退出。")
                return 0

            command = normalize_command(raw)
            if not command:
                continue
            if is_quit_command(command):
                print("退出。")
                return 0
            if run_builtin_command(command, args):
                continue

            ok = run_goal(args, goal=raw.strip(), executor_args=executor_args)
            if not ok:
                print("[agent] 任务失败。你可以调整场景或输入更明确的任务后重试。")
    finally:
        if args.close_services_on_exit:
            from services.runtime_service_cache import close_cached_services

            close_cached_services()


def main():
    parser = argparse.ArgumentParser(
        description="Simple CLI frontend: user command -> planning -> coding -> executing."
    )
    from hardware.config import DEFAULT_CONFIG
    parser.add_argument("--hardware-config", dest="hardware_config", default=str(DEFAULT_CONFIG),
                        help="FR3 配置文件；机械臂/夹爪/相机参数以此文件为准")
    parser.add_argument("--legacy-xarm", dest="legacy_xarm", action="store_true",
                        help="显式使用旧 xArm/ROS 1 链路；不适用于 FR3")
    parser.add_argument(
        "--goal",
        default="",
        help="用户任务，例如：将桌面上所有的药品盒子整理到粉色托盘中",
    )
    parser.add_argument("--scene_text", default="")
    parser.add_argument("--use_vdm", action="store_true")
    parser.add_argument("--use_llm_planner", action="store_true")
    parser.add_argument("--plan_json", default="runs/current_plan/plan_graph.json")
    parser.add_argument("--plan_md", default="runs/current_plan/plan_graph.md")
    parser.add_argument("--replan_json", default="runs/current_plan/replan_graph.json")
    parser.add_argument("--replan_md", default="runs/current_plan/replan_graph.md")
    parser.add_argument("--execution_state_path", default="runs/current_plan/execution_state.json")
    parser.add_argument("--failure_report_path", default="runs/current_plan/failure_report.json")
    parser.add_argument("--simple_memory_path", default="runs/current_plan/simple_execution_memory.json")
    parser.add_argument("--generated_code", default="runs/generated_code/current_plan_executor.py")
    parser.add_argument(
        "--simple_actions",
        action="store_true",
        help="使用新的简短技能链路，生成 grasp/place/observe 动作脚本并在同进程复用模型和相机。",
    )
    parser.add_argument("--no_run_executor", action="store_true")
    parser.add_argument("--no_replan", action="store_true")
    parser.add_argument(
        "--no_resume_from_failure",
        action="store_true",
        help="关闭失败记忆；重规划后从动作序列开头重新执行。",
    )
    parser.add_argument("--max_replan_attempts", type=int, default=1)
    parser.add_argument(
        "--subprocess_executor",
        action="store_true",
        help="使用旧的 subprocess 执行方式；默认同进程执行以复用 GroundingDINO/OCR。",
    )
    parser.add_argument(
        "--disable_prompt_toolkit",
        action="store_true",
        help="禁用 prompt_toolkit，回退到 Python input/readline。",
    )
    parser.add_argument("--warmup_camera", action="store_true")
    parser.add_argument("--no_camera_warmup", action="store_true")
    parser.add_argument("--camera_warmup_s", type=float, default=2.0)
    parser.add_argument(
        "--close_services_on_exit",
        action="store_true",
        help="退出交互 CLI 时关闭缓存相机和模型服务。",
    )
    parser.add_argument("--arm-speed", "--arm-speed-scaling", dest="arm_speed_scaling", type=float,
                        default=None, help="FR3 MoveIt 速度/加速度比例 (0,1]；默认读取硬件 YAML 的 arm.speed_scaling")
    parser.add_argument("--use-sam-for-grasp", action="store_true",
                        help="抓取目标选定后使用保留的 SAM3 分割，GSNet 按 mask 筛选")
    parser.add_argument("--grasp-selection-mode", choices=("best_good", "random_top_k_good", "random_good"),
                        default="best_good", help="姿态约束后按综合评分选最优，或从候选池随机选择")
    parser.add_argument("--grasp-random-select-top-k", type=int, default=5)
    parser.add_argument("--grasp-random-seed", type=int, default=None)
    parser.add_argument("--grasp-target-rx", type=float, default=3.141592653589793, help="基座系法兰目标 roll，弧度")
    parser.add_argument("--grasp-target-ry", type=float, default=0., help="基座系法兰目标 pitch，弧度")
    parser.add_argument("--grasp-max-rx-tilt-deg", type=float, default=65.)
    parser.add_argument("--grasp-max-ry-tilt-deg", type=float, default=65.)
    parser.add_argument("--grasp-lam-rx", type=float, default=1., help="roll 偏差惩罚，按弧度计算")
    parser.add_argument("--grasp-lam-ry", type=float, default=1., help="pitch 偏差惩罚，按弧度计算")
    parser.add_argument("--execute_robot", action="store_true")
    parser.add_argument(
        "--debug_output",
        action="store_true",
        help="显示底层技能、模型和生成脚本的完整调试输出。",
    )
    parser.add_argument(
        "--no_execution_vdm",
        action="store_true",
        help="只使用 VDM 做规划场景描述，不启用执行后的放置结果判别。",
    )
    parser.add_argument(
        "--verify_grasp_vdm",
        action="store_true",
        help="显式启用抓取后的 VDM 判别器；默认不启用，推荐只使用放置后判别间接判断抓取是否成功。",
    )
    parser.add_argument("--no_grasp_vdm_stop_on_failure", action="store_true")
    parser.add_argument("--grasp_vdm_base_url", default=os.getenv("GRASP_VDM_BASE_URL", ""))
    parser.add_argument("--grasp_vdm_api_key", default=os.getenv("GRASP_VDM_API_KEY", ""))
    parser.add_argument("--grasp_vdm_model", default=os.getenv("GRASP_VDM_MODEL", ""))
    parser.add_argument("--grasp_vdm_timeout", type=int, default=120)
    parser.add_argument("--grasp_vdm_temperature", type=float, default=0.0)
    parser.add_argument("--grasp_vdm_max_tokens", type=int, default=500)
    parser.add_argument("--grasp_vdm_min_success_confidence", type=float, default=0.60)
    parser.add_argument("--grasp_vdm_min_success_score", type=float, default=0.60)
    parser.add_argument(
        "--grasp_vdm_failure_policy",
        default="place_then_verify",
        choices=["stop", "place_then_verify", "warn_only"],
        help="抓取 VDM 判失败后的策略：停止、继续放置后验证、或仅警告。",
    )
    parser.add_argument("--grasp_vdm_local_retry_attempts", type=int, default=1)
    parser.add_argument("--grasp_vdm_retry_observe_z_mm", type=float, default=400.0)
    parser.add_argument("--grasp_vdm_retry_observe_wait_s", type=float, default=0.8)
    parser.add_argument(
        "--grasp_vdm_contact_sheet",
        action="store_true",
        help="给验证器额外发送一张 2x2 对比图，更稳但更慢。",
    )
    parser.add_argument("--grasp_vdm_no_contact_sheet", action="store_true")
    parser.add_argument("--verify_place_vdm", action="store_true")
    parser.add_argument("--place_vdm_stop_on_failure", action="store_true")
    parser.add_argument("--place_vdm_min_success_confidence", type=float, default=0.55)
    parser.add_argument("--place_vdm_min_visible_success_confidence", type=float, default=0.35)
    parser.add_argument("--place_vdm_min_failure_confidence", type=float, default=0.75)
    parser.add_argument(
        "--enable_place_slot_offsets",
        action="store_true",
        help="启用多物体放置时的左右错开放置；默认关闭，所有物体放在容器中心。",
    )
    parser.add_argument("--place_slot_spacing_mm", type=float, default=35.0)
    parser.add_argument(
        "--visualize_selected_grasp", "--visualize-selected-grasp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="显示最终选中的抓取位姿；FR3 默认开启。关闭窗口后继续流程，可用 --no-visualize-selected-grasp 关闭预览。",
    )
    parser.add_argument(
        "--visualize_bbox_filtered_grasps",
        action="store_true",
        help="可视化 bbox 范围内筛选出的抓取候选。",
    )
    parser.add_argument(
        "--bbox_filter_margin_px",
        type=int,
        default=0,
        help="抓取候选投影到 bbox 筛选时的外扩像素。默认 0。",
    )
    parser.add_argument(
        "--bbox_filter_inner_margin_ratio",
        type=float,
        default=0.05,
        help="抓取候选 bbox 内缩比例。越大越靠中心；默认 0.05，避免过度排除边缘抓取。",
    )
    parser.add_argument(
        "--disable_anygrasp_collision_detection",
        action="store_true",
        help="关闭 AnyGrasp SDK 碰撞过滤；一般不建议关闭，仅用于对比调试。",
    )
    parser.add_argument(
        "--hide_container_observation",
        action="store_true",
        help="隐藏容器二次观测后的中心坐标和放置位姿摘要。",
    )
    parser.add_argument(
        "--disable_nesyro_gate",
        action="store_true",
        help="关闭规则版 NeSyRo Gate 执行前判定。",
    )
    parser.add_argument("--nesyro_gate_threshold", type=float, default=0.7)
    parser.add_argument("--nesyro_gate_max_probe", type=int, default=3)
    parser.add_argument(
        "--nesyro_gate_trace_path",
        default="runs/current_plan/gate_trace.jsonl",
        help="NeSyRo Gate 每步判定 JSONL trace 输出路径；传空字符串可关闭。",
    )
    parser.add_argument(
        "--gate_mode",
        choices=["legacy", "learned", "shadow"],
        default="legacy",
        help=(
            "NeSyRo Gate 模式：legacy=原 CSCxLC 规则版；"
            "learned=可学习 Gate 决策（缺 checkpoint 自动回退 legacy）；"
            "shadow=legacy 决策但并行记录 learned 输出（不影响机器人行为）。"
        ),
    )
    parser.add_argument(
        "--gate_experience_log_path",
        default="logs/gate_experience/gate_experience.jsonl",
        help="Gate 经验 JSONL 输出路径；传空字符串可关闭。",
    )
    parser.add_argument(
        "--learned_gate_checkpoint",
        default="",
        help="Learnable Gate torch checkpoint 路径（可选）。",
    )
    parser.add_argument(
        "--enable_vlm_occlusion_gate",
        action="store_true",
        help="启用 VLM 抓取前遮挡判定；会在 grasp gate 前调用视觉模型判断是否有物体压住目标。",
    )
    parser.add_argument("--vlm_occlusion_base_url", default=os.getenv("VLM_OCCLUSION_BASE_URL", ""))
    parser.add_argument("--vlm_occlusion_api_key", default=os.getenv("VLM_OCCLUSION_API_KEY", ""))
    parser.add_argument("--vlm_occlusion_model", default=os.getenv("VLM_OCCLUSION_MODEL", ""))
    parser.add_argument("--vlm_occlusion_timeout", type=int, default=120)
    parser.add_argument("--vlm_occlusion_max_tokens", type=int, default=500)
    parser.add_argument("--vlm_occlusion_temperature", type=float, default=0.0)
    parser.add_argument(
        "--enable_occlusion_recovery",
        action="store_true",
        help="Gate 检测到目标遮挡后，自动抓取遮挡物并搬到固定 buffer，再重新观察目标。",
    )
    parser.add_argument("--occlusion_recovery_max_attempts", type=int, default=1)
    parser.add_argument(
        "--occlusion_buffer_pose_mmrad",
        type=float,
        nargs=6,
        default=[280.0, -260.0, 170.0, 3.14, 0.0, -1.57],
    )
    parser.add_argument("--occlusion_buffer_approach_clearance_mm", type=float, default=120.0)
    parser.add_argument("--occlusion_blocker_bbox_shrink_ratio", type=float, default=0.08)
    parser.add_argument("--occlusion_blocker_search_expand_ratio", type=float, default=1.8)
    parser.add_argument("--occlusion_blocker_min_area_ratio", type=float, default=0.50)
    parser.add_argument("--occlusion_blocker_depth_closer_margin_m", type=float, default=0.01)
    parser.add_argument(
        "--occlusion_blocker_detector_prompts",
        nargs="*",
        default=[
            "medicine_box",
            "snack",
            "box",
            "package",
            "object on top of another object",
        ],
    )
    parser.add_argument("--occlusion_blocker_detector_threshold", type=float, default=0.20)
    parser.add_argument("--occlusion_blocker_detector_text_threshold", type=float, default=0.16)
    parser.add_argument("--occlusion_blocker_detector_max_objects", type=int, default=12)
    parser.add_argument(
        "--occlusion_allow_unlocalized_target",
        action="store_true",
        help="目标完全不可定位时仍允许探索性搬走最大疑似遮挡物；默认关闭。",
    )
    parser.add_argument("--occlusion_blocker_max_image_area_ratio", type=float, default=0.50)
    parser.add_argument("--occlusion_blocker_max_target_area_ratio", type=float, default=6.0)
    parser.add_argument("--occlusion_unlocalized_vlm_min_confidence", type=float, default=0.85)
    parser.add_argument("--occlusion_unlocalized_detector_min_confidence", type=float, default=0.25)
    parser.add_argument("--occlusion_min_depth_valid_ratio", type=float, default=0.35)
    parser.add_argument("--occlusion_foreground_depth_band_m", type=float, default=0.018)
    parser.add_argument("--occlusion_min_foreground_support_ratio", type=float, default=0.15)
    parser.add_argument("--disable_occlusion_active_view", action="store_true")
    parser.add_argument("--occlusion_active_view_z_mm", type=float, default=550.0)
    parser.add_argument(
        "--occlusion_active_view_y_offsets_mm",
        type=float,
        nargs="*",
        default=[0.0, -120.0, 120.0],
    )
    parser.add_argument("--occlusion_active_view_wait_s", type=float, default=0.8)
    parser.add_argument("--verbose_events", action="store_true")

    parser.add_argument(
        "--api_base_url",
        default=os.getenv("ARM_AGENT_API_BASE_URL", os.getenv("PLANNER_API_BASE_URL", "")),
        help="统一的 VLM/LLM API base url，planner 和执行验证器默认共用这一项。",
    )
    parser.add_argument(
        "--api_key",
        default=os.getenv("ARM_AGENT_API_KEY", os.getenv("PLANNER_API_KEY", "")),
        help="统一的 VLM/LLM API key，planner 和执行验证器默认共用这一项。",
    )
    parser.add_argument(
        "--api_model",
        default=os.getenv("ARM_AGENT_API_MODEL", os.getenv("PLANNER_API_MODEL", "gpt-4.1")),
        help="统一的 VLM/LLM 模型名，planner 和执行验证器默认共用这一项。",
    )

    parser.add_argument(
        "--vlm_base_url",
        default=os.getenv("PLANNER_LLM_BASE_URL", ""),
    )
    parser.add_argument(
        "--vlm_api_key",
        default=os.getenv("PLANNER_LLM_API_KEY", ""),
    )
    parser.add_argument(
        "--vlm_model",
        default=os.getenv("PLANNER_LLM_MODEL", ""),
    )
    parser.add_argument("--vlm_timeout", type=int, default=120)
    parser.add_argument(
        "--fallback_vlm_base_url",
        default=os.getenv("PLANNER_VDM_FALLBACK_BASE_URL", os.getenv("PLANNER_API_BASE_URL", "")),
    )
    parser.add_argument(
        "--fallback_vlm_api_key",
        default=os.getenv("PLANNER_VDM_FALLBACK_API_KEY", os.getenv("PLANNER_API_KEY", "")),
    )
    parser.add_argument(
        "--fallback_vlm_model",
        default=os.getenv("PLANNER_VDM_FALLBACK_MODEL", os.getenv("PLANNER_API_MODEL", "")),
    )
    parser.add_argument(
        "--planner_llm_base_url",
        default=os.getenv("PLANNER_LLM_FALLBACK_BASE_URL", os.getenv("PLANNER_API_BASE_URL", "")),
    )
    parser.add_argument(
        "--planner_llm_api_key",
        default=os.getenv("PLANNER_LLM_FALLBACK_API_KEY", os.getenv("PLANNER_API_KEY", "")),
    )
    parser.add_argument(
        "--planner_llm_model",
        default=os.getenv("PLANNER_LLM_FALLBACK_MODEL", os.getenv("PLANNER_API_MODEL", "")),
    )

    parser.add_argument("--global_serial", default="213522251050")
    parser.add_argument("--wrist_serial", default="243222070053")
    parser.add_argument("--xarm_ip", default="192.168.1.237")
    parser.add_argument(
        "--handeye_config",
        default="/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
    )
    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)
    parser.add_argument("--grounding_dino_model", default="/home/wpy/models/grounding-dino-tiny")
    parser.add_argument("--detector_device", default="cuda")
    parser.add_argument(
        "--anygrasp_dir",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection",
    )
    parser.add_argument(
        "--anygrasp_ckpt",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar",
    )
    parser.add_argument("--ros_ws", default="/home/wpy/gt200_ws")
    parser.add_argument("--grip_service", default="/grip")
    parser.add_argument("--grip_timeout", type=float, default=5.0)
    parser.add_argument("--grip_open_value", type=int, default=1)
    parser.add_argument("--grip_close_value", type=int, default=0)
    parser.add_argument(
        "--gripper_open_repeat_count",
        type=int,
        default=2,
        help="每次需要打开夹爪时重复调用夹爪打开命令的次数，默认 2。",
    )
    parser.add_argument(
        "--gripper_open_repeat_interval_s",
        type=float,
        default=0.25,
        help="多次夹爪打开命令之间的间隔秒数，默认 0.25。",
    )
    parser.add_argument(
        "--gripper_close_repeat_count",
        type=int,
        default=2,
        help="每次抓取到位后重复调用夹爪闭合命令的次数，默认 2。",
    )
    parser.add_argument(
        "--gripper_close_repeat_interval_s",
        type=float,
        default=0.25,
        help="多次夹爪闭合命令之间的间隔秒数，默认 0.25。",
    )
    parser.add_argument(
        "--gripper_close_wait_s",
        type=float,
        default=1.5,
        help="最后一次夹爪闭合后等待夹爪稳定的秒数，默认 1.5。",
    )
    parser.add_argument("--ignore_grip_error", action="store_true", default=True)
    parser.add_argument("--disable_grip", action="store_true")

    parser.add_argument(
        "executor_args",
        nargs=argparse.REMAINDER,
        help="传给生成脚本的额外参数。用 -- 分隔，例如：-- --visualize_selected_grasp",
    )
    args = parser.parse_args()
    from hardware.agent_geometry import validate_pose_selection
    try:
        validate_pose_selection(**{name: getattr(args, 'grasp_' + name) for name in
            ('target_rx', 'target_ry', 'max_rx_tilt_deg', 'max_ry_tilt_deg', 'lam_rx', 'lam_ry')},
            grasp_selection_mode=args.grasp_selection_mode, random_select_top_k=args.grasp_random_select_top_k,
            random_seed=args.grasp_random_seed)
    except ValueError as error:
        parser.error(str(error))
    from hardware.runtime import apply_cli_profile
    try:
        apply_cli_profile(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.visualize_selected_grasp is None:
        args.visualize_selected_grasp = bool(args.hardware_config)
    apply_unified_api_defaults(args)

    executor_args = list(args.executor_args or [])
    if executor_args and executor_args[0] == "--":
        executor_args = executor_args[1:]
    executor_args = build_executor_args(args, executor_args)

    if args.use_vdm and args.execute_robot and not args.no_execution_vdm:
        print("[agent] execution VDM verifier: enabled")
    elif args.use_vdm:
        print("[agent] planning VDM: enabled; execution VDM verifier: disabled")

    if not args.goal:
        raise SystemExit(interactive_loop(args, executor_args))

    ok = run_goal(args, goal=args.goal, executor_args=executor_args)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
