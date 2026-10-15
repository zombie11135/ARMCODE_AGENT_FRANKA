"""Plan a task, generate actions, execute them and coordinate replanning.

Runtime configuration and service construction live in runtime/.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.agent_system import AgentSystemResult
from agents.planning_agent import PlanningAgent
from agents.replanning_agent import ReplanningAgent
from agents.simple_code_generator import SimpleCodeGenerator
from agents.simple_execution_runtime import SimpleExecutionRuntime
from core.actions import SimpleAction
# Re-export these names so existing callers keep their import paths.
from runtime.camera import TaskCachedCamera as TaskCachedCamera
from runtime.config import SimpleAgentSystemConfig as SimpleAgentSystemConfig
from runtime.services import build_camera, build_execution_context, build_planner_context
from skill_lib.evaluation.verify_medicine_placed_vdm import VerifyMedicinePlacedVDMSkill
from skill_lib.config import SimpleRobotConfig
from skill_lib.simple_actions import SimpleRobotActions


class SimpleArmCodeAgentSystem:
    def __init__(self, config: SimpleAgentSystemConfig):
        self.config = config
        self.project_root = Path(config.project_root).resolve()

    def run(self) -> AgentSystemResult:
        from services.object_names import NameResolutionRequired
        try:
            return self._run_task()
        except NameResolutionRequired as error:
            from services.object_reference_service import atomic_json
            atomic_json(self.config.failure_report_path, {
                'failure_type': 'target_name_ambiguous', 'error': str(error),
                'name_resolution': error.result, 'automatic_replan': False})
            print(f'[name-resolution] {error}')
            return AgentSystemResult(success=False, plan_json=self.config.plan_json,
                plan_md=self.config.plan_md, generated_code=self.config.generated_code,
                failure_report_path=self.config.failure_report_path, executor_returncode=1,
                error=str(error))

    def _run_task(self) -> AgentSystemResult:
        if self.config.run_executor:
            from hardware.runtime import require_task_geometry
            require_task_geometry(self.config)
        graph = self._planning_stage()
        replan_attempts = 0
        replan_paths: Dict[str, Any] = {}
        recovery = bool(self.config.recovery_mode)

        while True:
            actions = self._coding_stage(graph, recovery=recovery)
            if replan_attempts > 0 and self.config.resume_from_failure:
                actions = self._resume_actions_from_failure(actions)
                self._write_resumed_action_script(actions)
            if not self.config.run_executor:
                return AgentSystemResult(
                    success=True,
                    plan_json=self.config.plan_json,
                    plan_md=self.config.plan_md,
                    generated_code=self.config.generated_code,
                    failure_report_path=self.config.failure_report_path,
                    replan_attempts=replan_attempts,
                )

            ok = self._executing_stage(actions)
            if ok:
                return AgentSystemResult(
                    success=True,
                    plan_json=self.config.plan_json,
                    plan_md=self.config.plan_md,
                    generated_code=self.config.generated_code,
                    replan_json=replan_paths.get("replan_json"),
                    replan_md=replan_paths.get("replan_md"),
                    failure_report_path=self.config.failure_report_path,
                    executor_returncode=0,
                    replan_attempts=replan_attempts,
                )

            if not self.config.enable_replan or replan_attempts >= self.config.max_replan_attempts:
                return AgentSystemResult(
                    success=False,
                    plan_json=self.config.plan_json,
                    plan_md=self.config.plan_md,
                    generated_code=self.config.generated_code,
                    replan_json=replan_paths.get("replan_json"),
                    replan_md=replan_paths.get("replan_md"),
                    failure_report_path=self.config.failure_report_path,
                    executor_returncode=1,
                    replan_attempts=replan_attempts,
                    error="simple executor failed",
                )

            replan_paths = self._replanning_stage(graph)
            graph = replan_paths.get("graph") or graph
            recovery = True
            replan_attempts += 1
            print(f"[replanning] retry attempt {replan_attempts}/{self.config.max_replan_attempts}")

    def _planning_stage(self):
        print(f"\n[planning] 任务：{self.config.goal}")
        context = self._build_planner_context()
        planner = PlanningAgent(context=context, use_llm=self.config.use_llm_planner)
        graph = planner.plan(
            task_goal=self.config.goal,
            scene_text=self.config.scene_text,
            use_vdm=self.config.use_vdm,
            output_path=self.config.plan_json,
        )
        md = graph.to_markdown()
        Path(self.config.plan_md).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.plan_md).write_text(md, encoding="utf-8")
        print(f"[planning] 已生成任务图：{len(graph.nodes)} 个节点。")
        print(f"[planning] done: plan={self.config.plan_md}")
        return graph

    def _coding_stage(self, graph, recovery: bool) -> List[SimpleAction]:
        print("\n[coding] 生成简短动作脚本")
        generator = SimpleCodeGenerator(project_root=str(self.project_root))
        generator.generate(
            graph=graph,
            output_path=self.config.generated_code,
            recovery=recovery,
        )
        actions = generator.build_actions(graph=graph, recovery=recovery)
        print("[coding] 即将执行的动作：")
        for index, action in enumerate(actions, start=1):
            print(f"  {index}. {action.op}({action.target})")
        print("[coding] done: 执行脚本已生成")
        return actions

    def _resume_actions_from_failure(self, actions: List[SimpleAction]) -> List[SimpleAction]:
        report_path = Path(self.config.failure_report_path)
        if not report_path.exists():
            return actions

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[memory][WARN] 无法读取失败记忆，完整重试：{e}")
            return actions

        failure_type = str(report.get("failure_type") or "")
        failed_action = report.get("failed_action") or {}
        medicine_query = str(report.get("medicine_query") or "").strip()
        completed_actions = list(report.get("completed_actions") or [])

        if failure_type in {"container_observation_failed", "return_home_failed"}:
            print(f"[memory] 失败类型为 {failure_type}，保持完整动作序列。")
            return actions

        failure_start = self._find_resume_index(
            actions=actions,
            failed_action=failed_action,
            medicine_query=medicine_query,
            failed_action_index=report.get("failed_action_index"),
        )
        completed_start = self._find_completed_prefix_resume_index(
            actions=actions,
            completed_actions=completed_actions,
        )
        start_index = max(failure_start, completed_start)

        if failed_action.get("op") == "place" and medicine_query:
            grasp_index = self._find_grasp_index_for_target(actions, medicine_query)
            if grasp_index is not None:
                start_index = min(start_index, grasp_index)

        if start_index <= 0:
            return actions

        skipped = actions[:start_index]
        resumed = actions[start_index:]
        print("[memory] 使用失败记忆，从失败子目标继续执行。")
        print("[memory] 跳过已完成动作：")
        for action in skipped:
            print(f"  - {action.op}({action.target})")
        print("[memory] 本轮继续动作：")
        for action in resumed:
            print(f"  - {action.op}({action.target})")
        return resumed or actions

    def _write_resumed_action_script(self, actions: List[SimpleAction]) -> None:
        generator = SimpleCodeGenerator(project_root=str(self.project_root))
        script = generator._render_script(actions)
        path = Path(self.config.generated_code)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script, encoding="utf-8")
        print(f"[memory] 已将恢复后的动作脚本写入：{path}")

    @classmethod
    def _find_completed_prefix_resume_index(
        cls,
        actions: List[SimpleAction],
        completed_actions: List[Dict[str, Any]],
    ) -> int:
        action_index = 0
        for completed in completed_actions:
            completed_op = str(completed.get("op") or "")
            completed_target = str(completed.get("target") or "")
            while action_index < len(actions) and actions[action_index].op == "observe":
                action_index += 1
            if action_index >= len(actions):
                break
            action = actions[action_index]
            if not cls._same_action(action, completed_op, completed_target):
                break
            action_index += 1
        return action_index

    @staticmethod
    def _same_action(action: SimpleAction, op: str, target: str) -> bool:
        if action.op != op:
            return False
        action_target = str(action.target)
        target = str(target)
        return action_target == target or action_target in target or target in action_target

    @classmethod
    def _find_grasp_index_for_target(cls, actions: List[SimpleAction], target: str) -> Optional[int]:
        target = str(target or "").strip()
        for index, action in enumerate(actions):
            if action.op == "grasp" and cls._target_matches(str(action.target), target):
                if index > 0 and actions[index - 1].op == "observe":
                    return index - 1
                return index
        return None

    @staticmethod
    def _find_resume_index(
        actions: List[SimpleAction],
        failed_action: Dict[str, Any],
        medicine_query: str = "",
        failed_action_index: Optional[int] = None,
    ) -> int:
        failed_op = str(failed_action.get("op") or "").strip()
        failed_target = str(failed_action.get("target") or "").strip()

        failed_index = None
        try:
            report_index = int(failed_action_index)
        except (TypeError, ValueError):
            report_index = -1
        if 0 <= report_index < len(actions):
            indexed_action = actions[report_index]
            if indexed_action.op == failed_op and SimpleArmCodeAgentSystem._target_matches(
                str(indexed_action.target),
                failed_target,
            ):
                failed_index = report_index
        for index, action in enumerate(actions):
            if failed_index is not None:
                break
            if action.op == failed_op and SimpleArmCodeAgentSystem._target_matches(str(action.target), failed_target):
                failed_index = index
                break

        if failed_op == "place":
            if medicine_query:
                search_start = failed_index if failed_index is not None else len(actions) - 1
                for index in range(search_start, -1, -1):
                    action = actions[index]
                    if action.op == "grasp" and SimpleArmCodeAgentSystem._target_matches(str(action.target), medicine_query):
                        if index > 0 and actions[index - 1].op == "observe":
                            return index - 1
                        return index
            if failed_index is not None:
                for index in range(failed_index - 1, -1, -1):
                    if actions[index].op == "grasp":
                        if index > 0 and actions[index - 1].op == "observe":
                            return index - 1
                        return index

        if failed_op == "grasp" and failed_index is not None:
            if failed_index > 0 and actions[failed_index - 1].op == "observe":
                return failed_index - 1
            return failed_index

        if failed_index is not None:
            return failed_index

        return 0

    @staticmethod
    def _target_matches(left: str, right: str) -> bool:
        left = str(left or "").strip()
        right = str(right or "").strip()
        if not left or not right:
            return False
        return left == right or left in right or right in left

    def _executing_stage(self, actions: List[SimpleAction]) -> bool:
        return SimpleExecutionRuntime(self.config).run(actions)

    def _home_after_place(self, robot: SimpleRobotActions):
        print("[executing] 放置验证完成，回到 home 观察位。")
        result = robot.home()
        if not result.success:
            print(f"[executing][WARN] 回 home 失败：{result.error}")
        return result

    def _build_planner_context(self):
        context = build_planner_context(self.config)
        if self.config.use_vdm:
            camera, _ = self._build_camera()
            camera.start()
            context.camera = camera
        return context

    def _build_execution_context(self):
        camera, _ = self._build_camera()
        return build_execution_context(
            self.config, camera=camera, event_callback=self._event_callback
        )

    def _build_camera(self):
        return build_camera(self.config)

    def _verify_place_if_needed(self, context, camera_service, before_pair, medicine_query, container, step_index) -> bool:
        if not self.config.verify_place_vdm:
            return True
        if not (self.config.vdm_base_url and self.config.vdm_api_key and self.config.vdm_model):
            print("[verifying] 缺少 VDM API 参数，跳过放置判别。")
            return True
        after = self._capture_verify_pair(camera_service, f"runs/current_capture/simple_place_{step_index}_after")
        container_name, container_color = self._split_container_label(container)
        result = VerifyMedicinePlacedVDMSkill(context).run(
            medicine_query=medicine_query,
            container_name=container_name,
            container_color=container_color,
            before_global_color_path=before_pair["global"]["color_path"],
            before_wrist_color_path=before_pair["wrist"]["color_path"],
            after_global_color_path=after["global"]["color_path"],
            after_wrist_color_path=after["wrist"]["color_path"],
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
    def _capture_verify_pair(camera_service, prefix: str) -> Dict[str, Dict[str, Any]]:
        return {
            "global": camera_service.capture_global(),
            "wrist": camera_service.capture_wrist(),
        }

    def _replanning_stage(self, graph) -> Dict[str, Any]:
        print("\n[replanning] build repair constraints from executor failure")
        path = Path(self.config.failure_report_path)
        if not path.exists():
            return {}
        failure_report = ReplanningAgent.load_failure_report(str(path))
        replanner = ReplanningAgent()
        replan_graph = replanner.replan_from_failure_report(
            previous_graph=graph,
            failure_report=failure_report,
            output_json=self.config.replan_json,
            output_md=self.config.replan_md,
        )
        print(f"[replanning] done: plan={self.config.replan_md}")
        return {"graph": replan_graph, "replan_json": self.config.replan_json, "replan_md": self.config.replan_md}

    def _write_failure_report(self, failed_action, medicine_query, failure_type, error, evidence=None):
        report = {
            "success": False,
            "goal": self.config.goal,
            "failed_action": failed_action.__dict__,
            "medicine_query": medicine_query,
            "failure_type": failure_type,
            "error": str(error or ""),
            "evidence": evidence or {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = Path(self.config.failure_report_path)
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

    @staticmethod
    def _skill_failure(action, error):
        from core.skill_base import SkillResult

        return SkillResult(success=False, data={"action": action.__dict__}, error=error)

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
