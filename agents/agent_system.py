import contextlib
import importlib.util
import io
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.code_generator import CodeGenerator
from agents.planning_agent import PlanningAgent
from agents.replanning_agent import ReplanningAgent
from core.context import RuntimeContext


@dataclass
class AgentSystemConfig:
    goal: str
    project_root: str
    use_vdm: bool = False
    use_llm_planner: bool = False
    scene_text: str = ""
    plan_json: str = "runs/current_plan/plan_graph.json"
    plan_md: str = "runs/current_plan/plan_graph.md"
    replan_json: str = "runs/current_plan/replan_graph.json"
    replan_md: str = "runs/current_plan/replan_graph.md"
    execution_state_path: str = "runs/current_plan/execution_state.json"
    failure_report_path: str = "runs/current_plan/failure_report.json"
    generated_code: str = "runs/generated_code/current_plan_executor.py"
    execute_robot: bool = False
    run_executor: bool = True
    enable_replan: bool = True
    max_replan_attempts: int = 1
    in_process_executor: bool = True
    executor_args: List[str] = field(default_factory=list)
    verbose_events: bool = False
    debug_output: bool = False

    vlm_base_url: str = "http://10.201.102.163:8001/v1"
    vlm_api_key: str = "wpywpywpy"
    vlm_model: str = "qwen-vl-8b"
    vlm_timeout: int = 120
    fallback_vlm_base_url: str = ""
    fallback_vlm_api_key: str = ""
    fallback_vlm_model: str = ""
    planner_llm_base_url: str = ""
    planner_llm_api_key: str = ""
    planner_llm_model: str = ""

    global_serial: str = "213522251050"
    wrist_serial: str = "243222070053"


@dataclass
class AgentSystemResult:
    success: bool
    plan_json: str
    plan_md: str
    generated_code: str
    replan_json: Optional[str] = None
    replan_md: Optional[str] = None
    failure_report_path: Optional[str] = None
    executor_returncode: Optional[int] = None
    replan_attempts: int = 0
    error: Optional[str] = None


class ArmCodeAgentSystem:
    def __init__(self, config: AgentSystemConfig):
        self.config = config
        self.project_root = Path(config.project_root).resolve()

    def run(self) -> AgentSystemResult:
        try:
            graph = self._planning_stage()
            returncode = None
            replan_paths = {}
            replan_attempts = 0

            while True:
                self._coding_stage(graph)
                if not self.config.run_executor:
                    break

                returncode = self._executing_stage()
                if returncode == 0:
                    break

                if not self.config.enable_replan:
                    return self._failure_result(
                        returncode=returncode,
                        replan_paths=replan_paths,
                        replan_attempts=replan_attempts,
                    )

                replan_paths = self._replanning_stage(graph)
                replan_graph = replan_paths.get("graph")
                if replan_graph is None or replan_attempts >= self.config.max_replan_attempts:
                    return self._failure_result(
                        returncode=returncode,
                        replan_paths=replan_paths,
                        replan_attempts=replan_attempts,
                    )

                replan_attempts += 1
                graph = replan_graph
                self._stage("replanning", f"retry attempt {replan_attempts}/{self.config.max_replan_attempts}")

            return AgentSystemResult(
                success=True,
                plan_json=self.config.plan_json,
                plan_md=self.config.plan_md,
                generated_code=self.config.generated_code,
                replan_json=replan_paths.get("replan_json"),
                replan_md=replan_paths.get("replan_md"),
                failure_report_path=self.config.failure_report_path,
                executor_returncode=returncode,
                replan_attempts=replan_attempts,
            )

        except Exception as e:
            return AgentSystemResult(
                success=False,
                plan_json=self.config.plan_json,
                plan_md=self.config.plan_md,
                generated_code=self.config.generated_code,
                failure_report_path=self.config.failure_report_path,
                error=str(e),
            )

    def _failure_result(
        self,
        returncode: int,
        replan_paths: Dict[str, Any],
        replan_attempts: int,
    ) -> AgentSystemResult:
        return AgentSystemResult(
            success=False,
            plan_json=self.config.plan_json,
            plan_md=self.config.plan_md,
            generated_code=self.config.generated_code,
            replan_json=replan_paths.get("replan_json"),
            replan_md=replan_paths.get("replan_md"),
            failure_report_path=self.config.failure_report_path,
            executor_returncode=returncode,
            replan_attempts=replan_attempts,
            error=f"executor failed with return code {returncode}",
        )

    def _planning_stage(self):
        self._stage("planning", f"任务：{self.config.goal}")

        camera_service = None
        context = RuntimeContext(
            event_callback=lambda event: self._print_event(
                event,
                verbose=self.config.verbose_events,
            )
        )

        try:
            if self.config.use_vdm:
                from services.realsense_dual_camera_service import (
                    CameraConfig,
                    RealSenseDualCameraService,
                )
                from services.runtime_service_cache import get_or_create_service

                camera_params = {
                    "global_serial": self.config.global_serial,
                    "wrist_serial": self.config.wrist_serial,
                    "capture_dir": "runs/current_capture",
                    "show_window": True,
                    "global_color_width": 1280,
                    "global_color_height": 720,
                    "wrist_color_width": 1280,
                    "wrist_color_height": 720,
                    "wrist_depth_width": 1280,
                    "wrist_depth_height": 720,
                }
                camera_service = get_or_create_service(
                    "dual_realsense_camera",
                    camera_params,
                    lambda: RealSenseDualCameraService(
                        global_config=CameraConfig(
                            name="global",
                            serial=self.config.global_serial,
                            color_width=1280,
                            color_height=720,
                            color_fps=30,
                            enable_depth=False,
                        ),
                        wrist_config=CameraConfig(
                            name="wrist",
                            serial=self.config.wrist_serial,
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
                camera_service.start()
                time.sleep(2.0)
                context.camera = camera_service

            if self.config.use_vdm or self.config.use_llm_planner:
                from services.vlm_client import QwenVLClient

                context.vlm = QwenVLClient(
                    base_url=self.config.vlm_base_url,
                    api_key=self.config.vlm_api_key,
                    model_name=self.config.vlm_model,
                    timeout=self.config.vlm_timeout,
                )

                if (
                    self.config.fallback_vlm_base_url
                    and self.config.fallback_vlm_api_key
                    and self.config.fallback_vlm_model
                ):
                    context.fallback_vlm = QwenVLClient(
                        base_url=self.config.fallback_vlm_base_url,
                        api_key=self.config.fallback_vlm_api_key,
                        model_name=self.config.fallback_vlm_model,
                        timeout=self.config.vlm_timeout,
                    )

                if (
                    self.config.planner_llm_base_url
                    and self.config.planner_llm_api_key
                    and self.config.planner_llm_model
                ):
                    context.planner_llm = QwenVLClient(
                        base_url=self.config.planner_llm_base_url,
                        api_key=self.config.planner_llm_api_key,
                        model_name=self.config.planner_llm_model,
                        timeout=self.config.vlm_timeout,
                    )

            planner = PlanningAgent(
                context=context,
                use_llm=self.config.use_llm_planner,
            )
            graph = planner.plan(
                task_goal=self.config.goal,
                scene_text=self.config.scene_text,
                use_vdm=self.config.use_vdm,
                output_path=self.config.plan_json,
            )

            md = graph.to_markdown()
            plan_md = Path(self.config.plan_md)
            plan_md.parent.mkdir(parents=True, exist_ok=True)
            plan_md.write_text(md, encoding="utf-8")

            if self.config.debug_output:
                print("\n=== Plan Graph ===")
                print(md)
            else:
                executable_count = len([node for node in graph.nodes if node.skill])
                print(f"[planning] 已生成任务图：{len(graph.nodes)} 个节点，{executable_count} 个技能节点。")
            self._done("planning", f"plan={self.config.plan_md}")
            return graph

        finally:
            if camera_service is not None and not self.config.in_process_executor:
                camera_service.stop()

    def _coding_stage(self, graph):
        self._stage("coding", "生成机械臂执行脚本")
        generator = CodeGenerator(project_root=str(self.project_root))
        generator.generate(graph, output_path=self.config.generated_code)
        calls = generator.build_skill_calls(graph)
        if self.config.debug_output:
            print("\n=== Executable Skill Calls ===")
            for index, call in enumerate(calls, start=1):
                color = call.args.get("container_color")
                container_label = call.args["container_name"]
                if color:
                    container_label = f"{color} {container_label}"
                print(
                    f"- {index}. {call.node_id} {call.skill}: "
                    f"{call.args['medicine_query']} -> {container_label}"
                )
            self._done("coding", f"script={self.config.generated_code}")
        else:
            print("[coding] 即将执行的动作：")
            for index, call in enumerate(calls, start=1):
                color = call.args.get("container_color")
                container_label = call.args["container_name"]
                if color:
                    container_label = f"{color} {container_label}"
                print(
                    f"  {index}. 抓取“{call.args['medicine_query']}”"
                    f" -> 放入“{container_label}”"
                )
            self._done("coding", "执行脚本已生成")

    def _executing_stage(self) -> int:
        mode = "真机执行" if self.config.execute_robot else "干跑"
        self._stage("executing", mode)

        cmd = [
            sys.executable,
            "-u",
            self.config.generated_code,
            "--execution_state_path",
            self.config.execution_state_path,
            "--failure_report_path",
            self.config.failure_report_path,
        ]
        if self.config.execute_robot:
            cmd.append("--execute")
        cmd.extend(self.config.executor_args)

        if self.config.in_process_executor:
            if self.config.debug_output:
                print("\n=== Executor Args ===")
                print(" ".join(cmd[2:]))
            else:
                print("[executing] 开始执行机械臂任务。")
            returncode = self._run_executor_in_process(cmd[2:])
        elif self.config.debug_output:
            print("\n=== Executor Command ===")
            print(" ".join(cmd))
            returncode = subprocess.run(cmd, cwd=str(self.project_root)).returncode
        else:
            print("[executing] 开始执行机械臂任务。")
            returncode = self._run_executor_filtered(cmd)
        self._done("executing", f"returncode={returncode}")
        return int(returncode)

    def _run_executor_in_process(self, executor_argv: List[str]) -> int:
        script_path = Path(executor_argv[0]).resolve()
        module_name = f"_armcode_generated_executor_{int(time.time() * 1000)}"
        old_argv = sys.argv[:]
        old_cwd = Path.cwd()
        frontend_stdout = sys.stdout

        class _FrontendWriter(io.TextIOBase):
            def __init__(self, owner, output):
                self.owner = owner
                self.output = output
                self.buffer = ""

            def _emit(self, line):
                with contextlib.redirect_stdout(self.output):
                    self.owner._print_executor_frontend_line(line)

            def write(self, text):
                self.buffer += str(text)
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    self._emit(line)
                return len(text)

            def flush(self):
                if self.buffer.strip():
                    self._emit(self.buffer)
                self.buffer = ""

        try:
            sys.argv = list(executor_argv)
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import generated executor: {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if self.config.debug_output:
                module.main()
            else:
                writer = _FrontendWriter(self, frontend_stdout)
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    module.main()
                writer.flush()
            return 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            return int(code)
        finally:
            sys.argv = old_argv
            try:
                import os

                os.chdir(old_cwd)
            except Exception:
                pass

    def _run_executor_filtered(self, cmd: List[str]) -> int:
        process = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for raw_line in process.stdout:
            self._print_executor_frontend_line(raw_line.rstrip())
        return int(process.wait())

    @staticmethod
    def _print_executor_frontend_line(line: str):
        line = str(line or "").strip()
        if not line:
            return

        if line.startswith("[resume]"):
            print(line)
            return

        if line.startswith("[Warning]") or line.startswith("[Error]") or "[ERROR]" in line:
            print(line)
            return

        if line.startswith("=== Container Observation"):
            print("[executing] 定位放置容器。")
            return

        if line.startswith("move_pre_grasp_pose_after_container_observation:"):
            print("[executing] 容器位姿已缓存，回到抓取初始位。")
            return

        if line.startswith("return_home_after_step_pose:"):
            print("[executing] 子目标完成，回到抓取初始位。")
            return

        if line.startswith("=== Step "):
            print("[executing] 执行下一个抓放子目标。")
            return

        if line.startswith("medicine_query:"):
            print(f"[executing] 目标药品：{line.split(':', 1)[1].strip()}")
            return

        if line.startswith("container_color:") and not line.endswith("None"):
            print(f"[executing] 目标容器颜色：{line.split(':', 1)[1].strip()}")
            return

        if line.startswith("=== Grasp VDM Verification"):
            print("[verifying] 检查抓取是否成功。")
            return

        if line.startswith("grasp_vdm_policy:"):
            print("[verifying] 抓取验证不确定，继续放置后再验证。")
            return

        if line.startswith("place_vdm_policy: weak_negative_continue"):
            print("[verifying] 放置判定证据不足，按成功继续。")
            return

        if line.startswith("place_slot_offset_y_mm:"):
            print(f"[executing] 放置位置偏移：Y {line.split(':', 1)[1].strip()} mm")
            return

        if line.startswith("=== Place VDM Verification"):
            print("[verifying] 检查目标是否已放入容器。")
            return

        if line.startswith("=== Local Grasp Retry"):
            print("[executing] 抓取未成功，移动到目标上方进行局部重试。")
            return

        if line.startswith("move_vdm_retry_observe_pose:"):
            print("[executing] 移动到目标二次观察位。")
            return

        if line.startswith("local_retry_result: success"):
            print("[executing] 局部重试抓取成功。")
            return

        if line.startswith("local_retry_error:"):
            print(f"[executing] 局部重试失败：{line.split(':', 1)[1].strip()}")
            return

        if line.startswith("failure_type:") and not line.endswith("None"):
            print(f"[verifying] 失败类型：{line.split(':', 1)[1].strip()}")
            return

        if line.startswith("success_score:") and not line.endswith("None"):
            print(f"[verifying] 成功证据分：{line.split(':', 1)[1].strip()}")
            return

        if line.startswith("recommendation:") and not line.endswith("None"):
            print(f"[verifying] 建议：{line.split(':', 1)[1].strip()}")
            return

        if line.startswith("=== Step Failed"):
            print("[executing] 子目标失败，准备生成失败报告。")
            return

        if line.startswith("=== Cached Place Failed"):
            print("[executing] 放置失败，准备生成失败报告。")
            return

        if line.startswith("=== Failure Report"):
            print("[failure] 已生成失败报告。")
            return

        if line.startswith("path:"):
            print(f"[failure] 报告路径：{line.split(':', 1)[1].strip()}")
            return

        if line.startswith("=== Step Result"):
            print("[executing] 子目标完成。")
            return

        if line.startswith("=== Plan Execution Summary"):
            print("[executing] 执行结果已汇总。")
            return

    def _replanning_stage(self, graph) -> Dict[str, Any]:
        self._stage("replanning", "build repair constraints from executor failure")
        failure_report_path = Path(self.config.failure_report_path)
        if not failure_report_path.exists():
            self._done("replanning", f"skip: missing {failure_report_path}")
            return {}

        try:
            failure_report = ReplanningAgent.load_failure_report(str(failure_report_path))
            replanner = ReplanningAgent()
            replan_graph = replanner.replan_from_failure_report(
                previous_graph=graph,
                failure_report=failure_report,
                output_json=self.config.replan_json,
                output_md=self.config.replan_md,
            )
            if self.config.debug_output:
                print("\n=== Replan Graph ===")
                print(replan_graph.to_markdown())
            else:
                failure_type = failure_report.get("failure_type") or "unknown"
                failed_node_id = failure_report.get("failed_node_id") or "-"
                print(f"[replanning] 已根据失败信息更新计划：node={failed_node_id}, type={failure_type}")
            self._done(
                "replanning",
                f"plan={self.config.replan_md}",
            )
            return {
                "graph": replan_graph,
                "replan_json": self.config.replan_json,
                "replan_md": self.config.replan_md,
            }
        except Exception as e:
            self._done("replanning", f"failed: {e}")
            return {}

    @staticmethod
    def _print_event(event: Dict[str, Any], verbose: bool = False):
        event_type = event.get("type")
        if not verbose and event_type not in ["warning", "error"]:
            return

        if event_type == "image":
            print(f"[Image] {event.get('content')}: {event.get('image_path')}")
        elif event_type == "text":
            print(f"[Text] {event.get('content')}")
        elif event_type == "warning":
            print(f"[Warning] {event.get('content')}")
        elif event_type == "error":
            print(f"[Error] {event.get('content')}")
        else:
            print(event)

    @staticmethod
    def _stage(name: str, message: str):
        print(f"\n[{name}] {message}")

    @staticmethod
    def _done(name: str, message: str):
        print(f"[{name}] done: {message}")
