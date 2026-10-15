import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.plan_graph import PlanGraph, PlanNode


@dataclass
class GeneratedSkillCall:
    node_id: str
    label: str
    skill: str
    args: Dict[str, Any]


class CodeGenerator:
    """
    Generate a sequential Python executor from a PlanGraph.

    The generated script is intentionally conservative for a single xArm:
    graph branches are linearized by topological order, and only supported
    executable skill nodes are emitted.
    """

    SUPPORTED_SKILLS = {
        "pick_medicine_and_place_to_container",
    }

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path.cwd()).resolve()

    def generate(
        self,
        graph: PlanGraph,
        output_path: Optional[str] = None,
    ) -> str:
        calls = self.build_skill_calls(graph)
        if not calls:
            raise ValueError("PlanGraph 中没有可生成执行代码的技能节点。")

        script = self._render_script(graph=graph, calls=calls)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script, encoding="utf-8")

        return script

    def build_skill_calls(self, graph: PlanGraph) -> List[GeneratedSkillCall]:
        return self._build_skill_calls(graph)

    def _build_skill_calls(self, graph: PlanGraph) -> List[GeneratedSkillCall]:
        nodes = self._topological_nodes(graph)

        calls = []
        invalid_nodes = []
        for node in nodes:
            if node.skill not in self.SUPPORTED_SKILLS:
                continue

            args = self._normalize_pick_place_args(node)
            validation_error = self._validate_pick_place_call(node, args)
            if validation_error:
                invalid_nodes.append(validation_error)
                continue

            calls.append(
                GeneratedSkillCall(
                    node_id=node.node_id,
                    label=node.label,
                    skill=node.skill,
                    args=args,
                )
            )

        if invalid_nodes:
            details = "\n".join([f"- {item}" for item in invalid_nodes])
            raise ValueError(
                "PlanGraph 中存在无法安全生成抓取代码的节点：\n"
                f"{details}\n"
                "请重新规划：pick_medicine_and_place_to_container 的 medicine_query "
                "必须是药盒上的可读药名/文字，不能是颜色、形状或方位描述。"
            )

        # If a macro root expands into granular children, prefer the explicit
        # pick/place macro calls only once. Current LLM plans already produce
        # one macro node per medicine target, so this mainly protects heuristic
        # or hand-authored plans from duplicate execution.
        seen = set()
        unique_calls = []
        for call in calls:
            key = (
                call.skill,
                call.args.get("medicine_query"),
                call.args.get("container_name"),
                call.args.get("container_color"),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_calls.append(call)
        return unique_calls

    def _topological_nodes(self, graph: PlanGraph) -> List[PlanNode]:
        node_by_id = {node.node_id: node for node in graph.nodes}
        order_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
        indegree = {node.node_id: 0 for node in graph.nodes}
        outgoing = {node.node_id: [] for node in graph.nodes}

        for edge in graph.edges:
            if edge.source not in node_by_id or edge.target not in node_by_id:
                continue
            outgoing[edge.source].append(edge.target)
            indegree[edge.target] += 1

        ready = sorted(
            [node_id for node_id, degree in indegree.items() if degree == 0],
            key=lambda node_id: order_index[node_id],
        )
        result = []

        while ready:
            node_id = ready.pop(0)
            result.append(node_by_id[node_id])
            for target in sorted(outgoing[node_id], key=lambda item: order_index[item]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort(key=lambda item: order_index[item])

        if len(result) != len(graph.nodes):
            return list(graph.nodes)
        return result

    def _normalize_pick_place_args(self, node: PlanNode) -> Dict[str, Any]:
        args = dict(node.args or {})
        text = "\n".join([node.label or "", node.instruction or ""])

        medicine_query = (
            args.get("medicine_query")
            or args.get("query")
            or args.get("medicine_name")
            or args.get("medicine")
            or args.get("target")
            or self._infer_medicine_query(text)
        )
        container_name = (
            args.get("container_name")
            or args.get("container")
            or args.get("target_container")
            or self._infer_container_name(text)
        )
        container_color = (
            args.get("container_color")
            or args.get("target_container_color")
            or args.get("color")
            or self._infer_container_color(str(container_name))
            or self._infer_container_color(text)
        )

        return {
            "medicine_query": str(medicine_query or "目标药盒").strip(),
            "container_name": self._canonical_container_name(container_name),
            "container_color": self._canonical_container_color(container_color),
            "source_args": args,
        }

    @classmethod
    def _validate_pick_place_call(cls, node: PlanNode, args: Dict[str, Any]) -> Optional[str]:
        medicine_query = str(args.get("medicine_query") or "").strip()
        if not medicine_query:
            return f"{node.node_id} {node.label}: 缺少 medicine_query。"

        if cls._looks_like_non_box_object(medicine_query, node):
            return (
                f"{node.node_id} {node.label}: “{medicine_query}” 看起来不是药盒名称，"
                "当前抓取技能只支持药盒/药品包装盒。"
            )

        if cls._looks_like_visual_description(medicine_query):
            return (
                f"{node.node_id} {node.label}: medicine_query=“{medicine_query}” 是颜色/形状/方位描述，"
                "OCR 匹配不稳定。请让 planner 使用药盒上的文字，例如 布洛芬/抗病毒口服液。"
            )

        return None

    @staticmethod
    def _looks_like_visual_description(text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return True

        known_text_tokens = [
            "布洛芬",
            "抗病毒",
            "口服液",
            "蒙脱石",
            "西瓜霜",
            "胶囊",
            "颗粒",
            "片",
            "OTC",
            "IBUPROFEN",
            "999",
        ]
        if any(token.lower() in text.lower() for token in known_text_tokens):
            return False

        visual_tokens = [
            "白色",
            "蓝色",
            "绿色",
            "红色",
            "黄色",
            "黑色",
            "粉色",
            "左边",
            "右边",
            "中间",
            "前方",
            "后方",
            "上方",
            "下方",
            "较大",
            "较小",
            "长方形",
            "矩形",
            "盒子",
            "药盒",
            "包装",
        ]
        visual_hit_count = sum(1 for token in visual_tokens if token in text)
        return visual_hit_count > 0 and len(text) <= 12

    @staticmethod
    def _looks_like_non_box_object(text: str, node: PlanNode) -> bool:
        joined = "\n".join([text, node.label or "", node.instruction or ""])
        non_box_tokens = [
            "圆柱",
            "管状",
            "瓶",
            "饮料",
            "线材",
            "电缆",
            "急停",
        ]
        return any(token in joined for token in non_box_tokens)

    @staticmethod
    def _infer_medicine_query(text: str) -> str:
        known = [
            "布洛芬缓释胶囊",
            "布洛芬",
            "抗病毒口服液",
            "抗病毒",
            "西瓜霜",
            "蒙脱石散",
            "999",
        ]
        for name in known:
            if name in text:
                return name

        patterns = [
            r"将(.+?)(?:药盒|药品)?(?:放入|放到|放置到)",
            r"抓取(.+?)(?:药盒|药品)?(?:并|然后|放)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(" “”。")

        return "目标药盒"

    @staticmethod
    def _infer_container_name(text: str) -> str:
        if any(token in text for token in ["粉色圆盘", "圆盘", "托盘", "盘子"]):
            return "tray"
        if any(token in text for token in ["收纳盒", "容器", "盒子"]):
            return "container"
        if "碗" in text:
            return "bowl"
        return "tray"

    @staticmethod
    def _infer_container_color(text: str) -> Optional[str]:
        color_aliases = [
            ("pink", ["粉色", "粉红色", "粉红", "pink"]),
            ("red", ["红色", "red"]),
            ("blue", ["蓝色", "blue"]),
            ("green", ["绿色", "green"]),
            ("yellow", ["黄色", "yellow"]),
            ("orange", ["橙色", "orange"]),
            ("purple", ["紫色", "purple"]),
            ("white", ["白色", "white"]),
            ("black", ["黑色", "black"]),
            ("gray", ["灰色", "grey", "gray"]),
            ("brown", ["棕色", "褐色", "brown"]),
        ]
        container_tokens = ["盘", "托盘", "盘子", "容器", "盒子", "碗", "tray", "plate", "container", "box", "bowl"]
        for canonical, aliases in color_aliases:
            for alias in aliases:
                for token in container_tokens:
                    if f"{alias}{token}" in text or f"{alias}的{token}" in text:
                        return canonical
        return None

    @staticmethod
    def _canonical_container_name(value: Any) -> str:
        text = str(value or "").strip()
        if any(token in text for token in ["粉色圆盘", "圆盘", "托盘", "盘子", "tray", "plate"]):
            return "tray"
        if any(token in text for token in ["收纳盒", "整理盒", "透明盒", "半透明盒", "容器", "盒子", "container", "box", "bin", "organizer"]):
            return "container"
        if any(token in text for token in ["碗", "bowl"]):
            return "bowl"
        return text or "tray"

    @staticmethod
    def _canonical_container_color(value: Any) -> Optional[str]:
        text = str(value or "").strip().lower()
        if not text:
            return None
        if any(token in text for token in ["透明", "半透明", "transparent", "translucent", "clear"]):
            return None
        aliases = {
            "pink": ["pink", "magenta", "rose", "粉", "粉色", "粉红", "粉红色"],
            "red": ["red", "红", "红色"],
            "blue": ["blue", "蓝", "蓝色"],
            "green": ["green", "绿", "绿色"],
            "yellow": ["yellow", "黄", "黄色"],
            "orange": ["orange", "橙", "橙色"],
            "purple": ["purple", "violet", "紫", "紫色"],
            "white": ["white", "白", "白色"],
            "black": ["black", "黑", "黑色"],
            "gray": ["gray", "grey", "灰", "灰色"],
            "brown": ["brown", "棕", "棕色", "褐", "褐色"],
        }
        for canonical, names in aliases.items():
            if any(name in text for name in names):
                return canonical
        return text

    def _render_script(self, graph: PlanGraph, calls: List[GeneratedSkillCall]) -> str:
        calls_json = json.dumps([call.__dict__ for call in calls], ensure_ascii=False, indent=2)
        goal_json = json.dumps(graph.goal, ensure_ascii=False)
        project_root_json = json.dumps(str(self.project_root), ensure_ascii=False)

        return f'''#!/usr/bin/env python3
# Auto-generated by agents.code_generator.CodeGenerator.
# Goal: {graph.goal}

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint


PROJECT_ROOT = Path({project_root_json})
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.context import RuntimeContext
from services.grounding_dino_detector_service import GroundingDINODetectorService
from services.ocr_service import PaddleOCRService
from services.realsense_dual_camera_service import CameraConfig, RealSenseDualCameraService
from services.anygrasp_service import AnyGraspService
from services.runtime_service_cache import get_or_create_service
from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
from skill_lib.combound.place_to_container import PlaceToContainerSkill
from skill_lib.evaluation.verify_medicine_grasp_vdm import VerifyMedicineGraspVDMSkill
from skill_lib.evaluation.verify_medicine_placed_vdm import VerifyMedicinePlacedVDMSkill


GOAL = {goal_json}
PLAN_STEPS = {calls_json}


def print_event(event: dict, verbose: bool = False):
    event_type = event.get("type")
    if not verbose and event_type not in ["warning", "error"]:
        return

    if event_type == "image":
        print(f"[Image] {{event.get('content')}}: {{event.get('image_path')}}")
    elif event_type == "text":
        print(f"[Text] {{event.get('content')}}")
    elif event_type == "warning":
        print(f"[Warning] {{event.get('content')}}")
    elif event_type == "error":
        print(f"[Error] {{event.get('content')}}")
    else:
        print(event)


def check_code(arm, code, label: str = "") -> bool:
    if code != 0:
        print(f"[xArm][ERROR] {{label}} failed, code={{code}}")
        print(f"[xArm] state={{arm.state}}, error={{arm.error_code}}, warn={{arm.warn_code}}")
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


def call_grip_service(
    value: int,
    ros_ws: str,
    service_name: str,
    timeout: float,
    ignore_error: bool,
) -> bool:
    cmd = f"source {{ros_ws}}/devel/setup.bash && rosservice call {{service_name}} {{int(value)}}"
    print(f"[Gripper] {{cmd}}")
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
            print(f"[Gripper][WARN] rosservice returned code={{result.returncode}}")
            return bool(ignore_error)
        return True
    except subprocess.TimeoutExpired:
        print(f"[Gripper][WARN] rosservice timeout: {{service_name}}")
        return bool(ignore_error)
    except Exception as e:
        print(f"[Gripper][WARN] rosservice failed: {{e}}")
        return bool(ignore_error)


def build_context(args):
    camera_params = {{
        "global_serial": args.global_serial,
        "wrist_serial": args.wrist_serial,
        "capture_dir": args.capture_dir,
        "show_window": True,
        "global_color_width": 1280,
        "global_color_height": 720,
        "wrist_color_width": 1280,
        "wrist_color_height": 720,
        "wrist_depth_width": 1280,
        "wrist_depth_height": 720,
    }}
    camera_service = get_or_create_service(
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
            save_dir=args.capture_dir,
            show_window=True,
        ),
    )

    detector_params = {{
        "model_name_or_path": args.grounding_dino_model,
        "device": args.detector_device,
        "threshold": min(args.container_box_threshold, args.grasp_box_threshold),
        "text_threshold": min(args.container_text_threshold, args.grasp_text_threshold),
        "nms_iou_threshold": 0.50,
        "local_files_only": True,
        "use_fp16": False,
    }}
    detector_service = get_or_create_service(
        "grounding_dino",
        detector_params,
        lambda: GroundingDINODetectorService(**detector_params),
    )

    ocr_params = {{
        "lang": "ch",
        "device": "cpu",
        "use_textline_orientation": True,
    }}
    ocr_service = get_or_create_service(
        "paddle_ocr",
        ocr_params,
        lambda: PaddleOCRService(**ocr_params),
    )

    anygrasp_service = AnyGraspService(
        grasp_detection_dir=args.anygrasp_dir,
        checkpoint_path=args.anygrasp_ckpt,
        max_gripper_width=0.10,
        gripper_height=0.03,
        top_down_grasp=False,
        debug=False,
    )

    context = RuntimeContext(
        camera=camera_service,
        detector=detector_service,
        ocr=ocr_service,
        anygrasp=anygrasp_service,
        event_callback=lambda event: print_event(event, verbose=args.verbose_events),
    )
    return context, camera_service


def snapshot_capture(capture, output_prefix):
    stable = dict(capture or {{}})
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    for key, suffix in [
        ("color_path", "_color.png"),
        ("depth_path", "_depth.png"),
        ("depth_npy_path", "_depth.npy"),
        ("depth_meter_path", "_depth_meter.npy"),
    ]:
        src = stable.get(key)
        if not src:
            continue
        src_path = Path(src)
        if not src_path.exists():
            continue
        dst_path = Path(f"{{output_prefix}}{{suffix}}")
        if src_path.resolve() != dst_path.resolve():
            shutil.copy2(src_path, dst_path)
        stable[key] = str(dst_path)
    return stable


def capture_verification_pair(camera_service, output_prefix):
    global_capture = snapshot_capture(
        camera_service.capture_global(),
        f"{{output_prefix}}_global",
    )
    wrist_capture = snapshot_capture(
        camera_service.capture_wrist(),
        f"{{output_prefix}}_wrist",
    )
    return global_capture, wrist_capture


def verify_grasp_vdm_if_enabled(
    context,
    camera_service,
    args,
    medicine_query,
    before_global,
    before_wrist,
    step_id,
):
    if not args.verify_grasp_vdm:
        return None
    if not (args.grasp_vdm_base_url and args.grasp_vdm_api_key and args.grasp_vdm_model):
        raise RuntimeError(
            "--verify_grasp_vdm requires --grasp_vdm_base_url, "
            "--grasp_vdm_api_key and --grasp_vdm_model "
            "(or GRASP_VDM_BASE_URL / GRASP_VDM_API_KEY / GRASP_VDM_MODEL)."
        )

    after_global, after_wrist = capture_verification_pair(
        camera_service,
        f"runs/current_capture/grasp_verify_{{step_id}}_after",
    )

    result = VerifyMedicineGraspVDMSkill(context).run(
        medicine_query=medicine_query,
        before_global_color_path=before_global["color_path"],
        before_wrist_color_path=before_wrist["color_path"],
        after_global_color_path=after_global["color_path"],
        after_wrist_color_path=after_wrist["color_path"],
        base_url=args.grasp_vdm_base_url,
        api_key=args.grasp_vdm_api_key,
        model_name=args.grasp_vdm_model,
        temperature=args.grasp_vdm_temperature,
        max_tokens=args.grasp_vdm_max_tokens,
        timeout=args.grasp_vdm_timeout,
        min_success_confidence=args.grasp_vdm_min_success_confidence,
        min_success_score=args.grasp_vdm_min_success_score,
        use_contact_sheet=args.grasp_vdm_contact_sheet and not args.grasp_vdm_no_contact_sheet,
    )

    print("\\n=== Grasp VDM Verification ===")
    print("success:", result.success)
    print("failure_type:", result.data.get("failure_type") if result.data else None)
    print("recommendation:", result.data.get("recommendation") if result.data else None)
    print("confidence:", result.data.get("confidence") if result.data else None)
    print("success_score:", result.data.get("success_score") if result.data else None)
    print("error:", result.error)
    return result


def verify_place_vdm_if_enabled(
    context,
    camera_service,
    args,
    medicine_query,
    container_name,
    container_color,
    before_global,
    before_wrist,
    step_id,
):
    if not args.verify_place_vdm:
        return None
    if not (args.grasp_vdm_base_url and args.grasp_vdm_api_key and args.grasp_vdm_model):
        raise RuntimeError(
            "--verify_place_vdm requires --grasp_vdm_base_url, "
            "--grasp_vdm_api_key and --grasp_vdm_model."
        )

    after_global, after_wrist = capture_verification_pair(
        camera_service,
        f"runs/current_capture/place_verify_{{step_id}}_after",
    )

    result = VerifyMedicinePlacedVDMSkill(context).run(
        medicine_query=medicine_query,
        container_name=container_name,
        container_color=container_color,
        before_global_color_path=before_global["color_path"],
        before_wrist_color_path=before_wrist["color_path"],
        after_global_color_path=after_global["color_path"],
        after_wrist_color_path=after_wrist["color_path"],
        base_url=args.grasp_vdm_base_url,
        api_key=args.grasp_vdm_api_key,
        model_name=args.grasp_vdm_model,
        temperature=args.grasp_vdm_temperature,
        max_tokens=args.grasp_vdm_max_tokens,
        timeout=args.grasp_vdm_timeout,
        min_success_confidence=args.place_vdm_min_success_confidence,
        min_visible_success_confidence=args.place_vdm_min_visible_success_confidence,
    )

    print("\\n=== Place VDM Verification ===")
    print("success:", result.success)
    print("failure_type:", result.data.get("failure_type") if result.data else None)
    print("recommendation:", result.data.get("recommendation") if result.data else None)
    print("confidence:", result.data.get("confidence") if result.data else None)
    print("error:", result.error)
    return result


def group_steps_by_container(steps):
    groups = []
    group_by_container = {{}}
    for step in steps:
        container_name = step["args"]["container_name"]
        container_color = step["args"].get("container_color")
        container_key = f"{{container_name}}::{{container_color or ''}}"
        if container_key not in group_by_container:
            group = {{
                "container_key": container_key,
                "container_name": container_name,
                "container_color": container_color,
                "steps": [],
            }}
            group_by_container[container_key] = group
            groups.append(group)
        group_by_container[container_key]["steps"].append(step)
    return groups


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def load_json_file(path, default):
    json_path = Path(path)
    if not json_path.exists():
        return default
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ExecutionState][WARN] failed to load {{json_path}}: {{e}}")
        return default


def save_json_file(path, data):
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(json_path)


def new_execution_state():
    return {{
        "goal": GOAL,
        "completed_node_ids": [],
        "completed_targets": [],
        "cached_place_plans": {{}},
        "retry_count_by_node": {{}},
        "failed_node_id": None,
        "failure_type": None,
        "failure_reason": None,
        "last_failure_report": {{}},
        "updated_at": now_iso(),
    }}


def container_cache_key(container_name, container_color=None):
    return f"{{container_name}}__{{container_color}}" if container_color else str(container_name)


def cached_place_plan_path(args, container_name, container_color=None):
    key = container_cache_key(container_name, container_color)
    safe_name = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in str(key))
    base = Path(args.execution_state_path).parent
    return str(base / f"cached_place_plan_{{safe_name}}.json")


def load_execution_state(args):
    if args.reset_execution_state:
        return new_execution_state()
    state = load_json_file(args.execution_state_path, new_execution_state())
    if state.get("goal") and state.get("goal") != GOAL:
        print("[resume] previous execution state belongs to another goal, reset it.")
        return new_execution_state()
    state.setdefault("goal", GOAL)
    state.setdefault("completed_node_ids", [])
    state.setdefault("completed_targets", [])
    state.setdefault("cached_place_plans", {{}})
    state.setdefault("retry_count_by_node", {{}})
    state.setdefault("last_failure_report", {{}})
    state["failed_node_id"] = None
    state["failure_type"] = None
    state["failure_reason"] = None
    state["updated_at"] = now_iso()
    return state


def can_resume_from_state(args):
    return bool(args.execute or args.track_dry_run_state)


def mark_step_completed(args, state, step, medicine_query):
    if not can_resume_from_state(args):
        return
    node_id = step["node_id"]
    if node_id not in state["completed_node_ids"]:
        state["completed_node_ids"].append(node_id)
    if medicine_query and medicine_query not in state["completed_targets"]:
        state["completed_targets"].append(medicine_query)
    state["failed_node_id"] = None
    state["failure_type"] = None
    state["failure_reason"] = None
    state["updated_at"] = now_iso()
    save_json_file(args.execution_state_path, state)


def should_skip_step(args, state, step, medicine_query):
    if args.no_resume or not can_resume_from_state(args):
        return False
    completed_targets = set(state.get("completed_targets", []))
    if medicine_query and medicine_query in completed_targets:
        return True
    return False


def load_cached_place_plan(args, state, container_name, container_color=None):
    key = container_cache_key(container_name, container_color)
    cached = state.get("cached_place_plans", {{}}).get(key)
    if cached:
        return cached
    return load_json_file(cached_place_plan_path(args, container_name, container_color), None)


def save_cached_place_plan(args, state, container_name, container_color, place_plan):
    key = container_cache_key(container_name, container_color)
    state.setdefault("cached_place_plans", {{}})[key] = place_plan
    save_json_file(cached_place_plan_path(args, container_name, container_color), place_plan)
    if can_resume_from_state(args):
        save_json_file(args.execution_state_path, state)


def infer_grasp_failure_type(result):
    data = result.data or {{}}
    stage = str(data.get("stage") or data.get("failed_stage") or "").strip()
    error = str(result.error or "").lower()

    if stage in {{"verify_ocr", "ocr", "ocr_verify"}} or "ocr" in error:
        return "ocr_verify_failed"
    if stage in {{"plan_grasp", "anygrasp", "grasp_planning"}}:
        return "grasp_planning_failed"
    if stage in {{"execute_grasp", "execute", "motion"}}:
        return "grasp_execution_failed"
    if "anygrasp" in error or "grasp plan" in error:
        return "grasp_planning_failed"
    return "grasp_failed"


def write_failure_report(
    args,
    state,
    step,
    failure_type,
    error,
    container_name=None,
    medicine_query=None,
    evidence=None,
):
    node_id = step.get("node_id") if step else None
    retry_count = 1
    if node_id:
        retry_count_by_node = state.setdefault("retry_count_by_node", {{}})
        retry_count = int(retry_count_by_node.get(node_id, 0)) + 1
        retry_count_by_node[node_id] = retry_count

    evidence = evidence or {{}}
    report = {{
        "success": False,
        "goal": GOAL,
        "failed_node_id": node_id,
        "node_label": step.get("label") if step else None,
        "medicine_query": medicine_query,
        "container_name": container_name,
        "failure_type": failure_type,
        "error": str(error or ""),
        "recommendation": evidence.get("recommendation") if isinstance(evidence, dict) else None,
        "evidence": evidence,
        "retry_count": retry_count,
        "completed_node_ids": list(state.get("completed_node_ids", [])),
        "completed_targets": list(state.get("completed_targets", [])),
        "cached_place_plan_names": list((state.get("cached_place_plans") or {{}}).keys()),
        "timestamp": now_iso(),
    }}

    state["failed_node_id"] = node_id
    state["failure_type"] = failure_type
    state["failure_reason"] = str(error or "")
    state["last_failure_report"] = report
    state["updated_at"] = now_iso()

    save_json_file(args.execution_state_path, state)
    save_json_file(args.failure_report_path, report)
    print("\\n=== Failure Report ===")
    print("path:", args.failure_report_path)
    pprint(report)
    return report


def move_xarm(arm, pose, speed, acc) -> bool:
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


def execute_cached_place(context, place_plan, args, gripper_open):
    arm = context.require("xarm")
    approach_pose = place_plan["approach_pose_mmrad"]
    place_pose = place_plan["place_pose_mmrad"]
    lift_pose = place_plan["lift_pose_mmrad"]

    if not move_xarm(arm, approach_pose, args.move_speed, args.move_acc):
        return {{"success": False, "error": "移动到缓存 approach pose 失败。"}}

    if not move_xarm(arm, place_pose, args.move_speed, args.move_acc):
        return {{"success": False, "error": "移动到缓存 place pose 失败。"}}

    if gripper_open and not gripper_open():
        return {{"success": False, "error": "放置时打开夹爪失败。"}}

    if not move_xarm(arm, lift_pose, args.move_speed, args.move_acc):
        return {{"success": False, "error": "放置后抬升失败。"}}

    return {{"success": True, "message": "缓存放置位姿执行完成。"}}


def placement_slot_offset_mm(slot_index, spacing_mm):
    if slot_index <= 0:
        return 0.0
    magnitude = (slot_index + 1) // 2
    direction = -1.0 if slot_index % 2 == 1 else 1.0
    return direction * magnitude * float(spacing_mm)


def offset_place_plan(place_plan, slot_index, args):
    if not args.enable_place_slot_offsets:
        return place_plan
    offset_y = placement_slot_offset_mm(slot_index, args.place_slot_spacing_mm)
    if abs(offset_y) < 1e-6:
        return place_plan

    shifted = dict(place_plan)
    for key in ["approach_pose_mmrad", "place_pose_mmrad", "lift_pose_mmrad"]:
        pose = list(shifted[key])
        pose[1] += offset_y
        shifted[key] = pose
    shifted["place_slot_index"] = int(slot_index)
    shifted["place_slot_offset_y_mm"] = float(offset_y)
    return shifted


def run_medicine_grasp_once(
    grasp_skill,
    args,
    medicine_query,
    gripper_open,
    gripper_close,
):
    return grasp_skill.run(
        query=medicine_query,
        camera_name="wrist",
        min_score=args.grasp_min_score,
        detection_threshold=args.grasp_box_threshold,
        text_threshold=args.grasp_text_threshold,
        use_sam_for_selection=False,
        use_mask_for_grasp=False,
        plan_grasp=True,
        execute=args.execute,
        top_k=args.grasp_top_k,
        grasp_selection_mode=args.grasp_selection_mode,
        random_select_top_k=args.grasp_random_select_top_k,
        bbox_filter_inner_margin_ratio=args.bbox_filter_inner_margin_ratio,
        infer_grasps_on_full_cloud=True,
        filter_grasps_by_bbox=True,
        visualize_selected_grasp=args.visualize_selected_grasp,
        visualize_bbox_filtered_grasps=args.visualize_bbox_filtered_grasps,
        move_speed=args.move_speed,
        move_acc=args.move_acc,
        gripper_open=gripper_open,
        gripper_close=gripper_close,
        gripper_close_wait_s=args.gripper_close_wait_s,
        verify_ocr=not args.no_verify_ocr,
        ocr_verify_strict_mode=args.ocr_verify_strict_mode,
        ocr_verify_fuzzy_threshold=args.ocr_verify_fuzzy_threshold,
        reobserve_on_grasp_failure=not args.no_grasp_reobserve_on_plan_failure,
        reobserve_only_when_execute=not args.grasp_reobserve_in_dry_run,
        reobserve_z_mm=args.grasp_reobserve_z_mm,
        reobserve_wait_s=args.grasp_reobserve_wait_s,
        reobserve_depth_window_px=args.grasp_reobserve_depth_window_px,
        fallback_intrinsics={{
            "fx": 909.59,
            "fy": 909.80,
            "cx": 658.97,
            "cy": 355.42,
        }},
        xarm_ip=args.xarm_ip,
        handeye_config_path=args.handeye_config,
    )


def move_to_previous_target_observe_pose(context, grasp_skill, grasp_result, args):
    data = grasp_result.data or {{}}
    grasp_input = data.get("grasp_input") or {{}}
    selected = data.get("selected_instance") or {{"bbox": grasp_input.get("bbox")}}
    if not selected.get("bbox"):
        return {{"success": False, "error": "缺少上一次目标 bbox，无法移动到目标上方。"}}

    capture = {{
        "depth_path": grasp_input.get("depth_path"),
        "depth_scale": grasp_input.get("depth_scale", 1000.0),
        "depth_intrinsics": (
            grasp_input.get("depth_intrinsics")
            or grasp_input.get("camera_intrinsics")
            or grasp_input.get("color_intrinsics")
        ),
    }}
    try:
        observe_pose = grasp_skill._build_target_observe_pose(
            capture=capture,
            selected=selected,
            fallback_intrinsics={{
                "fx": 909.59,
                "fy": 909.80,
                "cx": 658.97,
                "cy": 355.42,
            }},
            xarm_ip=args.xarm_ip,
            handeye_config_path=args.handeye_config,
            observe_z_mm=args.grasp_vdm_retry_observe_z_mm,
            depth_window_px=args.grasp_reobserve_depth_window_px,
        )
        print("move_vdm_retry_observe_pose:")
        pprint(observe_pose)
        if args.execute:
            arm = context.require("xarm")
            if not move_xarm(arm, observe_pose, args.move_speed, args.move_acc):
                return {{"success": False, "error": "移动到 VDM 失败重试观察位姿失败。"}}
        if args.grasp_vdm_retry_observe_wait_s > 0:
            time.sleep(float(args.grasp_vdm_retry_observe_wait_s))
        return {{"success": True, "observe_pose_mmrad": observe_pose}}
    except Exception as e:
        return {{"success": False, "error": f"构造 VDM 失败重试观察位姿失败: {{e}}"}}


def locate_or_load_container_place_plan(place_skill, args, state, group):
    container_name = group["container_name"]
    container_color = group.get("container_color")
    group_steps = group["steps"]

    print(f"\\n=== Container Observation {{group.get('group_index')}}/{{group.get('group_count')}} ===")
    print("container_name:", container_name)
    print("container_color:", container_color)
    print("reuse_for_steps:", [step["node_id"] for step in group_steps])

    cached_place_plan = load_cached_place_plan(args, state, container_name, container_color)
    if (
        cached_place_plan
        and not args.no_resume
        and not args.reobserve_container_on_resume
    ):
        print("[resume] use cached container place plan:", container_name, container_color)
        return cached_place_plan

    place_result = place_skill.run(
        container_name=container_name,
        container_color=container_color,
        camera_name="wrist",
        detection_threshold=args.container_box_threshold,
        text_threshold=args.container_text_threshold,
        detection_index=args.container_detection_index,
        depth_window_px=args.container_depth_window_px,
        xarm_ip=args.xarm_ip,
        handeye_config_path=args.handeye_config,
        place_clearance_mm=args.place_clearance_mm,
        approach_clearance_mm=args.place_approach_clearance_mm,
        lift_after_place_mm=args.place_lift_after_mm,
        refine_observation=not args.no_container_refine,
        refine_only_when_execute=not args.execute,
        refine_observe_z_mm=args.container_refine_observe_z_mm,
        refine_wait_s=args.container_refine_wait_s,
        execute=False,
        move_speed=args.move_speed,
        move_acc=args.move_acc,
        gripper_open=None,
        fallback_intrinsics={{
            "fx": 909.59,
            "fy": 909.80,
            "cx": 658.97,
            "cy": 355.42,
        }},
    )

    if not place_result.success:
        print("\\n=== Container Observation Failed ===")
        print("container_name:", container_name)
        print("error:", place_result.error)
        write_failure_report(
            args=args,
            state=state,
            step=group_steps[0] if group_steps else None,
            failure_type="container_observation_failed",
            error=place_result.error,
            container_name=container_name,
            medicine_query=None,
            evidence={{"container_color": container_color, "place_result": place_result.data or {{}}}},
        )
        raise RuntimeError(place_result.error)

    place_plan = place_result.data
    save_cached_place_plan(args, state, container_name, container_color, place_plan)
    print("cached_place_pose_mmrad:")
    pprint(place_plan.get("place_pose_mmrad"))
    return place_plan


def return_home_after_step_if_needed(args, arm):
    if not args.execute or args.no_return_home_after_step:
        return {{"success": True, "skipped": True}}
    if args.no_pre_grasp_pose:
        return {{"success": True, "skipped": True}}
    print("return_home_after_step_pose:")
    pprint(args.pre_grasp_pose)
    ok = move_xarm(arm, args.pre_grasp_pose, args.move_speed, args.move_acc)
    if not ok:
        return {{"success": False, "error": "子目标完成后回到 home/pre_grasp 位姿失败。"}}
    return {{"success": True, "pose": args.pre_grasp_pose}}


def run_plan(args):
    context, camera_service = build_context(args)
    arm = None
    state = load_execution_state(args)
    if can_resume_from_state(args):
        save_json_file(args.execution_state_path, state)

    try:
        camera_service.start()
        time.sleep(args.camera_warmup_s)

        if args.execute:
            arm = init_arm(args.xarm_ip)
            context.xarm = arm

        gripper_open = None
        gripper_close = None
        if args.execute and not args.disable_grip:
            gripper_open = lambda: call_grip_service(
                value=args.grip_open_value,
                ros_ws=args.ros_ws,
                service_name=args.grip_service,
                timeout=args.grip_timeout,
                ignore_error=args.ignore_grip_error,
            )
            gripper_close = lambda: call_grip_service(
                value=args.grip_close_value,
                ros_ws=args.ros_ws,
                service_name=args.grip_service,
                timeout=args.grip_timeout,
                ignore_error=args.ignore_grip_error,
            )

        place_skill = PlaceToContainerSkill(context)
        grasp_skill = FindAndGraspMedicineSkill(context)
        results = []
        step_groups = group_steps_by_container(PLAN_STEPS)

        print("\\n=== Generated Plan Executor ===")
        print("goal:", GOAL)
        print("steps:", len(PLAN_STEPS))
        print("container observations:", len(step_groups))

        place_plans_by_group = {{}}
        for group_index, group in enumerate(step_groups, start=1):
            group["group_index"] = group_index
            group["group_count"] = len(step_groups)
            try:
                place_plans_by_group[group["container_key"]] = locate_or_load_container_place_plan(
                    place_skill=place_skill,
                    args=args,
                    state=state,
                    group=group,
                )
            except Exception as e:
                results.append({{
                    "container_name": group["container_name"],
                    "container_color": group.get("container_color"),
                    "success": False,
                    "error": str(e),
                }})
                return False, results

        if args.execute and not args.no_pre_grasp_pose:
            print("move_pre_grasp_pose_after_container_observation:")
            pprint(args.pre_grasp_pose)
            if not move_xarm(arm, args.pre_grasp_pose, args.move_speed, args.move_acc):
                write_failure_report(
                    args=args,
                    state=state,
                    step=PLAN_STEPS[0] if PLAN_STEPS else None,
                    failure_type="pre_grasp_move_failed",
                    error="容器全部定位后移动到抓取前观察位姿失败。",
                    container_name=None,
                    medicine_query=None,
                    evidence={{"pre_grasp_pose": args.pre_grasp_pose}},
                )
                return False, [{{"success": False, "error": "容器全部定位后移动到抓取前观察位姿失败。"}}]

        finished_steps = 0
        for group_index, group in enumerate(step_groups, start=1):
            container_name = group["container_name"]
            container_color = group.get("container_color")
            group_steps = group["steps"]
            place_plan = place_plans_by_group[group["container_key"]]

            for group_step_index, step in enumerate(group_steps):
                finished_steps += 1
                call_args = step["args"]
                medicine_query = call_args["medicine_query"]

                print(f"\\n=== Step {{finished_steps}}/{{len(PLAN_STEPS)}}: {{step['node_id']}} {{step['label']}} ===")
                print("medicine_query:", medicine_query)
                print("container_name:", container_name)
                print("container_color:", container_color)

                if should_skip_step(args, state, step, medicine_query):
                    print("[resume] skip completed step:", step["node_id"], medicine_query)
                    results.append({{
                        "node_id": step["node_id"],
                        "success": True,
                        "skipped": True,
                        "reason": "already_completed",
                    }})
                    continue

                if args.execute and not args.no_pre_grasp_pose:
                    print("move_pre_grasp_pose:")
                    pprint(args.pre_grasp_pose)
                    if not move_xarm(arm, args.pre_grasp_pose, args.move_speed, args.move_acc):
                        write_failure_report(
                            args=args,
                            state=state,
                            step=step,
                            failure_type="pre_grasp_move_failed",
                            error="移动到抓取前观察位姿失败。",
                            container_name=container_name,
                            medicine_query=medicine_query,
                            evidence={{"pre_grasp_pose": args.pre_grasp_pose}},
                        )
                        results.append({{
                            "node_id": step["node_id"],
                            "success": False,
                            "error": "移动到抓取前观察位姿失败。",
                        }})
                        return False, results

                before_global = None
                before_wrist = None
                if args.verify_grasp_vdm:
                    before_global, before_wrist = capture_verification_pair(
                        camera_service,
                        f"runs/current_capture/grasp_verify_{{step['node_id']}}_before",
                    )

                grasp_result = run_medicine_grasp_once(
                    grasp_skill=grasp_skill,
                    args=args,
                    medicine_query=medicine_query,
                    gripper_open=gripper_open,
                    gripper_close=gripper_close,
                )

                if not grasp_result.success:
                    print("\\n=== Step Failed ===")
                    print("node_id:", step["node_id"])
                    print("error:", grasp_result.error)
                    failure_type = infer_grasp_failure_type(grasp_result)
                    write_failure_report(
                        args=args,
                        state=state,
                        step=step,
                        failure_type=failure_type,
                        error=grasp_result.error,
                        container_name=container_name,
                        medicine_query=medicine_query,
                        evidence=grasp_result.data or {{}},
                    )
                    results.append({{
                        "node_id": step["node_id"],
                        "success": False,
                        "error": grasp_result.error,
                    }})
                    return False, results

                grasp_vdm_result = None
                if args.verify_grasp_vdm:
                    grasp_vdm_result = verify_grasp_vdm_if_enabled(
                        context=context,
                        camera_service=camera_service,
                        args=args,
                        medicine_query=medicine_query,
                        before_global=before_global,
                        before_wrist=before_wrist,
                        step_id=step["node_id"],
                    )
                    if (
                        args.grasp_vdm_stop_on_failure
                        and grasp_vdm_result is not None
                        and not grasp_vdm_result.success
                    ):
                        vdm_data = grasp_vdm_result.data or {{}}
                        local_retry_ok = False
                        retryable_vdm_failure = vdm_data.get("failure_type") in [
                            "still_on_table",
                            "held_unclear",
                            "uncertain",
                        ]
                        if (
                            args.execute
                            and args.grasp_vdm_local_retry_attempts > 0
                            and retryable_vdm_failure
                        ):
                            for retry_index in range(int(args.grasp_vdm_local_retry_attempts)):
                                print(
                                    f"\\n=== Local Grasp Retry {{retry_index + 1}}/"
                                    f"{{args.grasp_vdm_local_retry_attempts}} ==="
                                )
                                move_retry = move_to_previous_target_observe_pose(
                                    context=context,
                                    grasp_skill=grasp_skill,
                                    grasp_result=grasp_result,
                                    args=args,
                                )
                                if not move_retry.get("success", False):
                                    print("local_retry_error:", move_retry.get("error"))
                                    break

                                before_global, before_wrist = capture_verification_pair(
                                    camera_service,
                                    f"runs/current_capture/grasp_verify_{{step['node_id']}}_retry{{retry_index + 1}}_before",
                                )
                                retry_grasp_result = run_medicine_grasp_once(
                                    grasp_skill=grasp_skill,
                                    args=args,
                                    medicine_query=medicine_query,
                                    gripper_open=gripper_open,
                                    gripper_close=gripper_close,
                                )
                                if not retry_grasp_result.success:
                                    print("local_retry_error:", retry_grasp_result.error)
                                    grasp_result = retry_grasp_result
                                    continue

                                retry_vdm_result = verify_grasp_vdm_if_enabled(
                                    context=context,
                                    camera_service=camera_service,
                                    args=args,
                                    medicine_query=medicine_query,
                                    before_global=before_global,
                                    before_wrist=before_wrist,
                                    step_id=f"{{step['node_id']}}_retry{{retry_index + 1}}",
                                )
                                grasp_result = retry_grasp_result
                                grasp_vdm_result = retry_vdm_result
                                if retry_vdm_result is not None and retry_vdm_result.success:
                                    print("local_retry_result: success")
                                    local_retry_ok = True
                                    break
                                print(
                                    "local_retry_result:",
                                    (retry_vdm_result.data if retry_vdm_result else None),
                                )

                        if local_retry_ok:
                            pass
                        else:
                            vdm_data = grasp_vdm_result.data or {{}}
                            if args.grasp_vdm_failure_policy == "place_then_verify":
                                print(
                                    "grasp_vdm_policy: place_then_verify, "
                                    "continue to cached place despite grasp VDM failure."
                                )
                            elif args.grasp_vdm_failure_policy == "warn_only":
                                print("grasp_vdm_policy: warn_only, continue to cached place.")
                            else:
                                write_failure_report(
                                    args=args,
                                    state=state,
                                    step=step,
                                    failure_type=vdm_data.get("failure_type") or "grasp_vdm_verification_failed",
                                    error=grasp_vdm_result.error or "grasp_vdm_verification_failed",
                                    container_name=container_name,
                                    medicine_query=medicine_query,
                                    evidence=vdm_data,
                                )
                                results.append({{
                                    "node_id": step["node_id"],
                                    "success": False,
                                    "error": "grasp_vdm_verification_failed",
                                    "grasp_vdm": grasp_vdm_result.data,
                                }})
                                return False, results
                place_execution = None
                if args.execute:
                    place_verify_before_global = None
                    place_verify_before_wrist = None
                    if args.verify_place_vdm:
                        place_verify_before_global, place_verify_before_wrist = capture_verification_pair(
                            camera_service,
                            f"runs/current_capture/place_verify_{{step['node_id']}}_before",
                        )
                    step_place_plan = offset_place_plan(
                        place_plan=place_plan,
                        slot_index=group_step_index,
                        args=args,
                    )
                    if step_place_plan is not place_plan:
                        print(
                            "place_slot_offset_y_mm:",
                            step_place_plan.get("place_slot_offset_y_mm"),
                        )
                    place_execution = execute_cached_place(
                        context=context,
                        place_plan=step_place_plan,
                        args=args,
                        gripper_open=gripper_open,
                    )
                    if not place_execution.get("success", False):
                        print("\\n=== Cached Place Failed ===")
                        print("node_id:", step["node_id"])
                        print("error:", place_execution.get("error"))
                        write_failure_report(
                            args=args,
                            state=state,
                            step=step,
                            failure_type="cached_place_failed",
                            error=place_execution.get("error"),
                           container_name=container_name,
                            medicine_query=medicine_query,
                            evidence=place_execution,
                        )
                        results.append({{
                            "node_id": step["node_id"],
                            "success": False,
                            "error": place_execution.get("error"),
                        }})
                        return False, results

                    place_vdm_result = verify_place_vdm_if_enabled(
                        context=context,
                        camera_service=camera_service,
                        args=args,
                        medicine_query=medicine_query,
                        container_name=container_name,
                        container_color=container_color,
                        before_global=place_verify_before_global,
                        before_wrist=place_verify_before_wrist,
                        step_id=step["node_id"],
                    )
                    if (
                        args.place_vdm_stop_on_failure
                        and place_vdm_result is not None
                        and not place_vdm_result.success
                    ):
                        place_vdm_data = place_vdm_result.data or {{}}
                        raw_place_failure_type = (
                            place_vdm_data.get("failure_type") or "place_vdm_verification_failed"
                        )
                        place_confidence = float(place_vdm_data.get("confidence") or 0.0)
                        target_visible_elsewhere = bool(place_vdm_data.get("target_visible_elsewhere"))
                        target_visible_in_container = bool(place_vdm_data.get("target_visible_in_container"))
                        soft_negative = (
                            raw_place_failure_type in ["not_in_container", "uncertain", "container_occluded"]
                            and not target_visible_elsewhere
                            and place_confidence < args.place_vdm_min_failure_confidence
                        )
                        if target_visible_in_container or soft_negative:
                            print(
                                "place_vdm_policy: weak_negative_continue, "
                                f"failure_type={{raw_place_failure_type}}, confidence={{place_confidence}}"
                            )
                            place_vdm_result = None
                            place_vdm_data = {{}}
                        else:
                            if raw_place_failure_type in [
                                "not_in_container",
                                "wrong_object",
                                "uncertain",
                            ]:
                                failure_type = "pick_place_verification_failed"
                                error = (
                                    f"放置后未在容器中确认目标，下一轮应重新抓取该药盒。"
                                    f" raw_failure_type={{raw_place_failure_type}}"
                                )
                            else:
                                failure_type = raw_place_failure_type
                                error = place_vdm_result.error or raw_place_failure_type
                            write_failure_report(
                                args=args,
                                state=state,
                                step=step,
                                failure_type=failure_type,
                                error=error,
                                container_name=container_name,
                                medicine_query=medicine_query,
                                evidence=place_vdm_data,
                            )
                            results.append({{
                                "node_id": step["node_id"],
                                "success": False,
                                "error": "place_vdm_verification_failed",
                                "place_vdm": place_vdm_data,
                            }})
                            return False, results

                grasp_plan = (grasp_result.data or {{}}).get("grasp_plan", {{}})

                print("\\n=== Step Result ===")
                print("success:", grasp_result.success)
                print("grasp_pose_mmrad:")
                pprint(grasp_plan.get("best_grasp_xarm_mmrad"))
                print("grasp_choose_reason:", grasp_plan.get("choose_reason"))
                print("grasp_vdm_verification:", grasp_vdm_result.data if grasp_vdm_result else None)
                print("place_execution:", place_execution)
                mark_step_completed(args, state, step, medicine_query)
                results.append({{
                    "node_id": step["node_id"],
                    "success": True,
                    "container_reused": True,
                    "grasp_vdm": grasp_vdm_result.data if grasp_vdm_result else None,
                }})
                home_result = return_home_after_step_if_needed(args, arm)
                if not home_result.get("success", False):
                    write_failure_report(
                        args=args,
                        state=state,
                        step=step,
                        failure_type="return_home_failed",
                        error=home_result.get("error"),
                        container_name=container_name,
                        medicine_query=medicine_query,
                        evidence=home_result,
                    )
                    results.append({{
                        "node_id": step["node_id"],
                        "success": False,
                        "error": home_result.get("error"),
                    }})
                    return False, results

        return True, results

    finally:
        if args.stop_camera_after_task:
            camera_service.stop()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Generated executor for PlanGraph.")
    parser.add_argument("--execute", action="store_true")

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

    parser.add_argument("--global_serial", default="213522251050")
    parser.add_argument("--wrist_serial", default="243222070053")
    parser.add_argument("--capture_dir", default="runs/current_capture")
    parser.add_argument("--camera_warmup_s", type=float, default=2.0)
    parser.add_argument("--stop_camera_after_task", action="store_true")

    parser.add_argument("--xarm_ip", default="192.168.1.237")
    parser.add_argument(
        "--handeye_config",
        default="/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
    )
    parser.add_argument(
        "--pre_grasp_pose",
        type=float,
        nargs=6,
        default=[350, 0, 450, 3.14, 0.0, -1.57],
        metavar=("x", "y", "z", "rx", "ry", "rz"),
    )
    parser.add_argument("--no_pre_grasp_pose", action="store_true")
    parser.add_argument("--no_return_home_after_step", action="store_true")
    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)

    parser.add_argument("--container_box_threshold", type=float, default=0.25)
    parser.add_argument("--container_text_threshold", type=float, default=0.20)
    parser.add_argument("--container_detection_index", type=int, default=0)
    parser.add_argument("--container_depth_window_px", type=int, default=25)
    parser.add_argument("--no_container_refine", action="store_true")
    parser.add_argument("--reobserve_container_on_resume", action="store_true")
    parser.add_argument("--container_refine_observe_z_mm", type=float, default=400.0)
    parser.add_argument("--container_refine_wait_s", type=float, default=0.8)
    parser.add_argument("--place_clearance_mm", type=float, default=250.0)
    parser.add_argument("--place_approach_clearance_mm", type=float, default=120.0)
    parser.add_argument("--place_lift_after_mm", type=float, default=120.0)

    parser.add_argument("--grasp_box_threshold", type=float, default=0.22)
    parser.add_argument("--grasp_text_threshold", type=float, default=0.18)
    parser.add_argument("--grasp_min_score", type=float, default=0.50)
    parser.add_argument("--grasp_top_k", type=int, default=50)
    parser.add_argument(
        "--grasp_selection_mode",
        default="random_top_k_good",
        choices=["best_good", "random_good", "random_top_k_good"],
    )
    parser.add_argument("--grasp_random_select_top_k", type=int, default=5)
    parser.add_argument("--bbox_filter_inner_margin_ratio", type=float, default=0.15)
    parser.add_argument("--visualize_selected_grasp", action="store_true")
    parser.add_argument("--visualize_bbox_filtered_grasps", action="store_true")
    parser.add_argument("--no_verify_ocr", action="store_true")
    parser.add_argument("--ocr_verify_strict_mode", action="store_true")
    parser.add_argument("--ocr_verify_fuzzy_threshold", type=float, default=0.72)
    parser.add_argument("--no_grasp_reobserve_on_plan_failure", action="store_true")
    parser.add_argument("--grasp_reobserve_in_dry_run", action="store_true")
    parser.add_argument("--grasp_reobserve_z_mm", type=float, default=400.0)
    parser.add_argument("--grasp_reobserve_wait_s", type=float, default=0.8)
    parser.add_argument("--grasp_reobserve_depth_window_px", type=int, default=25)
    parser.add_argument("--verify_grasp_vdm", action="store_true")
    parser.add_argument("--grasp_vdm_stop_on_failure", action="store_true")
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
    )
    parser.add_argument("--grasp_vdm_local_retry_attempts", type=int, default=1)
    parser.add_argument("--grasp_vdm_retry_observe_z_mm", type=float, default=400.0)
    parser.add_argument("--grasp_vdm_retry_observe_wait_s", type=float, default=0.8)
    parser.add_argument("--grasp_vdm_contact_sheet", action="store_true")
    parser.add_argument("--grasp_vdm_no_contact_sheet", action="store_true")
    parser.add_argument("--verify_place_vdm", action="store_true")
    parser.add_argument("--place_vdm_stop_on_failure", action="store_true")
    parser.add_argument("--place_vdm_min_success_confidence", type=float, default=0.55)
    parser.add_argument("--place_vdm_min_visible_success_confidence", type=float, default=0.35)
    parser.add_argument("--place_vdm_min_failure_confidence", type=float, default=0.75)
    parser.add_argument("--enable_place_slot_offsets", action="store_true", default=True)
    parser.add_argument("--disable_place_slot_offsets", dest="enable_place_slot_offsets", action="store_false")
    parser.add_argument("--place_slot_spacing_mm", type=float, default=35.0)

    parser.add_argument("--disable_grip", action="store_true")
    parser.add_argument("--ros_ws", default="/home/wpy/gt200_ws")
    parser.add_argument("--grip_service", default="/grip")
    parser.add_argument("--grip_timeout", type=float, default=5.0)
    parser.add_argument("--grip_open_value", type=int, default=1)
    parser.add_argument("--grip_close_value", type=int, default=0)
    parser.add_argument("--gripper_close_wait_s", type=float, default=1.5)
    parser.add_argument("--ignore_grip_error", action="store_true", default=True)
    parser.add_argument("--execution_state_path", default="runs/current_plan/execution_state.json")
    parser.add_argument("--failure_report_path", default="runs/current_plan/failure_report.json")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--reset_execution_state", action="store_true")
    parser.add_argument("--track_dry_run_state", action="store_true")
    parser.add_argument("--verbose_events", action="store_true")
    args = parser.parse_args()

    success, results = run_plan(args)
    print("\\n=== Plan Execution Summary ===")
    pprint(results)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''
