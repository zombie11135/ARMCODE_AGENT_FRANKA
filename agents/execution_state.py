import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class ExecutionState:
    goal: str = ""
    plan_json: str = ""
    completed_node_ids: List[str] = field(default_factory=list)
    completed_targets: List[str] = field(default_factory=list)
    cached_place_plans: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    retry_count_by_node: Dict[str, int] = field(default_factory=dict)
    failed_node_id: Optional[str] = None
    failure_type: Optional[str] = None
    failure_reason: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    last_failure_report: Dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=now_iso)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        clean = {field_name: data.get(field_name) for field_name in cls.__dataclass_fields__}
        clean["completed_node_ids"] = list(clean.get("completed_node_ids") or [])
        clean["completed_targets"] = list(clean.get("completed_targets") or [])
        clean["cached_place_plans"] = dict(clean.get("cached_place_plans") or {})
        clean["retry_count_by_node"] = dict(clean.get("retry_count_by_node") or {})
        clean["constraints"] = list(clean.get("constraints") or [])
        clean["last_failure_report"] = dict(clean.get("last_failure_report") or {})
        clean["updated_at"] = clean.get("updated_at") or now_iso()
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str, default_goal: str = "", default_plan_json: str = ""):
        state_path = Path(path)
        if not state_path.exists():
            return cls(goal=default_goal, plan_json=default_plan_json)
        with open(state_path, "r", encoding="utf-8") as f:
            state = cls.from_dict(json.load(f))
        state.goal = state.goal or default_goal
        state.plan_json = state.plan_json or default_plan_json
        return state

    def save(self, path: str) -> str:
        self.updated_at = now_iso()
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return str(state_path)

    def mark_completed(self, node_id: str, target: str = ""):
        if node_id and node_id not in self.completed_node_ids:
            self.completed_node_ids.append(node_id)
        if target and target not in self.completed_targets:
            self.completed_targets.append(target)
        self.failed_node_id = None
        self.failure_type = None
        self.failure_reason = None

    def mark_failed(
        self,
        node_id: str,
        failure_type: str,
        reason: str,
        report: Optional[Dict[str, Any]] = None,
    ) -> int:
        self.failed_node_id = node_id
        self.failure_type = failure_type
        self.failure_reason = reason
        self.retry_count_by_node[node_id] = int(self.retry_count_by_node.get(node_id, 0)) + 1
        self.last_failure_report = dict(report or {})
        return self.retry_count_by_node[node_id]


def classify_failure_constraints(failure_report: Dict[str, Any]) -> List[str]:
    failure_type = str(failure_report.get("failure_type") or "").strip()
    node_id = str(failure_report.get("failed_node_id") or failure_report.get("node_id") or "").strip()
    medicine_query = str(failure_report.get("medicine_query") or "").strip()
    error = str(failure_report.get("error") or failure_report.get("reason") or "").strip()
    evidence = failure_report.get("evidence") if isinstance(failure_report.get("evidence"), dict) else {}
    gate = evidence.get("gate") if isinstance(evidence.get("gate"), dict) else {}
    suggested_constraint = (
        failure_report.get("suggested_constraint")
        or evidence.get("suggested_constraint")
        or gate.get("suggested_constraint")
    )
    failed_conditions = (
        failure_report.get("failed_conditions")
        or evidence.get("failed_conditions")
        or gate.get("failed_conditions")
        or []
    )
    constraints: List[str] = []

    prefix = f"失败节点 {node_id}: " if node_id else ""
    target_text = f"目标药品 {medicine_query}: " if medicine_query else ""

    if failure_type in {"nesyro_gate_replan", "nesyro_gate_blocked", "nesyro_gate_abort"}:
        if suggested_constraint:
            constraints.append(f"{prefix}{target_text}NeSyRo Gate 约束：{suggested_constraint}")
        if failed_conditions:
            constraints.append(f"{prefix}{target_text}Gate 失败条件：{', '.join(str(x) for x in failed_conditions)}")
        if "target_occluded" in failed_conditions or "target_occluded" in str(suggested_constraint):
            constraints.extend(
                [
                    f"{prefix}{target_text}目标当前被遮挡或被其他物体压住，禁止直接抓取该目标。",
                    "重新规划时应先主动观察换视角，或规划移除/避开遮挡物，再重新评估目标是否可抓。",
                    "即使 OCR 文字可见，也不能把该目标视为可抓取。",
                ]
            )
        elif "target_not_visible" in failed_conditions:
            constraints.append(f"{prefix}{target_text}目标不可见，重新规划时必须先执行主动观察或重新定位。")
        elif "target_not_graspable" in failed_conditions:
            constraints.append(f"{prefix}{target_text}目标当前不可抓，重新规划时应改变观察/抓取策略。")
        else:
            constraints.append(f"{prefix}{target_text}NeSyRo Gate 阻止执行，需要从失败节点开始恢复。")
    elif failure_type in {"verify_ocr", "ocr_verify_failed", "ocr_text_missing", "ocr_wrong_target"}:
        constraints.extend(
            [
                f"{prefix}{target_text}OCR 未可靠确认目标文字，重新规划时必须先近距离观察或要求更明确的药名。",
                "禁止用颜色、形状、方位描述作为 medicine_query。",
                "若仍无法确认药名，应暂停该目标，不直接执行抓取。",
            ]
        )
    elif failure_type in {
        "grasp_failed",
        "grasp_planning_failed",
        "grasp_execution_failed",
        "grasp_vdm_verification_failed",
        "pick_place_verification_failed",
        "not_in_container",
        "wrong_object",
        "still_on_table",
        "dropped",
        "held_unclear",
    }:
        constraints.extend(
            [
                f"{prefix}{target_text}抓取阶段失败，重新执行时从该目标重新抓取，不需要重新执行已完成目标。",
                "复用空手阶段缓存的容器位姿，避免抓取后再次观察容器。",
                "重新抓取时优先调整抓取候选筛选参数，例如增大候选数量、改变 bbox 内边距或更换得分前若干候选。",
            ]
        )
    elif failure_type in {"place_failed", "cached_place_failed", "container_observation_failed"}:
        constraints.extend(
            [
                f"{prefix}{target_text}放置或容器定位失败，重新规划时优先修复容器定位/放置子目标。",
                "如果已有 cached_place_plan 且容器未移动，应优先复用缓存位姿。",
            ]
        )
    else:
        constraints.extend(
            [
                f"{prefix}{target_text}执行失败，需要从失败节点开始恢复。",
                f"失败信息：{error}" if error else "保留已完成节点，不要重复执行已经成功的目标。",
            ]
        )

    recommendation = failure_report.get("recommendation")
    if recommendation:
        constraints.append(f"判别器建议：{recommendation}")

    return [constraint for constraint in constraints if constraint]
