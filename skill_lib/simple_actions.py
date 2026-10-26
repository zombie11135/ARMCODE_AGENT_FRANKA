"""Public observe/grasp/place facade and owner of per-task execution state.

Implementations live in skill_lib.actions by responsibility. Delegation keeps
existing signatures, mutable caches and method overrides available to callers.
Action values and configuration retain their historical import paths here.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from core.actions import (
    SimpleAction as SimpleAction,
    grasp as grasp,
    observe as observe,
    place as place,
)
from core.skill_base import SkillResult
from skill_lib.config import (
    FALLBACK_INTRINSICS as FALLBACK_INTRINSICS,
    SimpleRobotConfig as SimpleRobotConfig,
)
from skill_lib.actions import (
    gate,
    gate_logging,
    grasp as grasp_actions,
    motion,
    observation,
    occlusion,
    placement,
    targets,
)
# Retain helper imports used by existing scripts. New code can import targets directly.
from skill_lib.actions.targets import (
    container_key as container_key,
    parse_color as parse_color,
    parse_container_name as parse_container_name,
    parse_container_spec as parse_container_spec,
    placement_slot_offset_mm as placement_slot_offset_mm,
)
from skill_lib.actions.occlusion import _as_dict as _as_dict, _existing_path as _existing_path


class SimpleRobotActions:
    """Execute concise actions using one shared set of task caches.

    The facade owns observations, placement plans, probe/recovery counters and
    Gate instances. Modules in actions/ use that state directly, so memory
    restoration and caller overrides remain visible to every operation.
    """

    def __init__(
        self,
        context,
        config: Optional[SimpleRobotConfig] = None,
        gripper_open: Optional[Callable[[], bool]] = None,
        gripper_close: Optional[Callable[[], bool]] = None,
    ):
        self.context = context
        self.config = config or SimpleRobotConfig()
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.place_plans: Dict[str, Dict[str, Any]] = {}
        self.place_counts: Dict[str, int] = {}
        self.observations: Dict[str, Dict[str, Any]] = {}
        self._pipeline_frame_bindings = set()
        self._container_task_id = uuid4().hex
        self._task_container_plans = {}
        self._task_run_started = False
        from runtime.observation_memory import memory_for
        memory = memory_for(getattr(context, 'camera', None))
        if memory is not None:
            memory.on_invalidate(self._discard_scene_evidence)
        self.gate_probe_counts: Dict[str, int] = {}
        self.occlusion_recovery_counts: Dict[str, int] = {}
        self.geometry_buffer_slots_used = 0
        self.geometry_occlusion_events = []
        self.geometry_recovery_state = {'observation_counts': {}, 'history': []}
        # Phase-1 unified gate cache (feature reuse: one facade per config key).
        self._unified_gate = None
        self._unified_gate_key: Optional[Tuple[str, str, float, int]] = None
        self._gate_logger = None
        self._last_gate_decision_id: Optional[str] = None

    def _discard_scene_evidence(self, reason):
        self._pipeline_frame_bindings.clear()
        from skill_lib.actions.task_containers import discard_scene_evidence
        discard_scene_evidence(self, reason)

    def _is_task_container_plan(self, plan):
        from skill_lib.actions.task_containers import is_task_container_plan
        return is_task_container_plan(self, plan)

    def run(self, actions: Iterable[SimpleAction]) -> SkillResult:
        actions = list(actions)
        if self._task_run_started:
            from runtime.observation_memory import invalidate_camera, memory_for
            camera = getattr(self.context, 'camera', None)
            if memory_for(camera) is not None:
                invalidate_camera(camera, 'new_task')
            else:
                self._discard_scene_evidence('new_task')
        self._task_run_started = True
        try:
            preobserve_result = self.preobserve_containers(actions)
            if not preobserve_result.success:
                return preobserve_result
            self.home()

            step_results = []
            current_holding: Optional[str] = None
            last_gated_result: Optional[float] = None
            for index, action in enumerate(actions):
                if action.op in {"observe", "grasp", "place"}:
                    gate_result = self.gate_before_action(
                        action=action,
                        current_holding=current_holding,
                        target_plate=self._next_place_target(actions, index),
                        previous_action_result=last_gated_result,
                        step_index=index,
                        total_steps=len(actions),
                    )
                    if not gate_result.success:
                        step_results.append(
                            {
                                "action": action.__dict__,
                                "result": gate_result.data,
                                "success": False,
                            }
                        )
                        self.log_gate_outcome(
                            primitive_success=False,
                            skill_success=False,
                            execution_failure_type=(gate_result.data or {}).get("failure_type"),
                            task_progress_before=len(step_results) / len(actions),
                            task_progress_after=len(step_results) / len(actions),
                            trace_reference=str(self.config.nesyro_gate_trace_path or ""),
                        )
                        return SkillResult(
                            success=False,
                            data={"steps": step_results, "failed_action": action.__dict__},
                            error=gate_result.error,
                        )

                if action.op == "observe":
                    result = self.observe(action.target)
                elif action.op == "grasp":
                    result = self.grasp(action.target)
                    if result.success:
                        current_holding = action.target
                elif action.op == "place":
                    result = self.place(action.target)
                    if result.success:
                        current_holding = None
                        self.home()
                else:
                    return SkillResult(
                        success=False,
                        data={"action": action},
                        error=f"Unknown simple action: {action.op}",
                    )

                if action.op in {"observe", "grasp", "place"}:
                    self.log_gate_outcome(
                        primitive_success=result.success,
                        skill_success=result.success,
                        execution_failure_type=(result.data or {}).get("failure_type"),
                        task_progress_before=len(step_results) / len(actions),
                        task_progress_after=(len(step_results) + 1) / len(actions) if result.success else len(step_results) / len(actions),
                        trace_reference=str(self.config.nesyro_gate_trace_path or ""),
                    )
                    last_gated_result = 1.0 if result.success else -1.0

                step_results.append({"action": action.__dict__, "result": result.data, "success": result.success})
                if not result.success:
                    return SkillResult(
                        success=False,
                        data={"steps": step_results, "failed_action": action.__dict__},
                        error=result.error,
                    )

            return SkillResult(success=True, data={"steps": step_results})
        except Exception as e:
            return SkillResult(success=False, data={}, error=f"simple action execution failed: {e}")

    @staticmethod
    def _next_place_target(actions: List[SimpleAction], current_index: int) -> Optional[str]:
        for action in actions[current_index + 1:]:
            if action.op == "place":
                return action.target
            if action.op == "grasp":
                return None
        return None

    # Gate decisions and probes.

    def gate_before_action(
        self,
        action: SimpleAction,
        current_holding: Optional[str] = None,
        target_plate: Optional[str] = None,
        previous_action_result: Optional[float] = None,
        step_index: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> SkillResult:
        return gate.gate_before_action(
            self,
            action=action,
            current_holding=current_holding,
            target_plate=target_plate,
            previous_action_result=previous_action_result,
            step_index=step_index,
            total_steps=total_steps,
        )

    def build_scene_memory(self, current_holding: Optional[str] = None,
                           *, geometry_prepared_target: Optional[str] = None):
        return gate.build_scene_memory(self, current_holding=current_holding,
                                      geometry_prepared_target=geometry_prepared_target)

    def _action_to_gate_step(
        self,
        action: SimpleAction,
        current_holding: Optional[str],
    ):
        return gate._action_to_gate_step(self, action=action, current_holding=current_holding)

    def _action_to_gate_task_spec(
        self,
        action: SimpleAction,
        current_holding: Optional[str],
        target_plate: Optional[str],
    ):
        return gate._action_to_gate_task_spec(
            self,
            action=action,
            current_holding=current_holding,
            target_plate=target_plate,
        )

    def _get_unified_gate(self):
        return gate._get_unified_gate(self)

    def _build_gate_input(
        self,
        action: SimpleAction,
        current_holding: Optional[str],
        scene_memory,
        previous_action_result: Optional[float],
        step_index: Optional[int],
        total_steps: Optional[int],
    ):
        return gate._build_gate_input(
            self,
            action=action,
            current_holding=current_holding,
            scene_memory=scene_memory,
            previous_action_result=previous_action_result,
            step_index=step_index,
            total_steps=total_steps,
        )

    def _run_gate_probe(
        self,
        action: SimpleAction,
        current_holding: Optional[str],
        suggested_probe: Optional[str],
        reason: str,
    ) -> SkillResult:
        return gate._run_gate_probe(
            self,
            action=action,
            current_holding=current_holding,
            suggested_probe=suggested_probe,
            reason=reason,
        )

    def _has_cached_observation(self, target: str) -> bool:
        return gate._has_cached_observation(self, target=target)

    @staticmethod
    def _gate_failure_type(decision: str) -> str:
        return gate._gate_failure_type(decision=decision)

    # Occlusion evidence and recovery.

    def _should_run_occlusion_recovery(
        self,
        action: SimpleAction,
        failed_conditions: List[str],
        require_occlusion: bool = True,
    ) -> bool:
        return occlusion._should_run_occlusion_recovery(
            self,
            action=action,
            failed_conditions=failed_conditions,
            require_occlusion=require_occlusion,
        )

    def _run_occlusion_recovery(self, target: str) -> SkillResult:
        return occlusion._run_occlusion_recovery(self, target=target)

    def _invoke_clear_occlusion_skill(self, target: str) -> SkillResult:
        return occlusion._invoke_clear_occlusion_skill(self, target=target)

    def _should_try_active_occlusion_view(self, result: SkillResult) -> bool:
        return occlusion._should_try_active_occlusion_view(self, result=result)

    @staticmethod
    def _recovery_attempt_record(view: Any, result: SkillResult) -> Dict[str, Any]:
        return occlusion._recovery_attempt_record(view=view, result=result)

    def _refresh_after_occlusion_recovery(self, target: str) -> SkillResult:
        return occlusion._refresh_after_occlusion_recovery(self, target=target)

    def _run_vlm_occlusion_gate_if_enabled(self, target: str) -> SkillResult:
        return occlusion._run_vlm_occlusion_gate_if_enabled(self, target=target)

    # Gate traces and experience logging.

    def _get_gate_logger(self):
        return gate_logging._get_gate_logger(self)

    def _log_gate_decision(self, unified_result, action: SimpleAction) -> Optional[str]:
        return gate_logging._log_gate_decision(self, unified_result=unified_result, action=action)

    def log_gate_outcome(
        self,
        *,
        primitive_success: Optional[bool] = None,
        skill_success: Optional[bool] = None,
        execution_failure_type: Optional[str] = None,
        task_progress_before: Optional[float] = None,
        task_progress_after: Optional[float] = None,
        visual_change: Optional[Dict[str, Any]] = None,
        trace_reference: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> bool:
        return gate_logging.log_gate_outcome(
            self,
            primitive_success=primitive_success,
            skill_success=skill_success,
            execution_failure_type=execution_failure_type,
            task_progress_before=task_progress_before,
            task_progress_after=task_progress_after,
            visual_change=visual_change,
            trace_reference=trace_reference,
            decision_id=decision_id,
        )

    def _gate_record(
        self,
        action: SimpleAction,
        current_holding: Optional[str],
        gate_result,
        scene_memory,
    ) -> Dict[str, Any]:
        return gate_logging._gate_record(
            self,
            action=action,
            current_holding=current_holding,
            gate_result=gate_result,
            scene_memory=scene_memory,
        )

    def _append_gate_trace(self, record: Dict[str, Any]) -> None:
        return gate_logging._append_gate_trace(self, record=record)

    @staticmethod
    def _safe_json_for_gate(value):
        return gate_logging._safe_json_for_gate(value=value)

    # Observation and cache management.

    def preobserve_containers(self, actions: Iterable[SimpleAction]) -> SkillResult:
        return observation.preobserve_containers(self, actions=actions)

    def observe_container(self, container: str) -> SkillResult:
        return observation.observe_container(self, container=container)

    def observe(self, obj: str) -> SkillResult:
        return observation.observe(self, obj=obj)

    def _observe_container_action(self, container: str) -> SkillResult:
        return observation._observe_container_action(self, container=container)

    def _observe_medicine_action(self, obj: str) -> SkillResult:
        return observation._observe_medicine_action(self, obj=obj)

    def _observe_medicine_once(self, obj: str) -> SkillResult:
        return observation._observe_medicine_once(self, obj=obj)

    def _scan_until_found(
        self,
        target: str,
        observe_fn: Callable[[str], SkillResult],
    ) -> SkillResult:
        return observation._scan_until_found(self, target=target, observe_fn=observe_fn)

    def _invalidate_container_cache(self, container: str) -> None:
        return observation._invalidate_container_cache(self, container=container)

    # Grasping.

    def grasp(self, obj: str) -> SkillResult:
        return grasp_actions.grasp(self, obj=obj)

    # Placement.

    def place(self, container: str) -> SkillResult:
        return placement.place(self, container=container)

    def _execute_cached_place(self, place_plan: Dict[str, Any]) -> Dict[str, Any]:
        return placement._execute_cached_place(self, place_plan=place_plan)

    def _offset_place_plan(self, place_plan: Dict[str, Any], slot_index: int) -> Dict[str, Any]:
        return placement._offset_place_plan(self, place_plan=place_plan, slot_index=slot_index)

    # Motion through the hardware adapter.

    def home(self) -> SkillResult:
        return motion.home(self)

    def _move_to_scan_pose(self, offset_y_mm: float) -> SkillResult:
        return motion._move_to_scan_pose(self, offset_y_mm=offset_y_mm)

    def _move_to_cached_container_observe_pose(
        self,
        container: str,
        place_plan: Dict[str, Any],
    ) -> SkillResult:
        return motion._move_to_cached_container_observe_pose(
            self,
            container=container,
            place_plan=place_plan,
        )

    def _move_to_cached_medicine_observe_pose(
        self,
        obj: str,
        cached: Dict[str, Any],
    ) -> SkillResult:
        return motion._move_to_cached_medicine_observe_pose(self, obj=obj, cached=cached)

    def _move_xarm(self, pose: List[float]) -> bool:
        return motion._move_xarm(self, pose=pose)

    # Target classification.

    @staticmethod
    def _looks_like_container(text: str) -> bool:
        return targets._looks_like_container(text=text)
