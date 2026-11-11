"""Shared task progress and recovery decisions; no device access."""
from dataclasses import dataclass, field
from pathlib import Path
from services.object_reference_service import atomic_json


REFRESH_FAILURES = {'grasp_pose_not_found', 'observation_failed', 'geometry_occlusion_missing',
                    'geometry_occlusion_unresolved', 'target_not_found'}
GRAPH_FAILURES = {'nesyro_gate_replan', 'nesyro_gate_probe_exhausted', 'nesyro_gate_blocked'}


def recovery_decision(failure_type, progress, action):
    if progress.get('released') or progress.get('holding_state') in ('unknown', 'possibly_holding'):
        return dict(route='stop', reason='inspect_holding_or_release_state')
    if progress.get('motion_started') or action == 'place':
        return dict(route='stop', reason='physical_action_may_have_changed_scene')
    if failure_type == 'geometry_information_plan_ready':
        return dict(route='stop', reason='information_action_planned_not_executed')
    if failure_type in REFRESH_FAILURES:
        return dict(route='refresh', reason='new_observation_before_replanning')
    if failure_type in GRAPH_FAILURES:
        return dict(route='graph_repair', reason='update_failed_node_constraints')
    if failure_type == 'policy_failed':
        return dict(route='code_repair', reason='repair_current_subtask_program')
    return dict(route='stop', reason='no_automatic_recovery_for_this_failure')


@dataclass
class TaskRunState:
    goal: str
    path: Path
    completed_actions: list = field(default_factory=list)
    nodes: dict = field(default_factory=dict)
    budgets: dict = field(default_factory=dict)
    skill_budgets: dict = field(default_factory=dict)
    scene: dict = field(default_factory=dict)
    revision: int = 0
    holding_state: str = 'unknown'
    active_action: dict | None = None
    last_feedback: dict | None = None

    def consume(self, key, limit):
        used = self.budgets.get(key, 0)
        if used >= limit:
            return False
        self.budgets[key] = used + 1
        self.save()
        return True

    def begin(self, action, completed):
        self.active_action = dict(action.__dict__)
        self.completed_actions = list(completed)
        self.save()

    def complete(self, action, completed):
        self.completed_actions = list(completed)
        if action.op == 'grasp':
            self.holding_state = 'possibly_holding'
        elif action.op == 'place':
            self.holding_state = 'empty'
        self.save()

    def fail(self, report):
        progress = report['evidence'].get('manipulation_progress', {})
        self.holding_state = progress.get('holding_state', self.holding_state)
        self.last_feedback = dict(failure_type=report['failure_type'], error=report['error'],
            action=report['failed_action'], progress=progress, recovery=report['recovery'])
        self.completed_actions = list(report['completed_actions'])
        for node in self.nodes.values():
            if node['status'] == 'running':
                node['status'] = ('awaiting_information' if report['evidence'].get('recovery_pending') else
                                  'paused' if report['recovery']['route'] == 'stop' else 'failed')
        self.save()

    def save(self):
        atomic_json(self.path, dict(schema='task_run_state_v1', goal=self.goal,
            completed_actions=self.completed_actions, nodes=self.nodes, budgets=self.budgets,
            skill_budgets=self.skill_budgets, scene=self.scene,
            revision=self.revision, holding_state=self.holding_state,
            active_action=self.active_action, last_feedback=self.last_feedback))
