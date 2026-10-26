"""Repair a blocked, not-yet-started grasp without replacing the robot session."""
import json
from pathlib import Path

from agents.plan_graph import PlanGraph
from agents.replanning_agent import ReplanningAgent
from agents.simple_code_generator import SimpleCodeGenerator
from core.actions import SimpleAction
from runtime.observation_memory import invalidate_camera


def repair_gate_before_grasp(runtime, robot, context, completed_actions, report):
    config = runtime.config
    if (not getattr(config, 'enable_geometry_occlusion', False)
            or not getattr(config, 'enable_replan', True)
            or not getattr(config, 'resume_from_failure', True)
            or runtime.replan_attempts >= max(0, int(getattr(config, 'max_replan_attempts', 1)))):
        return None
    evidence = report.get('evidence', {})
    if (report['failure_type'] not in {'nesyro_gate_replan', 'nesyro_gate_probe_exhausted'}
            or evidence.get('automatic_replan') is False
            or evidence.get('occlusion_recovery_attempts')):
        return None

    graph = PlanGraph.load_json(config.plan_json)
    if graph.goal != config.goal:
        raise ValueError('重规划图不属于当前任务，保留现场并停止。')
    repaired = ReplanningAgent().replan_from_failure_report(graph, report)
    generator = SimpleCodeGenerator(project_root=config.project_root)
    actions = runtime._resolve_action_targets(context, generator.build_actions(repaired, recovery=True))

    # Recovery adds observations. Match completed physical actions exactly so
    # repeated plate names cannot resume at the wrong target or replay a pick.
    cursor = 0
    for done in completed_actions:
        if done['op'] == 'observe':
            continue
        while cursor < len(actions) and actions[cursor].op == 'observe':
            cursor += 1
        if cursor >= len(actions) or actions[cursor].__dict__ != done:
            raise ValueError('修复计划与已完成动作不一致，不能自动重跑已完成目标。')
        cursor += 1
    pending = actions[cursor:]
    next_action = next((action for action in pending if action.op != 'observe'), None)
    if next_action is None or next_action.__dict__ != report['failed_action']:
        raise ValueError('修复计划没有从本次失败的抓取动作恢复。')

    runtime.replan_attempts += 1
    repaired.metadata['same_session_repair'] = dict(
        attempt=runtime.replan_attempts, completed_actions=list(completed_actions),
        clear_counts=dict(robot.occlusion_recovery_counts),
        buffer_slots_used=robot.geometry_buffer_slots_used)
    repaired.save_json(config.replan_json)
    Path(config.replan_md).parent.mkdir(parents=True, exist_ok=True)
    Path(config.replan_md).write_text(repaired.to_markdown(), encoding='utf-8')
    script = Path(config.generated_code)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(generator._render_script(pending), encoding='utf-8')
    runtime.replan_paths = dict(replan_json=config.replan_json, replan_md=config.replan_md)
    report['same_session_replan'] = dict(
        attempt=runtime.replan_attempts, max_attempts=config.max_replan_attempts,
        pending_actions=[action.__dict__ for action in pending], **runtime.replan_paths)
    Path(config.failure_report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    # Expire image evidence only. The task's containers, recovery counters and
    # occupied slots stay on the SAME robot object throughout the repair.
    invalidate_camera(context.camera, 'same_session_replan', discard_scene=False)
    robot.observations.pop(report['failed_action']['target'], None)
    print(f'[replanning] same-session retry {runtime.replan_attempts}/{config.max_replan_attempts}; '
          f'skip {len(completed_actions)} completed actions; keep clearing budgets and buffer slots')
    for action in pending:
        print(f'[replanning] pending: {action.op}({action.target})')
    return [SimpleAction(**done) for done in completed_actions] + pending
