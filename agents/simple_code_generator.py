import json
from pathlib import Path
from typing import List, Optional

from agents.code_generator import CodeGenerator, GeneratedSkillCall
from agents.plan_graph import PlanGraph
from core.actions import SimpleAction, grasp, observe, place


class SimpleCodeGenerator:
    """
    Generate concise action code from PlanGraph.

    The generated file is intentionally small and readable. Execution is handled
    by SimpleArmCodeAgentSystem in-process so model/camera services can be reused.
    """

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.legacy_generator = CodeGenerator(project_root=str(self.project_root))

    def build_actions(
        self,
        graph: PlanGraph,
        recovery: bool = False,
    ) -> List[SimpleAction]:
        simple_actions = self._build_simple_actions(graph=graph)
        if simple_actions:
            self._validate_action_order(simple_actions)
            if recovery:
                return self._with_recovery_observes(simple_actions)
            return simple_actions

        calls = self.legacy_generator.build_skill_calls(graph)
        actions: List[SimpleAction] = []
        for call in calls:
            medicine = str(call.args.get("medicine_query") or "").strip()
            container = self._container_label(call)
            if recovery:
                actions.append(observe(medicine))
            actions.append(grasp(medicine))
            actions.append(place(container))
        self._validate_action_order(actions)
        return actions

    @staticmethod
    def _validate_action_order(actions: List[SimpleAction]) -> None:
        if not actions:
            raise ValueError('规划没有可执行动作：需要绑定 observe/grasp/place 技能，不能只返回文字 macro 节点。')
        holding = None
        for index, action in enumerate(actions, start=1):
            if not action.target.strip():
                raise ValueError(f'动作 {index} {action.op} 缺少 target，未启动执行')
            if action.op == 'grasp':
                if holding is not None:
                    raise ValueError(f'动作 {index} grasp({action.target}) 前尚未放置 {holding}')
                holding = action.target
            elif action.op == 'place':
                if holding is None:
                    raise ValueError(f'规划动作不完整：place({action.target}) 前缺少 grasp；未启动执行')
                holding = None

    def _build_simple_actions(self, graph: PlanGraph) -> List[SimpleAction]:
        actions: List[SimpleAction] = []
        for node in self._topological_nodes(graph):
            skill = str(node.skill or "").strip()
            if (not skill and node.node_type in ('pick', 'grasp', 'place', 'observe')
                    and not graph.child_nodes(node.node_id)):
                raise ValueError(f'规划节点 {node.node_id} ({node.label}) 未绑定 skill/target，不能跳过执行')
            if skill == 'pick':
                raise ValueError(f'规划节点 {node.node_id} 使用 pick，须先绑定为 grasp 和明确 target')
            args = node.args or {}
            target = str(args.get("target") or args.get("obj") or "").strip()
            if skill == "observe":
                if not target:
                    raise ValueError(f'规划节点 {node.node_id} observe 缺少 target')
                actions.append(observe(target))
            elif skill == "grasp":
                target = target or str(args.get("medicine_query") or args.get("query") or "").strip()
                if not target:
                    raise ValueError(f'规划节点 {node.node_id} grasp 缺少 target')
                actions.append(grasp(target))
            elif skill == "place":
                if not target:
                    name = str(args.get("container_name") or args.get("container") or "container").strip()
                    color = args.get("container_color")
                    target = f"{color} {name}" if color else name
                if target:
                    actions.append(place(target))
        return actions

    @staticmethod
    def _with_recovery_observes(actions: List[SimpleAction]) -> List[SimpleAction]:
        recovered: List[SimpleAction] = []
        for action in actions:
            if action.op == "grasp":
                if not recovered or not (
                    recovered[-1].op == "observe"
                    and recovered[-1].target == action.target
                ):
                    recovered.append(observe(action.target))
            recovered.append(action)
        return recovered

    @staticmethod
    def _topological_nodes(graph: PlanGraph):
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
            return graph.nodes
        return result

    def build_skill_calls(self, graph: PlanGraph) -> List[GeneratedSkillCall]:
        return self.legacy_generator.build_skill_calls(graph)

    def generate(
        self,
        graph: PlanGraph,
        output_path: Optional[str] = None,
        recovery: bool = False,
    ) -> str:
        actions = self.build_actions(graph=graph, recovery=recovery)
        script = self._render_script(actions)
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script, encoding="utf-8")
        return script

    @staticmethod
    def _container_label(call: GeneratedSkillCall) -> str:
        name = str(call.args.get("container_name") or "tray").strip()
        color = call.args.get("container_color")
        if color:
            return f"{color} {name}"
        return name

    @staticmethod
    def _render_script(actions: List[SimpleAction]) -> str:
        action_dicts = [action.__dict__ for action in actions]
        action_json = json.dumps(action_dicts, ensure_ascii=False, indent=2)
        lines = [
            "#!/usr/bin/env python3",
            "# Auto-generated concise action script.",
            "from skill_lib.simple_actions import grasp, observe, place",
            "",
            "ACTIONS = [",
        ]
        for action in actions:
            lines.append(f"    {action.op}({action.target!r}),")
        lines.extend(
            [
                "]",
                "",
                f"ACTION_JSON = {action_json!r}",
                "",
                "if __name__ == '__main__':",
                "    for index, action in enumerate(ACTIONS, start=1):",
                "        print(f'{index}. {action.op}({action.target})')",
                "",
            ]
        )
        return "\n".join(lines)
