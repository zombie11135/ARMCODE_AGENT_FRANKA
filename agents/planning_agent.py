import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.plan_graph import PlanEdge, PlanGraph, PlanNode


PLANNER_SYSTEM_PROMPT = """
你是机械臂任务规划 agent。你需要参考 InstructFlow 的思想：
- 将长程任务分解成层级 instruction graph；
- 每个节点必须是语义明确的子目标；
- 可执行节点必须绑定已有技能；
- 失败约束会被注入到图中，用于后续重新规划或局部修复。

请只输出 JSON，不要输出 Markdown。JSON schema:
{
  "goal": "...",
  "root_id": "n0",
  "constraints": ["..."],
  "nodes": [
    {
      "node_id": "n0",
      "label": "...",
      "node_type": "macro|observe|grasp|place|verify",
      "instruction": "...",
      "skill": null 或技能名,
      "args": {},
      "preconditions": [],
      "effects": [],
      "constraints": [],
      "parent": null 或父节点 id,
      "metadata": {}
    }
  ],
  "edges": [
    {"source": "n1", "target": "n2", "edge_type": "control", "condition": null}
  ]
}

当前可用的执行技能只有三个：
- observe: 辅助观察目标。args 必须是 {"target": "..."}。target 可以是药名或容器名。
- grasp: 根据药名识别并抓取药盒。args 必须是 {"target": "药名"}。
- place: 将当前夹爪中的物体放入容器。args 必须是 {"target": "容器名"}。

执行技能绑定规则：
- observe/grasp/place 节点必须填写同名 skill 和非空 args.target；不能只有动作文字却写 skill=null。
- 每个 place 前必须有对应的 grasp；只有观察、尚不能抓取的计划不得包含 place。
- 对抓放任务，必须直接生成 grasp/place 节点，不要生成 pick_medicine_and_place_to_container、find_and_grasp_medicine、place_to_container、use_cached_place_plan 等旧高层技能。
- 常规顺序是 grasp("药名") -> place("容器")。只有失败恢复、目标看不清或需要近距离确认时，才在 grasp 前加入 observe("药名")。
- grasp 只能用于“药盒/药品包装盒”，不要用于圆柱形管状物、瓶子、饮料、线材等非药盒物体。
- grasp 的 target 必须优先使用包装上可读文字或药品名称，例如“布洛芬”“抗病毒口服液”“蒙脱石散”。
- place 的 target 必须是容器，例如“container”“pink tray”“blue tray”。
- 如果任务明确要求“粉色盘子/蓝色托盘/绿色盒子”等目标容器颜色，place target 写成 "pink tray" / "blue tray" / "green container"。
- 如果任务要求“收纳盒/盒子/半透明盒/透明盒/container/box/bin”，且没有明确指定颜色，则 place target 必须是 "container"；不要因为场景中看到粉色盘子就自动改成 "pink tray"。
- medicine_query 必须优先使用包装上可读文字或药品名称，例如“布洛芬”“抗病毒口服液”“蒙脱石散”。
- 禁止把“白色蓝色药盒”“绿色白色药盒”“左边药盒”“圆柱形管状物”这类颜色、形状或方位描述当作 medicine_query。
- 用户已明确指定药名时，grasp.target 使用该药名，抓取技能会重新检测并用 OCR 检索确认目标；不能把颜色描述直接视为药名已确认。
- 用户没有提供药名、且画面只能看到颜色/形状时，该物体不要绑定抓取技能；应只生成 observe/verify 节点并在 constraints 中写明“需要更近距离 OCR 或人工确认药名后再抓取”，不能再接 place。
- 对“整理所有药品盒子”任务，只规划已确认是药盒且有可检索文字的目标；不确定的物体应保留为待确认目标，不要直接执行抓取。
"""


class PlanningAgent:
    """
    Planner that creates an InstructFlow-style instruction graph.

    The default planner is intentionally deterministic and conservative, so it
    can be used before an LLM planner is wired in. If context.vlm is available,
    use_llm=True can ask the VLM/LLM service to produce the same graph schema.
    """

    def __init__(self, context=None, use_llm: bool = False):
        self.context = context
        self.use_llm = use_llm

    def plan(
        self,
        task_goal: str,
        scene_text: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        use_vdm: bool = False,
        output_path: Optional[str] = None,
    ) -> PlanGraph:
        scene_text = scene_text or ""
        constraints = constraints or []

        if use_vdm and not scene_text:
            scene_text = self._run_vdm()

        graph = None

        if self.use_llm and self.context is not None and getattr(self.context, "vlm", None):
            try:
                graph = self._plan_with_llm(
                    task_goal=task_goal,
                    scene_text=scene_text,
                    constraints=constraints,
                )
            except Exception as e:
                graph = self._plan_heuristic(
                    task_goal=task_goal,
                    scene_text=scene_text,
                    constraints=[
                        *constraints,
                        f"LLM planner failed, fallback to heuristic planner: {e}",
                    ],
                )
        else:
            graph = self._plan_heuristic(
                task_goal=task_goal,
                scene_text=scene_text,
                constraints=constraints,
            )

        graph = self._normalize_simple_skill_bindings(graph, task_goal)
        graph = self._ensure_multi_target_plan(
            graph=graph,
            task_goal=task_goal,
            scene_text=scene_text,
            constraints=constraints,
        )
        graph = self._normalize_container_from_task(graph, task_goal)
        graph = self._convert_to_simple_action_graph(graph)
        graph = self._linearize_simple_action_edges(graph)

        from services.object_names import bind_plan_names
        graph = bind_plan_names(graph, task_goal, getattr(self.context, 'object_references', None),
                                getattr(self.context, 'target_name_bindings', None))

        if output_path:
            graph.save_json(output_path)

        return graph

    @classmethod
    def _normalize_simple_skill_bindings(cls, graph: PlanGraph, task_goal: str) -> PlanGraph:
        """Bind an existing explicit pick node; never invent a missing grasp step."""
        names = cls._extract_explicit_goal_medicines(task_goal)
        if cls._is_all_medicine_task(task_goal):
            names = []
        # A color/position description alone is not a medicine identity.
        generic = r'(?:(?:红|绿|蓝|白|黑|黄|粉|紫|橙|灰)色|左边|右边|上面|下面|目标|那个|这个|药盒|盒子|药品|物体|的|\s)+'
        names = [name for name in names if not re.fullmatch(generic, name)]
        repairs = []
        for node in graph.nodes:
            if graph.child_nodes(node.node_id):
                continue  # Hierarchical task labels need not be executable skills.
            original_skill = node.skill
            original_args = dict(node.args or {})
            skill = str(node.skill or '').strip()
            kind = str(node.node_type or '').strip()
            if skill == 'pick':
                skill = 'grasp'
            if not skill and kind in ('observe', 'pick', 'grasp', 'place'):
                skill = 'grasp' if kind == 'pick' else kind
            if skill not in ('observe', 'grasp', 'place'):
                continue
            args = dict(node.args or {})
            target = str(args.get('target') or args.get('obj') or '').strip()
            if skill == 'grasp' and not target:
                target = str(args.get('medicine_query') or args.get('query') or '').strip()
                if not target:
                    # Recover only a user-named medicine also named in THIS pick node.
                    text = f'{node.label}\n{node.instruction}'
                    matching = [name for name in names if name in text]
                    if len(matching) == 1:
                        target = matching[0]
            if not target:
                continue  # Code generation reports an incomplete node instead of skipping it.
            args['target'] = target
            node.skill, node.args = skill, args
            if original_skill != skill or original_args != args:
                repairs.append(dict(node_id=node.node_id, original_skill=original_skill,
                                    original_args=original_args, skill=skill, target=target))
        if repairs:
            graph.metadata['skill_binding_repairs'] = repairs
        return graph

    @classmethod
    def _normalize_container_from_task(cls, graph: PlanGraph, task_goal: str) -> PlanGraph:
        if not cls._task_requests_plain_container(task_goal):
            return graph

        for node in graph.nodes:
            if not isinstance(node.args, dict):
                continue
            if node.skill == "place":
                node.args["target"] = "container"
                node.instruction = re.sub(
                    r"(粉色|粉红色|粉红|蓝色|绿色|红色|黄色|pink|blue|green|red|yellow)?(盘子|托盘|tray|plate)",
                    "收纳盒",
                    node.instruction,
                    flags=re.IGNORECASE,
                )
                continue
            if "container_name" not in node.args and "container_color" not in node.args:
                continue
            node.args["container_name"] = "container"
            node.args["container_color"] = None
            node.instruction = re.sub(
                r"(粉色|粉红色|粉红|蓝色|绿色|红色|黄色|pink|blue|green|red|yellow)?(盘子|托盘|tray|plate)",
                "收纳盒",
                node.instruction,
                flags=re.IGNORECASE,
            )
        graph.metadata["container_normalized_from_task"] = "plain_container_no_color"
        return graph

    @classmethod
    def _convert_to_simple_action_graph(cls, graph: PlanGraph) -> PlanGraph:
        simple_skills = {"observe", "grasp", "place"}
        executable = [node for node in graph.nodes if node.skill]
        if executable and all(node.skill in simple_skills for node in executable):
            graph.metadata["action_graph"] = "simple_native"
            return graph

        pairs = []
        for node in graph.nodes:
            args = node.args or {}
            if node.skill == "pick_medicine_and_place_to_container":
                medicine = str(args.get("medicine_query") or args.get("query") or "").strip()
                container = cls._container_target(
                    args.get("container_name") or "container",
                    args.get("container_color"),
                )
                if medicine:
                    pairs.append((medicine, container))
            elif node.skill == "find_and_grasp_medicine":
                medicine = str(args.get("query") or args.get("medicine_query") or "").strip()
                if medicine:
                    pairs.append((medicine, "container"))

        if not pairs:
            return graph

        simple = PlanGraph(
            goal=graph.goal,
            scene_text=graph.scene_text,
            constraints=list(graph.constraints),
            metadata={
                **dict(graph.metadata),
                "action_graph": "simple_converted",
                "converted_from_legacy_skills": True,
            },
        )
        root = simple.add_node(
            PlanNode(
                node_id="n0",
                label="执行简短抓放动作序列",
                node_type="macro",
                instruction="按照 grasp/place 简短技能顺序执行任务。",
                skill=None,
                args={},
                preconditions=["机器人处于可运动状态", "夹爪可控", "相机服务可用"],
                effects=["目标物体被放入目标容器"],
            )
        )
        simple.root_id = root.node_id

        previous = root.node_id
        for index, (medicine, container) in enumerate(pairs, start=1):
            grasp_id = f"n{2 * index - 1}"
            place_id = f"n{2 * index}"
            simple.add_node(
                PlanNode(
                    node_id=grasp_id,
                    label=f"抓取{medicine}",
                    node_type="grasp",
                    instruction=f"实时拍摄腕部图像，识别并抓取药盒“{medicine}”。",
                    skill="grasp",
                    args={"target": medicine},
                    parent=root.node_id,
                    effects=[f"夹爪抓住 {medicine}"],
                )
            )
            simple.add_node(
                PlanNode(
                    node_id=place_id,
                    label=f"放入{container}",
                    node_type="place",
                    instruction=f"将夹爪中的“{medicine}”放入“{container}”。",
                    skill="place",
                    args={"target": container},
                    parent=root.node_id,
                    preconditions=[f"夹爪抓住 {medicine}"],
                    effects=[f"{medicine} 位于 {container} 中"],
                )
            )
            simple.add_edge(previous, grasp_id)
            simple.add_edge(grasp_id, place_id)
            previous = place_id

        return simple

    @classmethod
    def _linearize_simple_action_edges(cls, graph: PlanGraph) -> PlanGraph:
        simple_skills = {"observe", "grasp", "place"}
        executable = [node for node in graph.nodes if node.skill]
        if not executable or any(node.skill not in simple_skills for node in executable):
            return graph

        ordered = cls._topological_nodes(graph)
        if not ordered:
            return graph

        root_id = graph.root_id or ordered[0].node_id
        root = graph.get_node(root_id)

        sequence = []
        if root is not None:
            sequence.append(root)

        for node in ordered:
            if node.node_id == root_id:
                continue
            sequence.append(node)

        if len(sequence) <= 1:
            return graph

        graph.edges = []
        for previous, current in zip(sequence, sequence[1:]):
            graph.edges.append(PlanEdge(source=previous.node_id, target=current.node_id))

        graph.metadata["edge_layout"] = "linear_execution_order"
        return graph

    @staticmethod
    def _topological_nodes(graph: PlanGraph) -> List[PlanNode]:
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

    @staticmethod
    def _task_requests_plain_container(task_goal: str) -> bool:
        text = str(task_goal or "").lower()
        has_container = any(
            token in text
            for token in [
                "收纳盒",
                "整理盒",
                "透明盒",
                "半透明盒",
                "半透明",
                "透明",
                "container",
                "box",
                "bin",
                "organizer",
            ]
        )
        if not has_container:
            return False
        has_explicit_color = any(
            token in text
            for token in [
                "粉色",
                "粉红色",
                "粉红",
                "蓝色",
                "绿色",
                "红色",
                "黄色",
                "pink",
                "blue",
                "green",
                "red",
                "yellow",
            ]
        )
        return not has_explicit_color

    def replan_with_constraints(
        self,
        previous_graph: PlanGraph,
        new_constraints: List[str],
        failed_node_id: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> PlanGraph:
        graph = PlanGraph.from_dict(previous_graph.to_dict())
        graph.apply_constraints(new_constraints, target_node_id=failed_node_id)
        graph.metadata["replanned_at"] = datetime.now().isoformat(timespec="seconds")
        graph.metadata["failed_node_id"] = failed_node_id

        # For now constraints are injected directly. This gives the code
        # generator and future LLM planner a stable symbolic repair surface.
        if output_path:
            graph.save_json(output_path)

        return graph

    def _run_vdm(self) -> str:
        if self.context is None:
            raise RuntimeError("PlanningAgent missing RuntimeContext for VDM.")

        from skill_lib.perception.vdm import VDMSkill

        result = VDMSkill(self.context).run(use_wrist=True)
        if not result.success:
            raise RuntimeError(result.error)

        return result.data.get("scene_text", "")

    def _plan_with_llm(
        self,
        task_goal: str,
        scene_text: str,
        constraints: List[str],
    ) -> PlanGraph:
        vlm = (
            getattr(self.context, "planner_llm", None)
            or getattr(self.context, "fallback_vlm", None)
            or self.context.require("vlm")
        )
        prompt = self._build_llm_prompt(task_goal, scene_text, constraints)
        text = vlm.chat(text=prompt, max_tokens=2048, temperature=0.1)
        data = self._extract_json(text)
        graph = PlanGraph.from_dict(data)
        graph.scene_text = scene_text
        graph.metadata.setdefault("planner", "llm")
        graph.metadata.setdefault("source", "PlanningAgent._plan_with_llm")
        return graph

    def _plan_heuristic(
        self,
        task_goal: str,
        scene_text: str,
        constraints: List[str],
    ) -> PlanGraph:
        medicine_queries = self._extract_medicine_queries(task_goal, scene_text)
        if len(medicine_queries) > 1:
            return self._plan_multi_pick_place_heuristic(
                task_goal=task_goal,
                scene_text=scene_text,
                constraints=constraints,
                medicine_queries=medicine_queries,
            )

        medicine_query = self._extract_medicine_query(task_goal, scene_text)
        container_name = self._extract_container_name(task_goal, scene_text)
        container_color = self._extract_container_color(task_goal, scene_text)

        graph = PlanGraph(
            goal=task_goal,
            scene_text=scene_text,
            constraints=list(constraints),
            metadata={
                "planner": "heuristic",
                "style": "simple_action_graph",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "medicine_query": medicine_query,
                "container_name": container_name,
                "container_color": container_color,
            },
        )

        root = graph.add_node(
            PlanNode(
                node_id="n0",
                label="完成药盒抓取并放置到容器",
                node_type="macro",
                instruction=f"将目标药品“{medicine_query}”放置到“{self._container_target(container_name, container_color)}”中。",
                skill=None,
                args={},
                preconditions=["机器人处于可运动状态", "夹爪可控", "相机服务可用"],
                effects=[f"{medicine_query} 位于 {container_name} 中"],
                constraints=[
                    "执行器会在任务开始时空手预观测所有 place 目标容器。",
                ],
            )
        )
        graph.root_id = root.node_id

        graph.add_node(
            PlanNode(
                node_id="n1",
                label=f"抓取{medicine_query}",
                node_type="grasp",
                instruction=f"实时拍摄腕部图像，识别并抓取药盒“{medicine_query}”。",
                skill="grasp",
                args={"target": medicine_query},
                parent=root.node_id,
                preconditions=["目标药盒可见或可通过药名检索"],
                effects=[f"夹爪抓住 {medicine_query}"],
            )
        )

        graph.add_node(
            PlanNode(
                node_id="n2",
                label=f"放入{self._container_target(container_name, container_color)}",
                node_type="place",
                instruction=f"将夹爪中的药盒放入“{self._container_target(container_name, container_color)}”。",
                skill="place",
                args={"target": self._container_target(container_name, container_color)},
                parent=root.node_id,
                preconditions=[f"夹爪抓住 {medicine_query}"],
                effects=[f"{medicine_query} 已释放到 {container_name}"],
            )
        )

        graph.add_node(
            PlanNode(
                node_id="n3",
                label="验证任务结果",
                node_type="verify",
                instruction="用可用视觉或执行反馈判断药盒是否已放入容器。",
                skill=None,
                args={},
                parent=root.node_id,
                preconditions=[f"{medicine_query} 已释放到 {container_name}"],
                effects=["任务完成或生成失败约束"],
            )
        )

        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")

        return graph

    def _plan_multi_pick_place_heuristic(
        self,
        task_goal: str,
        scene_text: str,
        constraints: List[str],
        medicine_queries: List[str],
    ) -> PlanGraph:
        container_name = self._extract_container_name(task_goal, scene_text)
        container_colors = self._extract_container_colors(task_goal) or [
            self._extract_container_color(task_goal, scene_text)
        ]
        container_colors = [color for color in container_colors if color]
        if not container_colors:
            container_colors = [None]

        graph = PlanGraph(
            goal=task_goal,
            scene_text=scene_text,
            constraints=list(constraints),
            metadata={
                "planner": "heuristic",
                "style": "multi_target_simple_pick_place",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "medicine_queries": medicine_queries,
                "container_name": container_name,
                "container_colors": container_colors,
                "classification_policy": "round_robin_by_detected_order",
            },
        )
        if len(container_colors) > 1:
            graph.constraints.append(
                "任务要求分类到多个容器，但未提供明确分类规则；当前按识别到的药盒顺序轮流分配到目标颜色盘子。"
            )

        root = graph.add_node(
            PlanNode(
                node_id="n0",
                label="整理所有药盒到目标容器",
                node_type="macro",
                instruction="将识别到的所有可确认药盒按当前分类策略放入目标容器中。",
                skill=None,
                args={},
                preconditions=["机器人处于可运动状态", "夹爪可控", "相机服务可用"],
                effects=["所有已确认药盒进入目标容器"],
            )
        )
        graph.root_id = root.node_id

        previous = root.node_id
        for index, medicine_query in enumerate(medicine_queries, start=1):
            color = container_colors[(index - 1) % len(container_colors)]
            container_target = self._container_target(container_name, color)
            grasp_id = f"n{2 * index - 1}"
            place_id = f"n{2 * index}"
            graph.add_node(
                PlanNode(
                    node_id=grasp_id,
                    label=f"抓取{medicine_query}",
                    node_type="grasp",
                    instruction=f"实时拍摄腕部图像，识别并抓取药盒“{medicine_query}”。",
                    skill="grasp",
                    args={"target": medicine_query},
                    parent=root.node_id,
                    preconditions=["目标药盒可见或可通过药名检索"],
                    effects=[f"夹爪抓住 {medicine_query}"],
                    metadata={"classification_index": index - 1},
                )
            )
            graph.add_node(
                PlanNode(
                    node_id=place_id,
                    label=f"放入{container_target}",
                    node_type="place",
                    instruction=f"将夹爪中的“{medicine_query}”放入“{container_target}”。",
                    skill="place",
                    args={"target": container_target},
                    parent=root.node_id,
                    preconditions=[f"夹爪抓住 {medicine_query}"],
                    effects=[f"{medicine_query} 位于 {container_target} 中"],
                    metadata={"classification_index": index - 1},
                )
            )
            graph.add_edge(previous, grasp_id)
            graph.add_edge(grasp_id, place_id)
            previous = place_id

        return graph

    def _ensure_multi_target_plan(
        self,
        graph: PlanGraph,
        task_goal: str,
        scene_text: str,
        constraints: List[str],
    ) -> PlanGraph:
        medicine_queries = self._extract_medicine_queries(task_goal, scene_text)
        if len(medicine_queries) <= 1:
            return graph

        planned = []
        for node in graph.nodes:
            args = node.args or {}
            value = args.get("medicine_query") or args.get("query")
            if not value and node.skill == "grasp":
                value = args.get("target")
            if value:
                planned.append(str(value))
        missing = [name for name in medicine_queries if name not in planned]
        if not missing and len(planned) >= len(medicine_queries):
            return graph

        expanded = self._plan_multi_pick_place_heuristic(
            task_goal=task_goal,
            scene_text=scene_text,
            constraints=[
                *constraints,
                "原始规划未覆盖所有已识别药盒，已自动展开为多目标抓放计划。",
            ],
            medicine_queries=medicine_queries,
        )
        expanded.metadata["source"] = "PlanningAgent._ensure_multi_target_plan"
        return expanded

    @staticmethod
    def _container_target(container_name: str, container_color: Optional[str] = None) -> str:
        name = str(container_name or "container").strip() or "container"
        color = str(container_color or "").strip()
        if color:
            return f"{color} {name}"
        return name

    def _build_llm_prompt(
        self,
        task_goal: str,
        scene_text: str,
        constraints: List[str],
    ) -> str:
        references = getattr(self.context, "object_references", None)
        catalog = references.catalog() if references is not None else []
        reference_rules = ""
        if catalog:
            reference_rules = ("\n补充的参考图库规则（对下列登记物体优先于药盒/OCR限制）：\n"
                "登记名称或别名可直接作为 observe/grasp 的 target，包括方块、瓶子等非药盒物体。"
                "图库是身份先验，不证明物体当前在场；执行前仍由当前图像匹配确认，禁止凭图库虚构可见物体。"
                "grasp 内部会按需选择图片匹配、分割或 OCR，不要为这些基础操作生成额外任务节点。"
                "未登记物体继续遵循原有识别规则；注册新参考图必须由用户明确提供标签，不能在执行中自动登记。\n"
                "正式名称和别名属于同一 object_id。品牌/描述可能对应多个物品，不得擅自选择一个。"
                "用户只说品牌或共享描述时，保留其原词作为 target，交给执行前场景解析。"
                "已明确身份可用 object_id 作 target，requested_name 保存用户原称呼。"
                "陌生叫法可从本目录提出候选，不能虚构编号或自动写入别名。\n"
                + json.dumps(catalog, ensure_ascii=False)
                + "\n仅本任务有效的用户绑定：" + json.dumps(getattr(self.context, 'target_name_bindings', {}) or {}, ensure_ascii=False))
        constraints_text = "\n".join([f"- {c}" for c in constraints]) or "- 无"
        return f"""{PLANNER_SYSTEM_PROMPT}
{reference_rules}

任务目标：
{task_goal}

VDM 场景描述：
{scene_text or "无"}

已有约束：
{constraints_text}

请生成 instruction graph JSON。
"""

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _extract_medicine_query(task_goal: str, scene_text: str) -> str:
        explicit = PlanningAgent._extract_explicit_goal_medicines(task_goal)
        if len(explicit) == 1 and not PlanningAgent._is_all_medicine_task(task_goal):
            return explicit[0]
        text = f"{task_goal}\n{scene_text}"
        for name in PlanningAgent._known_medicine_names():
            if name in text:
                return name

        patterns = [
            r"把(.+?)(?:放|放置|移动|拿|抓)",
            r"抓取(.+?)(?:并|然后|放|到)",
            r"目标(?:药品|药盒|物体)[：: ]*([^，。,.\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, task_goal)
            if match:
                return match.group(1).strip()

        return "目标药盒"

    @staticmethod
    def _known_medicine_names() -> List[str]:
        return [
            "布洛芬缓释胶囊",
            "布洛芬",
            "抗病毒口服液",
            "抗病毒",
            "蒙脱石散",
            "西瓜霜",
            "999",
        ]

    @classmethod
    def _extract_medicine_queries(cls, task_goal: str, scene_text: str) -> List[str]:
        goal_queries = cls._extract_known_medicines_in_text(task_goal)
        explicit = cls._extract_explicit_goal_medicines(task_goal)
        if explicit and not cls._is_all_medicine_task(task_goal):
            return explicit
        if goal_queries:
            return goal_queries

        scene_queries = cls._extract_known_medicines_in_text(scene_text)
        if scene_queries:
            return scene_queries

        return explicit

    @classmethod
    def _extract_explicit_goal_medicines(cls, task_goal: str) -> List[str]:
        match = re.search(
            r"(?:将|把)(.+?)(?:放到|放入|放置到|移动到|拿到)",
            str(task_goal or ""),
        )
        if not match:
            return []
        fragments = re.split(r"(?:和|以及|及|、|,|，)", match.group(1))
        queries = []
        for fragment in fragments:
            name = re.sub(r"^(?:药盒|药品|药物)", "", fragment.strip())
            name = re.sub(r"(?:药盒|药品|药物)$", "", name).strip()
            known = cls._extract_known_medicines_in_text(name)
            values = known or ([name] if name else [])
            for value in values:
                if value not in queries:
                    queries.append(value)
        return queries

    @classmethod
    def _extract_known_medicines_in_text(cls, text: str) -> List[str]:
        text = str(text or "")
        matches = []
        for name in cls._known_medicine_names():
            index = text.find(name)
            if index < 0:
                continue
            if name == "抗病毒" and "抗病毒口服液" in text:
                continue
            if name == "布洛芬" and "布洛芬缓释胶囊" in text:
                continue
            matches.append((index, -len(name), name))
        matches.sort()
        return [name for _, _, name in matches]

    @staticmethod
    def _is_all_medicine_task(task_goal: str) -> bool:
        text = str(task_goal or "")
        return (
            any(token in text for token in ["所有", "全部", "全部的", "桌面上的药盒", "药盒分类", "整理药盒", "分类"])
            and any(token in text for token in ["药盒", "药品", "药物"])
        )

    @staticmethod
    def _extract_container_name(task_goal: str, scene_text: str) -> str:
        text = f"{task_goal}\n{scene_text}"
        container_aliases = [
            ("tray", ["托盘", "盘子", "tray", "plate"]),
            ("container", ["容器", "盒子", "收纳盒", "整理盒", "透明盒", "半透明盒", "container", "box", "bin", "organizer"]),
            ("bowl", ["碗", "bowl"]),
        ]
        for canonical, aliases in container_aliases:
            if any(alias in text for alias in aliases):
                return canonical

        patterns = [
            r"(?:放到|放入|放置到|放进)([^，。,.\n]+)",
            r"容器[：: ]*([^，。,.\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, task_goal)
            if match:
                value = match.group(1).strip()
                if value:
                    return value

        return "tray"

    @staticmethod
    def _extract_container_color(task_goal: str, scene_text: str) -> Optional[str]:
        text = f"{task_goal}\n{scene_text}"
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
                for container_token in container_tokens:
                    if f"{alias}{container_token}" in text or f"{alias}的{container_token}" in text:
                        return canonical

        return None

    @staticmethod
    def _extract_container_colors(text: str) -> List[str]:
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
        hits = []
        text = str(text or "")
        for canonical, aliases in color_aliases:
            for alias in aliases:
                for token in container_tokens:
                    patterns = [f"{alias}{token}", f"{alias}的{token}"]
                    for pattern in patterns:
                        start = text.find(pattern)
                        if start >= 0:
                            hits.append((start, canonical))
                if alias in text and any(token in text for token in container_tokens):
                    hits.append((text.find(alias), canonical))
        colors = []
        for _, color in sorted(hits, key=lambda item: item[0]):
            if color not in colors:
                colors.append(color)
        return colors
