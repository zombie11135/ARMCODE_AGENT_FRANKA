import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PlanNode:
    node_id: str
    label: str
    node_type: str
    instruction: str
    skill: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)


@dataclass
class PlanEdge:
    source: str
    target: str
    edge_type: str = "control"
    condition: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)


@dataclass
class PlanGraph:
    goal: str
    scene_text: str = ""
    root_id: Optional[str] = None
    nodes: List[PlanNode] = field(default_factory=list)
    edges: List[PlanEdge] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: PlanNode) -> PlanNode:
        if self.get_node(node.node_id) is not None:
            raise ValueError(f"Duplicate node_id: {node.node_id}")
        self.nodes.append(node)
        if self.root_id is None:
            self.root_id = node.node_id
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str = "control",
        condition: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanEdge:
        edge = PlanEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            condition=condition,
            metadata=metadata or {},
        )
        self.edges.append(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[PlanNode]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def child_nodes(self, parent_id: str) -> List[PlanNode]:
        return [node for node in self.nodes if node.parent == parent_id]

    def executable_nodes(self) -> List[PlanNode]:
        return [node for node in self.nodes if node.skill]

    def apply_constraints(self, constraints: List[str], target_node_id: Optional[str] = None):
        clean = [str(c).strip() for c in constraints if str(c).strip()]
        if not clean:
            return

        if target_node_id:
            node = self.get_node(target_node_id)
            if node is None:
                raise KeyError(f"Unknown target_node_id: {target_node_id}")
            node.constraints.extend([c for c in clean if c not in node.constraints])
        else:
            self.constraints.extend([c for c in clean if c not in self.constraints])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "scene_text": self.scene_text,
            "root_id": self.root_id,
            "constraints": self.constraints,
            "metadata": self.metadata,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        graph = cls(
            goal=data.get("goal", ""),
            scene_text=data.get("scene_text", ""),
            root_id=data.get("root_id"),
            constraints=list(data.get("constraints", [])),
            metadata=dict(data.get("metadata", {})),
        )
        graph.nodes = [PlanNode.from_dict(x) for x in data.get("nodes", [])]
        graph.edges = [PlanEdge.from_dict(x) for x in data.get("edges", [])]
        return graph

    def save_json(self, path: str) -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return str(output_path)

    @classmethod
    def load_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_markdown(self) -> str:
        lines = [
            f"# PlanGraph: {self.goal}",
            "",
            f"- root: `{self.root_id}`",
        ]

        if self.constraints:
            lines.append("- global constraints:")
            for constraint in self.constraints:
                lines.append(f"  - {constraint}")

        lines.extend(["", "## Nodes"])
        for node in self.nodes:
            skill = node.skill or "-"
            parent = node.parent or "-"
            lines.append(
                f"- `{node.node_id}` [{node.node_type}] {node.label} "
                f"(skill={skill}, parent={parent})"
            )
            lines.append(f"  - instruction: {node.instruction}")
            if node.constraints:
                lines.append(f"  - constraints: {'; '.join(node.constraints)}")

        lines.extend(["", "## Edges"])
        for edge in self.edges:
            condition = f" if {edge.condition}" if edge.condition else ""
            lines.append(f"- `{edge.source}` -> `{edge.target}` ({edge.edge_type}){condition}")

        return "\n".join(lines)
