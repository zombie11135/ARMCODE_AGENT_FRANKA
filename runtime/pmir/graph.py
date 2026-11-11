"""Inspectable causal edges; resource conflicts are separate node requirements."""
from dataclasses import asdict, dataclass, field
import json


@dataclass
class Node:
    id: str
    kind: str
    phase: str
    target: str
    destination: str | None = None
    inputs: tuple = ()
    resources: tuple = ()
    reads: tuple = ()
    writes: tuple = ()
    owner: bool = False
    physical: bool = False
    gate: str | None = None
    conditions: tuple = ()
    line: int = 0
    arguments: tuple = ()


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str
    reason: str


@dataclass
class Graph:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)

    def link(self, source, target, kind, reason):
        edge = Edge(source, target, kind, reason)
        if source != target and edge not in self.edges:
            self.edges.append(edge)

    def parents(self, node_id):
        return {e.source for e in self.edges if e.target == node_id}

    def validate(self):
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes) or any(e.source not in ids or e.target not in ids for e in self.edges):
            raise ValueError('Duplicate nodes or dangling dependency')
        done = set()
        while len(done) < len(ids):
            ready = {i for i in ids-done if self.parents(i) <= done}
            if not ready:
                raise ValueError('Dependency cycle')
            done.update(ready)

    def to_dict(self):
        return dict(schema='pmir_v1', nodes=[asdict(n) for n in self.nodes],
                    edges=[asdict(e) for e in self.edges],
                    note='Resources are scheduling constraints, not causal edges. '
                         'Physical effects come from skill contracts, not learned inference.')

    def mermaid(self):
        lines = ['flowchart TD']
        for n in self.nodes:
            label = f'{n.kind}: {n.phase} / {n.target}'
            lines.append(f'  {n.id}[{json.dumps(label)}]')
            if n.gate:
                lines.append(f'  gate_{n.id}{{{json.dumps(n.gate)}}} -. confirmed event .-> {n.id}')
        for e in self.edges:
            lines.append(f'  {e.source} -->|{e.kind}| {e.target}')
        return '\n'.join(lines)+'\n'
