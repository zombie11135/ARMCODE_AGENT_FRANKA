import argparse
import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.plan_graph import PlanGraph


def mermaid_escape(text) -> str:
    value = str(text or "")
    value = value.replace("\\", "\\\\")
    value = value.replace('"', "'")
    value = value.replace("\n", "<br/>")
    return value


def node_title(node) -> str:
    args = node.args or {}
    target = args.get("target")
    if target:
        return f"{node.node_id}: {node.node_type}({target})"
    if args.get("medicine_query"):
        return f"{node.node_id}: {node.node_type}({args.get('medicine_query')})"
    return f"{node.node_id}: {node.label or node.node_type}"


def node_body(node) -> str:
    parts = [node_title(node)]
    if node.skill:
        parts.append(f"skill={node.skill}")
    if node.instruction:
        parts.append(node.instruction)
    return "<br/>".join(mermaid_escape(part) for part in parts if part)


def node_shape(node, body: str) -> str:
    if node.node_type == "macro":
        return f'["{body}"]'
    if node.node_type in {"grasp", "pick"}:
        return f'(["{body}"])'
    if node.node_type == "place":
        return f'(["{body}"])'
    if node.node_type == "observe":
        return f'{{"{body}"}}'
    if node.node_type == "verify":
        return f'{{"{body}"}}'
    return f'["{body}"]'


def topological_nodes(graph: PlanGraph):
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


def graph_to_mermaid(graph: PlanGraph, layout: str = "linear") -> str:
    lines = [
        "flowchart TD",
        f'  title["{mermaid_escape(graph.goal)}"]',
    ]

    for node in graph.nodes:
        body = node_body(node)
        lines.append(f"  {node.node_id}{node_shape(node, body)}")

    if graph.root_id:
        lines.append(f"  title --> {graph.root_id}")

    if layout == "linear":
        ordered = topological_nodes(graph)
        root_id = graph.root_id
        sequence = []
        if root_id and graph.get_node(root_id):
            sequence.append(graph.get_node(root_id))
        for node in ordered:
            if node.node_id == root_id:
                continue
            sequence.append(node)
        for previous, current in zip(sequence, sequence[1:]):
            lines.append(f"  {previous.node_id} --> {current.node_id}")
    else:
        for edge in graph.edges:
            label = str(edge.condition or edge.edge_type or "").strip()
            if label and label != "control":
                lines.append(f'  {edge.source} -- "{mermaid_escape(label)}" --> {edge.target}')
            else:
                lines.append(f"  {edge.source} --> {edge.target}")

    lines.extend(
        [
            "  classDef grasp fill:#e8f4ff,stroke:#1677ff,stroke-width:2px;",
            "  classDef place fill:#ecfff1,stroke:#2da44e,stroke-width:2px;",
            "  classDef observe fill:#fff7e6,stroke:#fa8c16,stroke-width:2px;",
            "  classDef verify fill:#f9f0ff,stroke:#722ed1,stroke-width:2px;",
            "  classDef macro fill:#f6f8fa,stroke:#57606a,stroke-width:2px;",
        ]
    )

    for node in graph.nodes:
        cls = node.node_type if node.node_type in {"grasp", "place", "observe", "verify", "macro"} else "macro"
        if node.skill in {"grasp", "place", "observe"}:
            cls = node.skill
        lines.append(f"  class {node.node_id} {cls};")

    return "\n".join(lines) + "\n"


def mermaid_to_html(mermaid_text: str, title: str) -> str:
    escaped = html.escape(mermaid_text)
    page_title = html.escape(title or "Plan Graph")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{page_title}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f8fa;
      color: #24292f;
    }}
    header {{
      padding: 16px 24px;
      background: #ffffff;
      border-bottom: 1px solid #d0d7de;
    }}
    main {{
      padding: 24px;
      overflow: auto;
    }}
    .mermaid {{
      background: #ffffff;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 24px;
      min-width: 960px;
      font-size: 22px;
    }}
    .mermaid svg {{
      font-size: 22px !important;
    }}
    .mermaid .nodeLabel,
    .mermaid .edgeLabel,
    .mermaid .label {{
      font-size: 22px !important;
    }}
  </style>
</head>
<body>
  <header>
    <h2>{page_title}</h2>
  </header>
  <main>
    <pre class="mermaid">
{escaped}
    </pre>
  </main>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, securityLevel: "loose" }});
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Render PlanGraph JSON as Mermaid HTML.")
    parser.add_argument("--input", default="runs/current_plan/plan_graph.json")
    parser.add_argument("--output_html", default="runs/current_plan/plan_graph.html")
    parser.add_argument("--output_mmd", default="runs/current_plan/plan_graph.mmd")
    parser.add_argument(
        "--layout",
        choices=["linear", "edges"],
        default="linear",
        help="linear shows execution order; edges shows raw graph edges.",
    )
    args = parser.parse_args()

    graph = PlanGraph.load_json(args.input)
    mermaid = graph_to_mermaid(graph, layout=args.layout)

    mmd_path = Path(args.output_mmd)
    mmd_path.parent.mkdir(parents=True, exist_ok=True)
    mmd_path.write_text(mermaid, encoding="utf-8")

    html_path = Path(args.output_html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(mermaid_to_html(mermaid, graph.goal), encoding="utf-8")

    print("Mermaid:", mmd_path)
    print("HTML:", html_path)


if __name__ == "__main__":
    main()
