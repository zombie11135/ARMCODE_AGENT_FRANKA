import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.execution_state import classify_failure_constraints
from agents.plan_graph import PlanGraph
from agents.planning_agent import PlanningAgent


class ReplanningAgent:
    """
    Lightweight repair layer for the first closed-loop version.

    It reads executor failure reports, converts them into explicit constraints,
    and injects those constraints into the failed node of the previous plan.
    The next step can then regenerate executor code from this repaired graph.
    """

    def __init__(self, planner: Optional[PlanningAgent] = None):
        self.planner = planner or PlanningAgent(context=None, use_llm=False)

    def replan_from_failure_report(
        self,
        previous_graph: PlanGraph,
        failure_report: Dict[str, Any],
        output_json: Optional[str] = None,
        output_md: Optional[str] = None,
    ) -> PlanGraph:
        failed_node_id = (
            failure_report.get("failed_node_id")
            or failure_report.get("node_id")
            or failure_report.get("step_id")
        )
        constraints = self.build_constraints(failure_report)

        graph = self.planner.replan_with_constraints(
            previous_graph=previous_graph,
            new_constraints=constraints,
            failed_node_id=failed_node_id,
            output_path=output_json,
        )
        graph.metadata["replan_source"] = "ReplanningAgent.replan_from_failure_report"
        graph.metadata["failure_report"] = failure_report

        if output_json:
            graph.save_json(output_json)
        if output_md:
            path = Path(output_md)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(graph.to_markdown(), encoding="utf-8")

        return graph

    @staticmethod
    def build_constraints(failure_report: Dict[str, Any]) -> List[str]:
        return classify_failure_constraints(failure_report)

    @staticmethod
    def load_failure_report(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

