import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.planning_agent import PlanningAgent
from agents.simple_agent_system import SimpleArmCodeAgentSystem
from agents.simple_code_generator import SimpleCodeGenerator
from skill_lib.simple_actions import grasp, place


SEPARATE_DESTINATIONS = "将西瓜霜放到蓝色盘子中，将抗病毒口服液放到粉色盘子中"
EXPECTED_ACTIONS = [
    ("grasp", "西瓜霜"), ("place", "blue tray"),
    ("grasp", "抗病毒口服液"), ("place", "pink tray"),
]


def test_repeated_transfer_clauses_keep_all_names_and_their_destinations():
    graph = PlanningAgent().plan(SEPARATE_DESTINATIONS)
    assert graph.metadata["medicine_queries"] == ["西瓜霜", "抗病毒口服液"]
    assert [(a.op, a.target) for a in SimpleCodeGenerator().build_actions(graph)] == EXPECTED_ACTIONS


@pytest.mark.parametrize("kind", ["macro_only", "missing_second", "wrong_destination", "valid"])
def test_llm_transfer_coverage_checks_actions_and_object_destination_pairs(kind):
    if kind == "macro_only":
        nodes = [dict(node_id=f"n{i}", label=instruction, node_type="macro",
                      instruction=instruction, skill=None, args={})
                 for i, instruction in enumerate([
                     "抓取西瓜霜并放置到蓝色盘子", "抓取抗病毒口服液并放置到粉色盘子"])]
    else:
        actions = EXPECTED_ACTIONS[:2] if kind == "missing_second" else list(EXPECTED_ACTIONS)
        if kind == "wrong_destination":
            actions[1] = ("place", "pink tray")
        nodes = [dict(node_id=f"n{i}", label=target, node_type=op, instruction=target,
                      skill=op, args={"target": target}) for i, (op, target) in enumerate(actions)]
    llm = Mock()
    llm.chat.return_value = json.dumps(dict(goal=SEPARATE_DESTINATIONS, root_id="n0", nodes=nodes,
                                          edges=[], constraints=["保留原计划约束"]))
    planner = PlanningAgent(context=SimpleNamespace(vlm=llm, planner_llm=llm), use_llm=True)
    graph = planner.plan(SEPARATE_DESTINATIONS)
    assert [(a.op, a.target) for a in SimpleCodeGenerator().build_actions(graph)] == EXPECTED_ACTIONS
    assert "保留原计划约束" in graph.constraints
    if kind == "valid":
        assert graph.metadata["planner"] == "llm"
    llm.chat.assert_called_once()


def test_grouped_unknown_names_and_different_container_types_keep_explicit_mapping():
    goal = "把阿莫西林和西瓜霜放进蓝色盘子，然后把泡腾片放入收纳盒"
    graph = PlanningAgent().plan(goal)
    assert [(a.op, a.target) for a in SimpleCodeGenerator().build_actions(graph)] == [
        ("grasp", "阿莫西林"), ("place", "blue tray"),
        ("grasp", "西瓜霜"), ("place", "blue tray"),
        ("grasp", "泡腾片"), ("place", "container"),
    ]


def test_shared_destination_list_keeps_existing_round_robin_policy():
    graph = PlanningAgent().plan("将西瓜霜和抗病毒口服液放到蓝色和粉色盘子中")
    assert graph.metadata["classification_policy"] == "round_robin_by_detected_order"
    assert [(a.op, a.target) for a in SimpleCodeGenerator().build_actions(graph)] == EXPECTED_ACTIONS


def test_explicit_two_medicines_expand_to_two_pick_place_pairs() -> None:
    goal = "将西瓜霜和抗病毒口服液放到粉色盘子中"

    graph = PlanningAgent(context=None, use_llm=False).plan(task_goal=goal)
    actions = SimpleCodeGenerator().build_actions(graph)

    assert graph.metadata["medicine_queries"] == ["西瓜霜", "抗病毒口服液"]
    assert [(action.op, action.target) for action in actions] == [
        ("grasp", "西瓜霜"),
        ("place", "pink tray"),
        ("grasp", "抗病毒口服液"),
        ("place", "pink tray"),
    ]


def test_explicit_multi_target_parser_keeps_unknown_medicine_names() -> None:
    queries = PlanningAgent._extract_medicine_queries(
        "把西瓜霜、阿莫西林放入蓝色托盘",
        "",
    )

    assert queries == ["西瓜霜", "阿莫西林"]


def test_single_medicine_task_remains_single_pick_place_pair() -> None:
    graph = PlanningAgent(context=None, use_llm=False).plan(
        task_goal="将抗病毒口服液放到粉色盘子中"
    )
    actions = SimpleCodeGenerator().build_actions(graph)

    assert [(action.op, action.target) for action in actions] == [
        ("grasp", "抗病毒口服液"),
        ("place", "pink tray"),
    ]


def test_resume_uses_action_index_for_second_place_to_same_tray() -> None:
    actions = [
        grasp("西瓜霜"),
        place("pink tray"),
        grasp("抗病毒口服液"),
        place("pink tray"),
    ]

    index = SimpleArmCodeAgentSystem._find_resume_index(
        actions=actions,
        failed_action={"op": "place", "target": "pink tray"},
        medicine_query="抗病毒口服液",
        failed_action_index=3,
    )

    assert index == 2
