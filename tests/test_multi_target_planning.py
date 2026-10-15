from agents.planning_agent import PlanningAgent
from agents.simple_agent_system import SimpleArmCodeAgentSystem
from agents.simple_code_generator import SimpleCodeGenerator
from skill_lib.simple_actions import grasp, place


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
