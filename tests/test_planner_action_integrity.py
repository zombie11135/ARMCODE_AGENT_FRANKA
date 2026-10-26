"""Regression for an LLM pick node silently disappearing before place."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.plan_graph import PlanGraph
from agents.planning_agent import PlanningAgent
from agents.simple_code_generator import SimpleCodeGenerator
from core.skill_base import SkillResult
from skill_lib.simple_actions import SimpleRobotActions, SimpleRobotConfig


def failed_graph():
    return dict(goal='将泡腾片放到粉色盘子中', root_id='n0', nodes=[
        dict(node_id='n0', label='放置泡腾片', node_type='macro', instruction='完成任务'),
        dict(node_id='n1', label='确认泡腾片药盒', node_type='observe',
             instruction='观察绿色药盒，确认其是否为泡腾片', skill='observe',
             args={'target': '绿色药盒'}, parent='n0'),
        dict(node_id='n2', label='抓取泡腾片药盒', node_type='pick',
             instruction='抓取泡腾片药盒', skill=None, args={}, parent='n0',
             preconditions=['已确认绿色药盒为泡腾片']),
        dict(node_id='n3', label='放到粉色盘子', node_type='place',
             instruction='放置药盒', skill='place', args={'target': 'pink tray'}, parent='n0'),
    ], edges=[dict(source='n0', target='n1'), dict(source='n1', target='n2'),
              dict(source='n2', target='n3')])


def test_llm_pick_node_gets_user_named_target_and_is_not_lost():
    llm = Mock(); llm.chat.return_value = json.dumps(failed_graph(), ensure_ascii=False)
    planner = PlanningAgent(context=SimpleNamespace(vlm=llm, planner_llm=llm), use_llm=True)
    graph = planner.plan('将泡腾片放到粉色盘子中', scene_text='桌上还有抗病毒口服液和蒙脱石散')
    llm.chat.assert_called_once()
    assert graph.metadata['planner'] == 'llm'
    assert graph.metadata['skill_binding_repairs'][0]['node_id'] == 'n2'
    assert [(a.op, a.target) for a in SimpleCodeGenerator().build_actions(graph)] == [
        ('observe', '绿色药盒'), ('grasp', '泡腾片'), ('place', 'pink tray')]


def test_saved_incomplete_graph_is_rejected_instead_of_silently_skipped():
    with pytest.raises(ValueError, match='n2.*未绑定'):
        SimpleCodeGenerator().build_actions(PlanGraph.from_dict(failed_graph()))


def test_macro_only_plan_cannot_generate_an_empty_success_script():
    data = failed_graph()
    data["nodes"] = data["nodes"][:1]
    data["edges"] = []
    with pytest.raises(ValueError, match="没有可执行动作"):
        SimpleCodeGenerator().build_actions(PlanGraph.from_dict(data))


def test_no_new_grasp_is_invented_for_observe_then_place():
    data = failed_graph(); data['nodes'] = [n for n in data['nodes'] if n['node_id'] != 'n2']
    data['edges'] = [dict(source='n0', target='n1'), dict(source='n1', target='n3')]
    graph = PlanningAgent._normalize_simple_skill_bindings(PlanGraph.from_dict(data), data['goal'])
    with pytest.raises(ValueError, match='place.*缺少 grasp'):
        SimpleCodeGenerator().build_actions(graph)


def test_ambiguous_unbound_pick_is_not_filled_from_an_unrelated_goal():
    data = failed_graph(); data['nodes'][2]['label'] = '抓取绿色药盒'
    data['nodes'][2]['instruction'] = '抓取这个药盒'
    graph = PlanningAgent._normalize_simple_skill_bindings(PlanGraph.from_dict(data), data['goal'])
    with pytest.raises(ValueError, match='未绑定'):
        SimpleCodeGenerator().build_actions(graph)


def test_color_only_goal_does_not_become_a_medicine_name():
    data = failed_graph(); data['goal'] = '将绿色药盒放到粉色盘子中'
    data['nodes'][2]['label'] = data['nodes'][2]['instruction'] = '抓取绿色药盒'
    graph = PlanningAgent._normalize_simple_skill_bindings(PlanGraph.from_dict(data), data['goal'])
    assert graph.get_node('n2').skill is None


def test_user_named_target_takes_precedence_over_other_visible_medicines():
    scene = '看到抗病毒口服液、蒙脱石散，另有一个绿色包装'
    assert PlanningAgent._extract_medicine_query('将泡腾片放到粉色盘子中', scene) == '泡腾片'
    assert PlanningAgent._extract_medicine_queries('将泡腾片放到粉色盘子中', scene) == ['泡腾片']
    assert PlanningAgent._extract_medicine_queries('将所有药盒放到粉色盘子中', scene) == ['抗病毒口服液', '蒙脱石散']


@pytest.mark.parametrize('target', ['绿色药盒', '白色蓝色药盒', '泡腾片药品包装盒', 'green medicine box', 'pill box'])
def test_medicine_packaging_uses_medicine_observation(target):
    robot = SimpleRobotActions(context=None)
    robot._observe_medicine_action = Mock(return_value=SkillResult(success=True, data={}))
    robot._observe_container_action = Mock()
    assert robot.observe(target).success
    robot._observe_medicine_action.assert_called_once_with(target)
    robot._observe_container_action.assert_not_called()


@pytest.mark.parametrize('target', ['绿色盒子', '粉色盘子', 'pink tray', '药品收纳盒', 'medicine storage box'])
def test_explicit_destination_still_uses_container_observation(target):
    assert SimpleRobotActions._looks_like_container(target)


def test_empty_gripper_gate_remains_enabled():
    robot = SimpleRobotActions(context=None, config=SimpleRobotConfig(
        nesyro_gate_trace_path='', gate_experience_log_path=''))
    from skill_lib.simple_actions import place
    result = robot.gate_before_action(place('pink tray'), current_holding=None)
    assert not result.success
    assert result.data['failure_type'] == 'nesyro_gate_empty_gripper'


def test_pick_alias_with_explicit_target_is_normalized():
    data = failed_graph(); data['nodes'][2].update(skill='pick', args={'target': '泡腾片'})
    graph = PlanningAgent._normalize_simple_skill_bindings(PlanGraph.from_dict(data), data['goal'])
    assert graph.get_node('n2').skill == 'grasp'
    assert [a.op for a in SimpleCodeGenerator().build_actions(graph)] == ['observe', 'grasp', 'place']
