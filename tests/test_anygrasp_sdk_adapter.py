from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from hardware.anygrasp_sdk import AnyGrasp


def options():
    return NS(checkpoint_path='/test/checkpoint.tar', max_gripper_width=.085,
              gripper_height=.03, top_down_grasp=False)


def test_new_api_keeps_width_before_inference_and_full_scene_collision(monkeypatch):
    group=object()
    model=Mock();model.get_grasp.return_value=group
    sdk=NS(create_detector=Mock(return_value=model))
    monkeypatch.setattr('hardware.anygrasp_sdk.import_module',lambda name:sdk)
    opts=options();detector=AnyGrasp(opts);detector.load_net()
    sdk.create_detector.assert_called_once_with(opts)
    assert sdk.create_detector.call_args.args[0].max_gripper_width==.085
    points=np.ones((20,3),np.float32);colors=np.ones_like(points)
    returned,_=detector.get_grasp(points,colors,lims=[0,2]*3)
    assert returned is group
    assert model.get_grasp.call_args.args[0] is points
    assert model.get_grasp.call_args.args[1]=={'dense_grasp':False,'collision_detection':True}


def test_new_license_failure_is_an_error_not_an_empty_success(monkeypatch):
    monkeypatch.setattr('hardware.anygrasp_sdk.import_module',lambda name:NS(create_detector=lambda cfg:None))
    with pytest.raises(RuntimeError,match='license'):
        AnyGrasp(options()).load_net()


def test_legacy_sdk_remains_usable_without_changing_arguments(monkeypatch):
    model=Mock(); model.get_grasp.return_value=('group','cloud')
    sdk=NS(AnyGrasp=Mock(return_value=model))
    monkeypatch.setattr('hardware.anygrasp_sdk.import_module',lambda name:sdk)
    detector=AnyGrasp(options());detector.load_net()
    model.load_net.assert_called_once_with()
    assert detector.get_grasp('points','colors',lims=[0,1]*3)==('group','cloud')
    model.get_grasp.assert_called_once_with('points','colors',lims=[0,1]*3,
        apply_object_mask=True,dense_grasp=False,collision_detection=True)


def test_new_api_empty_output_has_valid_zero_candidate_shape(monkeypatch):
    # Agent tests do not import the SDK environment's graspnetAPI dependency.
    monkeypatch.setitem(__import__('sys').modules, 'graspnetAPI',
        NS(GraspGroup=lambda:NS(grasp_group_array=np.empty((0,17)))))
    model=Mock();model.get_grasp.return_value=None
    monkeypatch.setattr('hardware.anygrasp_sdk.import_module',lambda name:NS(create_detector=lambda cfg:model))
    detector=AnyGrasp(options());detector.load_net()
    result,_=detector.get_grasp(np.ones((20,3)),None,lims=[0,2]*3)
    assert result.grasp_group_array.shape==(0,17)
