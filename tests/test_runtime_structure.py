"""Compatibility and lifecycle checks for the shared runtime assembly.

Backends are replaced at their import boundary; no models or devices are opened.
"""
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from core.actions import SimpleAction, grasp, observe, place
from runtime.camera import TaskCachedCamera
from runtime.config import SimpleAgentSystemConfig
from skill_lib.config import SimpleRobotConfig


def fake_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


def test_empty_action_sequence_is_rejected_before_context_or_hardware(monkeypatch):
    from agents.simple_execution_runtime import SimpleExecutionRuntime

    geometry = Mock()
    fake_module(monkeypatch, "hardware.runtime", require_task_geometry=geometry)
    runtime = SimpleExecutionRuntime(SimpleAgentSystemConfig(goal="offline", execute_robot=True))
    runtime._build_execution_context = Mock(side_effect=AssertionError("must not start services"))
    with pytest.raises(ValueError, match="没有可执行动作"):
        runtime.run([])
    geometry.assert_not_called()
    runtime._build_execution_context.assert_not_called()


def test_shared_definitions_import_without_agent_model_or_device_backends():
    # A fresh interpreter prevents previously cached imports from hiding coupling.
    code = """
import sys
class BlockBackends:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'agents', 'services', 'hardware', 'torch', 'cv2', 'serial', 'pyrealsense2'
        }:
            raise AssertionError('Unexpected backend import: ' + fullname)
sys.meta_path.insert(0, BlockBackends())
from core.actions import grasp
from runtime.config import SimpleAgentSystemConfig
from runtime.camera import TaskCachedCamera
from runtime.services import build_camera, build_execution_context
from skill_lib.config import SimpleRobotConfig
assert grasp('box').op == 'grasp'
assert not SimpleAgentSystemConfig(goal='offline').execute_robot
assert not SimpleRobotConfig().execute
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_historical_imports_keep_the_same_public_objects():
    from app import cli
    from skill_lab import run_agent_system
    from agents import simple_agent_system, simple_execution_runtime
    from skill_lib import simple_actions

    # Sharing the module also preserves patches of helpers used by CLI functions.
    assert run_agent_system is cli
    assert simple_agent_system.SimpleAgentSystemConfig is SimpleAgentSystemConfig
    assert simple_agent_system.TaskCachedCamera is TaskCachedCamera
    assert simple_execution_runtime.TaskCachedCamera is TaskCachedCamera
    assert simple_actions.SimpleRobotConfig is SimpleRobotConfig
    assert simple_actions.SimpleAction is SimpleAction
    assert (simple_actions.grasp, simple_actions.observe, simple_actions.place) == (
        grasp, observe, place,
    )


def test_task_camera_cache_can_fall_back_to_live_capture():
    camera = Mock()
    cached = TaskCachedCamera(camera)
    snapshot = {"color_path": "saved/wrist.png"}
    cached.set_task_captures(wrist_capture=snapshot)
    snapshot["color_path"] = "replaced.png"
    cached.enable_cache()

    capture = cached.capture("wrist")
    assert capture["color_path"] == "saved/wrist.png"
    capture["color_path"] = "changed_by_caller.png"
    assert cached.capture_wrist()["color_path"] == "saved/wrist.png"
    camera.capture_wrist.assert_not_called()

    # Missing snapshots and disabled caching must still reach the live service.
    assert cached.capture_global() is camera.capture_global.return_value
    cached.disable_cache()
    assert cached.capture_wrist() is camera.capture_wrist.return_value
    assert cached.capture("other") is camera.capture.return_value
    camera.capture.assert_called_once_with("other")


@pytest.fixture
def backends(monkeypatch):
    from services import runtime_service_cache

    monkeypatch.setattr(runtime_service_cache, "_SERVICE_CACHE", {})
    camera = Mock()
    camera_factory = Mock(return_value=camera)
    detector_factory = Mock(return_value=object())
    grasp_service, segmenter, references = object(), object(), object()
    fake_module(monkeypatch, "services.realsense_dual_camera_service",
                CameraConfig=SimpleNamespace, RealSenseDualCameraService=camera_factory)
    fake_module(monkeypatch, "services.grounding_dino_detector_service",
                GroundingDINODetectorService=detector_factory)
    fake_module(monkeypatch, "services.grasp_service",
                detector_model_path=lambda config: config.grounding_dino_model,
                build_grasp_service=lambda config: grasp_service,
                build_segmenter=lambda config: segmenter)
    fake_module(monkeypatch, "services.object_reference_service",
                build_reference_service=lambda config: references)
    # The shared factory must leave the hardware profile to the hardware adapter.
    profile_camera = Mock(return_value=(camera, {"profile": "test-fr3.yaml"}))
    fake_module(monkeypatch, "hardware.runtime", camera_service=profile_camera)
    return SimpleNamespace(
        camera=camera, camera_factory=camera_factory, detector_factory=detector_factory,
        grasp_service=grasp_service, segmenter=segmenter, references=references,
        profile_camera=profile_camera,
    )


@pytest.mark.parametrize("profile", [None, "test-fr3.yaml"])
def test_both_callers_share_services_without_starting_devices(backends, profile):
    from agents.simple_agent_system import SimpleArmCodeAgentSystem
    from agents.simple_execution_runtime import SimpleExecutionRuntime

    config = SimpleAgentSystemConfig(goal="offline", hardware_config_path=profile)
    callers = [SimpleArmCodeAgentSystem(config), SimpleExecutionRuntime(config)]
    contexts = []
    for caller in callers:
        caller._event_callback = Mock()
        context, camera = caller._build_execution_context()
        assert camera is backends.camera
        assert context.camera.camera is camera
        assert context.anygrasp is backends.grasp_service
        assert context.segmenter is backends.segmenter
        assert context.object_references is backends.references
        context.emit_text("ready")
        caller._event_callback.assert_called_once_with({"type": "text", "content": "ready"})
        contexts.append(context)

    assert contexts[0].detector is contexts[1].detector
    assert contexts[0].ocr is contexts[1].ocr
    # A task cache is local to its context even when the camera is reused.
    assert contexts[0].camera is not contexts[1].camera
    backends.detector_factory.assert_called_once()
    backends.camera.start.assert_not_called()
    backends.camera.capture_global.assert_not_called()
    backends.camera.capture_wrist.assert_not_called()
    if profile:
        assert backends.profile_camera.call_count == 2
        backends.profile_camera.assert_called_with(profile, config.capture_dir)
        backends.camera_factory.assert_not_called()
    else:
        backends.profile_camera.assert_not_called()
        backends.camera_factory.assert_called_once()
        options = backends.camera_factory.call_args.kwargs
        assert options["global_config"].serial == config.global_serial
        assert options["wrist_config"].serial == config.wrist_serial
        assert not options["global_config"].enable_depth
        assert options["wrist_config"].enable_depth


@pytest.mark.parametrize("use_vdm", [False, True])
def test_planner_only_starts_camera_for_vdm(monkeypatch, backends, use_vdm):
    from agents.simple_agent_system import SimpleArmCodeAgentSystem

    client_factory = Mock(side_effect=lambda **kwargs: Mock(settings=kwargs))
    fake_module(monkeypatch, "services.openai_chatgpt_vlm_client",
                OpenAIChatGPTVLMClient=client_factory)
    config = SimpleAgentSystemConfig(
        goal="offline", use_vdm=use_vdm, use_llm_planner=True,
        planner_llm_base_url="http://planner.invalid", planner_llm_model="planner",
        fallback_vlm_base_url="http://fallback.invalid", fallback_vlm_model="fallback",
        target_name_bindings={"box": "object-1"},
    )
    system = SimpleArmCodeAgentSystem(config)
    system._build_camera = Mock(return_value=(backends.camera, {}))
    context = system._build_planner_context()
    assert context.object_references is backends.references
    assert context.vlm.settings["model_name"] == "planner"
    assert context.fallback_vlm.settings["model_name"] == "fallback"
    context.vlm.chat.assert_not_called()
    context.fallback_vlm.chat.assert_not_called()
    context.target_name_bindings["box"] = "changed"
    assert config.target_name_bindings["box"] == "object-1"
    if use_vdm:
        assert context.camera is backends.camera
        backends.camera.start.assert_called_once()
    else:
        system._build_camera.assert_not_called()
        backends.camera.start.assert_not_called()
        assert context.camera is None
