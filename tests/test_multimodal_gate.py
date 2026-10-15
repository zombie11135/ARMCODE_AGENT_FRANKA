"""Contract tests for the raw RGB-D/task multimodal Gate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gate_learning.input_adapter import (
    ACTION_NAMES,
    MULTIMODAL_STATE_FEATURE_NAMES,
    MultimodalGateInput,
    build_policy_state_vector,
    from_sim_policy_observation,
)
from gate_learning.losses import multimodal_gate_loss
from gate_learning.model import HybridGateConfig, HybridMultimodalGate
from gate_learning.policy import MultimodalGatePolicy, save_checkpoint


def test_state_vector_uses_op_and_explicit_validity_without_vlm_scores() -> None:
    vector = build_policy_state_vector(
        robot_state={"holding": "target"},
        memory={"buffer_occupied": True},
        action_mask={"EXECUTE": True, "VISUAL_PROBE": False},
        current_step={"op": "place"},
        task_progress=None,
        previous_action_result=None,
    )
    values = dict(zip(MULTIMODAL_STATE_FEATURE_NAMES, vector.tolist()))
    assert values["previous_action_result_valid"] == 0.0
    assert values["task_progress_valid"] == 0.0
    assert values["is_place"] == 1.0
    assert values["gripper_holding"] == 1.0
    assert all("confidence" not in name and "vlm" not in name for name in values)


def test_sim_adapter_rejects_evaluator_truth() -> None:
    rgb = np.zeros((16, 20, 3), np.uint8)
    depth = np.ones((16, 20), np.float32)
    observation = {
        "policy_observation": {
            "rgb": rgb,
            "depth_m": depth,
            "robot_state": {},
            "memory": {"task_node": "grasp_target"},
        },
        "action_mask": {name: name == "EXECUTE" for name in ACTION_NAMES},
        "evaluator_state": {"target_pose": [0, 0, 0]},
    }
    with pytest.raises(ValueError, match="evaluator"):
        from_sim_policy_observation(observation, task_text="抓取目标")


def test_input_requires_aligned_metric_depth() -> None:
    item = MultimodalGateInput(
        rgb=np.zeros((16, 20, 3), np.uint8),
        depth_m=np.zeros((15, 20), np.float32),
        task_text="抓取西瓜霜",
        state_vector=np.zeros(len(MULTIMODAL_STATE_FEATURE_NAMES), np.float32),
        action_mask=np.ones(4, np.bool_),
    )
    with pytest.raises(ValueError, match="aligned"):
        item.validate()


def test_network_shapes_and_hard_action_mask() -> None:
    torch.manual_seed(3)
    config = HybridGateConfig(
        image_feature_dim=32,
        text_feature_dim=24,
        model_dim=64,
        feedforward_dim=96,
        dropout=0.0,
    )
    model = HybridMultimodalGate(config).eval()
    mask = torch.tensor([[True, False, True, False], [False, True, False, False]])
    with torch.inference_mode():
        outputs = model(
            image_features=torch.randn(2, 5, 32),
            text_features=torch.randn(2, 24),
            depth_m=torch.rand(2, 1, 48, 64),
            state=torch.randn(2, len(MULTIMODAL_STATE_FEATURE_NAMES)),
            action_mask=mask,
            history_features=torch.randn(2, 3, 32),
            history_valid=torch.tensor([[True, True, False], [True, False, False]]),
        )
    assert outputs["action_values"].shape == (2, 4)
    assert outputs["probe_gain"].shape == (2, 2)
    selected = model.select_actions(outputs, mask)
    assert selected[0].item() in {0, 2}
    assert selected[1].item() == 1
    assert torch.all(outputs["masked_action_values"][~mask] < -1e20)


def test_unknown_action_targets_do_not_change_loss() -> None:
    outputs = {"action_values": torch.tensor([[0.1, 0.2, 0.3, 99.0]])}
    targets_a = {
        "action_values": torch.tensor([[0.0, 0.0, 0.0, -10.0]]),
        "action_values_valid": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
    }
    targets_b = {
        "action_values": torch.tensor([[0.0, 0.0, 0.0, 1000.0]]),
        "action_values_valid": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
    }
    assert torch.equal(
        multimodal_gate_loss(outputs, targets_a)["total"],
        multimodal_gate_loss(outputs, targets_b)["total"],
    )


def test_checkpoint_roundtrip_and_policy_mask(tmp_path) -> None:
    class FakeEncoder:
        feature_dim = 16

        def encode_images(self, images):
            rows = []
            for image in images:
                rows.append(torch.full((16,), float(np.asarray(image).mean()) / 255.0))
            return torch.stack(rows)

        def encode_texts(self, texts):
            return torch.stack([torch.full((16,), len(text) / 100.0) for text in texts])

        def metadata(self):
            return {"backend": "fake", "feature_dim": 16, "frozen": True}

    config = HybridGateConfig(
        image_feature_dim=16, text_feature_dim=16, model_dim=32,
        attention_heads=4, feedforward_dim=48, dropout=0.0,
    )
    model = HybridMultimodalGate(config).eval()
    checkpoint = tmp_path / "gate.pt"
    encoder = FakeEncoder()
    save_checkpoint(str(checkpoint), model, encoder_metadata=encoder.metadata())
    policy = MultimodalGatePolicy.from_checkpoint(str(checkpoint), encoder)
    item = MultimodalGateInput(
        rgb=np.zeros((16, 20, 3), np.uint8),
        depth_m=np.ones((16, 20), np.float32),
        task_text="抓取西瓜霜",
        state_vector=np.zeros(len(MULTIMODAL_STATE_FEATURE_NAMES), np.float32),
        action_mask=np.asarray([False, False, True, False]),
    )
    prediction = policy.predict(item)
    assert prediction.action == "MANIPULATION_PROBE"
    assert prediction.action_mask["MANIPULATION_PROBE"] is True
