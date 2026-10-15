"""Tests for converting NeSyRo Gate failures into replanning constraints."""

from __future__ import annotations

from agents.execution_state import classify_failure_constraints


def test_nesyro_gate_occlusion_failure_builds_replanning_constraints() -> None:
    report = {
        "failure_type": "nesyro_gate_replan",
        "medicine_query": "布洛芬",
        "failed_conditions": ["target_not_graspable", "target_occluded"],
        "suggested_constraint": "not_graspable(布洛芬) and target_occluded(布洛芬)",
        "error": "low NeSyConf cannot be solved by another probe",
    }

    constraints = classify_failure_constraints(report)
    text = "\n".join(constraints)

    assert "target_occluded(布洛芬)" in text
    assert "禁止直接抓取" in text
    assert "主动观察" in text
    assert "OCR" in text
