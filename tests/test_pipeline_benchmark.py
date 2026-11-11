import json

import pytest

from runtime.pipeline_benchmark import (
    StubClient, VARIANTS, Workload, aggregate, interval_seconds, overlap_seconds, run_trial,
)


def test_interval_metrics_union_nested_and_concurrent_spans():
    def span(start, duration):
        return dict(start_s=start, duration_s=duration)
    left = [span(0, 4), span(1, 2), span(3, 3), span(10, 2)]
    right = [span(2, 3), span(4, 7)]
    assert interval_seconds(left) == 8
    assert overlap_seconds(left, right) == 5
    assert overlap_seconds([], right) == 0


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('scenario', ['success', 'repair_first', 'stop_first'])
def test_matched_scheduler_replay_preserves_actions_budgets_and_separates_evidence(tmp_path, variant, scenario):
    directory = tmp_path/'run'
    row = run_trial(directory, variant, scenario, Workload(0, 0, 0, 0), StubClient(0), 'stub')
    assert row['correct']
    assert row['retries'] == int(scenario == 'repair_first')
    timing = json.loads((directory/'task_timing.json').read_text())
    assert timing['metadata']['evidence_level'] == 'offline_scheduler'
    assert timing['metadata']['execute_robot'] is False
    perception = json.loads((directory/'perception_pipeline.json').read_text())
    assert perception['status'] == 'published'
    assert perception['candidate_prefetch'] == 'skipped_missing_transforms'
    assert perception['shutdown_wait_s'] >= 0
    if variant in ('serial', 'policy'):
        assert row['perception_overlap_s'] == 0
    if variant in ('serial', 'perception'):
        assert row['policy_overlap_s'] == 0
    with pytest.raises(FileExistsError):
        run_trial(directory, variant, scenario, Workload(), StubClient(0), 'stub')


def test_failure_or_unequal_retry_count_cannot_create_speedup_claim():
    common = dict(scenario='success', repeat=0, correct=True, attempts=['a', 'b'], retries=0,
        perception_overlap_s=0, policy_overlap_s=0, useful_policy_overlap_s=0, perception_join_s=0, policy_join_s=0,
        policy_foreground_s=0, policy_shutdown_s=0, resource_wait_s=0, prefetch_wasted_s=0)
    serial = dict(common, variant='serial', success=True, duration_s=2)
    faster_failure = dict(common, variant='both', success=False, duration_s=1)
    assert aggregate([serial, faster_failure])[-1]['paired_speedup_median'] is None
    extra_retry = dict(common, variant='both', success=True, duration_s=1, retries=1)
    assert aggregate([serial, extra_retry])[-1]['paired_speedup_median'] is None
