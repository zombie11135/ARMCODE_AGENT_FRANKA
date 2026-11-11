"""Ground a proposed information action in saved scene evidence, without devices."""


def build_information_plan(snapshot, selected):
    action = selected['action']
    if action not in ('plan_clear', 'plan_observe'):
        raise ValueError('Expected a planning-only information action')
    prerequisites = []
    if action == 'plan_clear':
        node = selected['node_id']
        candidate = next(c for c in snapshot['information_candidates'] if c['node_id'] == node)
        slots = [s for s in snapshot['buffer_slots'] if not s['occupied']]
        prerequisites.extend(snapshot['clear_unavailable_reasons'])
        prerequisites.extend(['current_scene_binding', 'collision_free_grasp_and_transport'])
        first = dict(action='clear_visible_instance_to_buffer', skill='clear_occlusion_to_buffer', node_id=node,
            target_identity_required=False, exploratory=candidate['exploratory'],
            expected_visible_regions=candidate['directly_reveals'],
            reveal_branch_count=candidate['reveal_branch_count'],
            buffer_slot=slots[0]['index'] if slots else None,
            buffer_pose_mmrad=slots[0]['pose_mmrad'] if slots else None)
    else:
        first = dict(action='observe_from_configured_view', view_id=selected['view_id'])
        prerequisites.extend(['current_scene_binding', 'collision_free_view_motion'])
    return dict(schema='information_action_plan_v1', status='planned_not_executed',
        goal=snapshot['goal'], target=snapshot['target'], reason=selected['reason'],
        source_report=snapshot['graph_report'], source_sha256=snapshot['graph_sha256'],
        snapshot_id=snapshot['snapshot_id'], prerequisites=prerequisites,
        information_before=snapshot.get('information_state'),
        expected_information=next((a for a in snapshot.get('action_information_opportunities', [])
                                   if a['action_id'] == selected.get('action_id')), None),
        evaluation='compare_fresh_observation_not_action_completion',
        steps=[first, dict(action='capture_and_rebuild_scene', reuse_old_geometry=False),
               dict(action='reassess_target',
                    if_identified='evaluate_occlusion_then_plan_target_grasp',
                    if_unknown='choose_next_information_action_with_remaining_budget')],
        clear_remaining=snapshot['clear_remaining'], observation_remaining=snapshot['observation_remaining'],
        target_grasp_requires_identity=True,
        outcome_verified=False,
        note='Expected visibility is a geometry heuristic, not proof of target location. Replan after each physical change.')
