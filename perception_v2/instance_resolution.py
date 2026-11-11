"""Compare visible-instance hypotheses using measured RGB-D, without identity labels.

Keep a small, separately predicted foreground surface instead of letting a broad
mask swallow it. Keep visible regions separate by default. Joint-model
reconciliation and supplied bidirectional validators are experimental opt-ins;
no hidden pixels are filled.
"""
import cv2
import numpy as np

from occlusion_reasoning.planar import _planes, PlanarConfig


def resolve_visible_instances(proposals, depth_m, intrinsics, containment_threshold=.8, *, fragment_validator=None, image=None, association_memory=None, association_context=None, fragment_advisor=None, enable_joint_reconciliation=False):
    from .proposal_refinement import select_visible_hypotheses
    depth = np.asarray(depth_m)
    y, x = np.indices(depth.shape)
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx = intrinsics['ppx'] if 'ppx' in intrinsics else intrinsics['cx']
    cy = intrinsics['ppy'] if 'ppy' in intrinsics else intrinsics['cy']
    rays = np.stack(((x-cx)/fx, (y-cy)/fy, np.ones(depth.shape)), axis=-1)
    valid = np.isfinite(depth) & (depth > 0)
    points = rays * depth[..., None]
    rows = [{**p, 'mask': np.asarray(p['mask'], bool).copy()} for p in proposals]
    if any(r['mask'].shape != depth.shape for r in rows):
        raise ValueError('Instance masks and aligned depth must have the same shape')
    config = PlanarConfig(max_planes=1, plane_tolerance_m=.004, min_plane_points=40)

    def plane(mask):
        pts = points[mask & valid]
        fitted = next(_planes(pts, config), None)
        if fitted is None:
            return None
        normal, offset, indices = fitted
        return normal, offset, len(indices)/len(pts)

    def foreground_fraction(mask, surface):
        normal, offset, _ = surface
        selected = mask & valid
        denominator = rays[selected] @ normal
        predicted = np.divide(-offset, denominator, out=np.full(len(denominator), np.nan),
                              where=np.abs(denominator) > 1e-8)
        return float(np.count_nonzero(predicted-depth[selected] > .008)/max(1, selected.sum())) if selected.sum() >= 40 else 0.

    nested = []
    # Work from original model masks for evidence; record the changed visible
    # support separately. No point or depth value is added or altered.
    for small in sorted(rows, key=lambda r: (r['mask'].sum(), -r['confidence'])):
        sm = small['mask']
        if sm.sum() < 40:
            continue
        small_plane = None
        for large in rows:
            lm = large['mask']
            if sm.sum() > .45 * lm.sum() or np.count_nonzero(sm & lm) < .5*sm.sum():
                continue
            if small_plane is None:
                small_plane = plane(sm)
            if small_plane is None or small_plane[2] < .7:
                continue
            back = plane(lm & ~sm)
            if back is None or back[2] < .6:
                continue
            fraction = foreground_fraction(sm, back)
            if fraction < .75:
                continue
            large['mask'] = lm & ~sm
            large.setdefault('visible_mask_subtracted_sources', []).append(small['source_id'])
            nested.append(dict(foreground=small['source_id'], background=large['source_id'],
                foreground_depth_support_fraction=fraction,
                reason='separate_predicted_surface_in_front_of_larger_mask'))

    selected, audit = select_visible_hypotheses(rows, containment_threshold)
    association = None
    if fragment_validator is not None:
        from .fragment_association import associate_fragments
        selected, association = associate_fragments(selected, depth, intrinsics, image,
            fragment_validator, memory=association_memory, context=association_context,
            supporting_proposals=rows, advisor=fragment_advisor)
    replacements = []
    # Prefer a joint *model prediction*, not the union of fragments. Its score
    # stays unchanged. Confidence alone must not force two partial object nodes.
    for whole in sorted(rows if enable_joint_reconciliation else (),
                        key=lambda r: (-r['confidence'], r['source_id'])):
        wm = whole['mask']
        fragments = [s for s in selected if s['source_id'] != whole['source_id']
                     and np.count_nonzero(s['mask'] & wm) >= .8*s['mask'].sum()]
        if len(fragments) < 2 or any(s.get('object_id') for s in fragments):
            continue
        combined = np.logical_or.reduce([s['mask'] for s in fragments])
        if wm.sum() > 1.35*combined.sum() or np.count_nonzero(wm & combined) < .8*wm.sum():
            continue
        surface = plane(wm)
        if surface is None or surface[2] < .6:
            continue
        normal, offset, _ = surface
        support = [float(np.count_nonzero(np.abs(points[s['mask'] & valid] @ normal + offset) < .006)
                         / max(1, np.count_nonzero(s['mask'] & valid))) if np.count_nonzero(s['mask'] & valid) >= 40 else 0. for s in fragments]
        if min(support) < .6:
            continue
        yy, xx = np.nonzero(wm)
        hull = np.zeros(depth.shape, np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.column_stack((xx, yy)).astype(np.int32)), 1)
        gap = hull.astype(bool) & ~wm
        others = [s for s in selected if s['source_id'] not in {f['source_id'] for f in fragments}]
        occupied = np.logical_or.reduce([s['mask'] for s in others]) if others else np.zeros(depth.shape,bool)
        gap_support = foreground_fraction(gap & occupied, surface) if np.any(gap & occupied) else 0.
        occupied_fraction = float(np.count_nonzero(gap & occupied)/max(1,gap.sum()))
        if gap.sum() < 40 or occupied_fraction < .5 or gap_support < .7:
            continue
        ids = {s['source_id'] for s in fragments}
        # A joint prediction must remain compatible with other retained surfaces.
        if any(np.count_nonzero(wm & s['mask'])/max(1,min(wm.sum(),s['mask'].sum())) > .05
               for s in others):
            continue
        selected = [s for s in selected if s['source_id'] not in ids and s['source_id'] != whole['source_id']]
        selected.append(whole)
        replacements.append(dict(retained_source_id=whole['source_id'], fragment_source_ids=sorted(ids),
            fragment_plane_support=support, occupied_gap_fraction=occupied_fraction,
            closer_gap_depth_fraction=gap_support,
            reason='joint_model_mask_with_measured_occluder_between_visible_fragments',
            identity_confirmed=False))
    selected.sort(key=lambda r: (-r['confidence'], r['source_id']))
    audit.update(policy='rgbd_nested_surface_and_joint_instance_hypotheses' if enable_joint_reconciliation
                 else 'rgbd_separate_visible_regions',
        joint_reconciliation_enabled=enable_joint_reconciliation,
        selected_proposals=len(selected), initial_mask_selection=audit['alternatives'],
        alternatives=[a for a in audit['alternatives'] if a['source_id'] not in {s['source_id'] for s in selected}],
        nested_foreground_recoveries=nested, fragment_reconciliations=replacements,
        masks_modified=bool(nested), hidden_pixels_filled=False,
        risk='joint-instance grouping is a hypothesis; touching coplanar objects can remain ambiguous')
    if association is not None:
        for check in association['checks']:
            check['resolved_by_joint_prediction'] = any(
                set(check['fragment_source_ids']) <= set(r['fragment_source_ids']) for r in replacements)
        audit['fragment_associations'] = association
    return selected, audit
