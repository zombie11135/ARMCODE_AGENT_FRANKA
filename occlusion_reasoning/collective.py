"""Combine independent local depth evidence for multi-occluder silhouettes."""
from collections import defaultdict


def add_collective_relations(pairs, config):
    candidates = defaultdict(list)
    for pair in pairs:
        completion = pair.get('completion')
        if (completion and pair['depth']['order'] in ('a_closer', 'b_closer')
                and pair['depth']['valid_pair_count'] >= config.min_valid_pairs
                and pair['depth']['valid_fraction'] >= config.min_valid_fraction
                and pair['boundary']['contact_fraction'] >= config.min_order_fraction
                and completion['front_overlap_pixels'] >= config.min_completion_overlap_pixels):
            candidates[completion['back_instance']].append(pair)
    groups = []
    for target, members in sorted(candidates.items()):
        if len(members) < 2:
            continue
        # Contributions use exclusive instance pixels, hence do not double count.
        pixels = sum(p['completion']['front_overlap_pixels'] for p in members)
        fraction = pixels / max(1, members[0]['completion']['missing_silhouette_pixels'])
        if fraction < config.min_completion_overlap_fraction:
            continue
        sources = sorted(p['a'] if p['b'] == target else p['b'] for p in members)
        group_id = 'collective_' + target
        groups.append(dict(group_id=group_id, target=target, sources=sources,
                           joint_completion_fraction=fraction, confirmed=False))
        for pair in members:
            pair['collective_group_id'] = group_id
            if any(edge['relation'] == 'possible_occludes' for edge in pair['edges']):
                continue
            depth_edge = next(edge for edge in pair['edges'] if edge['relation'] == 'locally_in_front_of')
            pair['status'] = 'possible_occlusion'
            pair['edges'].append(dict(depth_edge, relation='possible_occludes', scope='current_camera',
                evidence='collective_convex_completion', collective_group_id=group_id,
                assumption='farther_object_has_approximately_convex_projected_silhouette'))
    return groups


def local_depth_regions(ya, yb, za, zb, front_hulls, config):
    """Retain spatially distinct depth orders instead of averaging them away."""
    import cv2
    import numpy as np
    valid = np.isfinite(za) & np.isfinite(zb)
    delta = zb - za
    midpoint = np.rint((ya + yb)/2).astype(int)
    shape = front_hulls[0]['mask'].shape
    regions = []
    for direction, selected in [('a_closer', valid & (delta > config.min_depth_delta_m)),
                                ('b_closer', valid & (delta < -config.min_depth_delta_m))]:
        bitmap = np.zeros(shape, np.uint8)
        bitmap[midpoint[selected, 0], midpoint[selected, 1]] = 1
        # Join nearby samples on the same interface, keeping separated regions separate.
        _, labels = cv2.connectedComponents(cv2.dilate(bitmap, np.ones((3, 3), np.uint8)), connectivity=8)
        for label in np.unique(labels[midpoint[selected, 0], midpoint[selected, 1]]):
            if label == 0:
                continue
            take = selected & (labels[midpoint[:, 0], midpoint[:, 1]] == label)
            if take.sum() < config.min_valid_pairs:
                continue
            front, back = front_hulls if direction == 'a_closer' else front_hulls[::-1]
            zone = cv2.dilate((labels == label).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            intrusion = int((zone & front['mask'] & back['hull'] & ~back['mask']).sum()) if back['hull'] is not None else 0
            xy = midpoint[take][:, ::-1]
            regions.append(dict(order=direction, source=front['id'], target=back['id'],
                valid_pair_count=int(take.sum()), depth_gap_m=float(np.median(np.abs(delta[take]))),
                bbox_xyxy=[int(xy[:, 0].min()), int(xy[:, 1].min()), int(xy[:, 0].max()+1), int(xy[:, 1].max()+1)],
                completion_intrusion_pixels=intrusion,
                possible_local_occlusion=intrusion >= config.min_completion_overlap_pixels,
                scope='local_interface_only', confirmed=False))
    return regions
