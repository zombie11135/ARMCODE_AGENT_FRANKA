"""Dependency structure over observed occlusion hypotheses, not hidden geometry."""
from collections import deque


def summarize_structure(node_ids, pairs):
    parents = {key: set() for key in node_ids}
    for pair in pairs:
        parents[pair['target']].add(pair['source'])
    summaries = {}
    for target in sorted(parents):
        distances, queue = {}, deque((p, 1) for p in sorted(parents[target]))
        while queue:
            node, depth = queue.popleft()
            if node in distances:
                continue
            distances[node] = depth
            queue.extend((p, depth + 1) for p in sorted(parents[node]))
        remaining, layers = set(distances), []
        while remaining:
            layer = sorted(n for n in remaining if not (parents[n] & remaining))
            if not layer:
                break
            layers.append(layer)
            remaining.difference_update(layer)
        summaries[target] = dict(direct_blockers=sorted(parents[target]),
            ancestor_ids=sorted(distances), dependency_distance=distances,
            clearing_layers=layers, unresolved_dependency_ids=sorted(remaining),
            has_dependency_conflict=bool(remaining),
            multiple_direct_blockers=len(parents[target]) > 1,
            multi_layer=len(layers) > 1,
            relation_scope='dependency_paths_not_direct_occlusion_or_contact')
    return dict(schema='occlusion_structure_v1', targets=summaries)


def root_priority(node_id, target, pairs):
    """Rank by supported branches leading to target, then pair evidence strength."""
    children = {}
    for pair in pairs:
        children.setdefault(pair['source'], set()).add(pair['target'])
    reached, todo = set(), [node_id]
    while todo:
        node = todo.pop()
        for child in children.get(node, ()):
            if child not in reached:
                reached.add(child);todo.append(child)
    direct = {p['source'] for p in pairs if p['target'] == target}
    coverage = len((reached | {node_id}) & direct) if target is not None else len(reached)
    scores = [float(e.get('evidence_score', e.get('fraction_of_completed_patch', 0.)))
              for p in pairs if p['source'] == node_id for e in p.get('evidence', [])]
    return coverage, max(scores, default=0.)


def vertical_relations(nodes, arrays, base_T_camera, *, min_gap_m=.008, max_gap_m=.20):
    """Describe measured surface height ordering over overlapping XY projections.

    Visible surface overlap does not establish physical support or object mass.
    These relations are descriptive; they do not authorize clearing a blocker.
    """
    import cv2
    import numpy as np
    from itertools import combinations
    measured = []
    for node in nodes:
        points = arrays[node['observed_points_camera_key']]
        if len(points) < 20:
            continue
        base = points @ base_T_camera[:3, :3].T + base_T_camera[:3, 3]
        hull = cv2.convexHull(base[:, :2].astype(np.float32))
        area = cv2.contourArea(hull)
        if area < 1e-6:
            continue
        measured.append((node['node_id'], hull, area, np.quantile(base[:, 2], [.1, .5, .9])))
    edges = []
    for a, b in combinations(measured, 2):
        area, _ = cv2.intersectConvexConvex(a[1], b[1])
        fraction = float(area / min(a[2], b[2]))
        if fraction < .15:
            continue
        upper, lower = (a, b) if a[3][1] > b[3][1] else (b, a)
        gap = float(upper[3][0] - lower[3][2])
        if min_gap_m <= gap <= max_gap_m:
            edges.append(dict(source=upper[0], target=lower[0], relation='possible_stack_above',
                observed_surface_gap_m=gap, projected_xy_overlap=fraction,
                coordinate_frame='fr3_link0', confirmed=False,
                scope='measured_surfaces_not_support_contact', calibrated_probability=None))
    return edges
