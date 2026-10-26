"""Online adapter for the tested single-view evidence extractor.

Model construction is lazy. Camera ownership, robot execution and scene epochs
remain with the existing agent runtime; fixed-view evidence is never fused.
"""
from pathlib import Path
import hashlib
import json
import shutil
import uuid

import cv2
import numpy as np

from runtime.perception_pipeline import serialized_resource


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_capture(capture):
    return {k: v for k, v in capture.items() if not isinstance(v, np.ndarray)}


def file_binding(capture):
    files = {}
    for key in ('color_path', 'depth_path', 'depth_npy_path', 'points_path'):
        if capture.get(key):
            files[key] = dict(path=str(Path(capture[key]).resolve()), sha256=sha(capture[key]))
    return files


def verify_binding(binding):
    for item in binding.values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Occlusion observation source changed; observe again')


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., x1-x0)*max(0., y1-y0)
    return intersection/max(1., (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def associate_target(nodes, selected):
    """Image-local correspondence, not a new semantic identity prediction."""
    if not selected or not selected.get('bbox'):
        return dict(status='target_not_identified', node_id=None, candidates=[])
    box = np.asarray(selected['bbox'], float)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Target bbox must be a finite XYXY rectangle')
    mask = None
    if selected.get('mask_npy_path'):
        mask = np.load(selected['mask_npy_path'], allow_pickle=False).astype(bool)
    elif selected.get('mask_path'):
        mask = cv2.imread(str(selected['mask_path']), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError('Target identity mask cannot be read')
        mask = mask > 0
    rows = []
    for node in nodes:
        if mask is None:
            score = _iou(box, node['bbox_xyxy'])
        else:
            visible = np.load(node['visible_mask_path'], allow_pickle=False).astype(bool)
            if visible.shape != mask.shape:
                raise ValueError('Target identity mask differs from the observed image shape')
            score = np.count_nonzero(mask & visible)/max(1, np.count_nonzero(mask | visible))
        rows.append(dict(node_id=node['node_id'], score=float(score)))
    rows.sort(key=lambda row: (-row['score'], row['node_id']))
    accepted = bool(rows and rows[0]['score'] >= .35 and
                    (len(rows) == 1 or rows[0]['score']-rows[1]['score'] >= .10))
    return dict(status='associated' if accepted else 'target_association_ambiguous',
                node_id=rows[0]['node_id'] if accepted else None, candidates=rows,
                method='visible_mask_iou' if mask is not None else 'bbox_iou',
                min_iou=.35, min_margin=.10, calibrated_probability=None)


class WristOcclusionService:
    def __init__(self, checkpoint, device='cuda', output_dir='runs/agent_occlusion', backend=None):
        self.checkpoint, self.device = str(checkpoint), device
        self.output_dir = Path(output_dir)
        self.backend = backend

    def freeze(self, capture, camera='wrist'):
        """Make a same-frame, immutable adapter snapshot before identity inference."""
        from occlusion_reasoning.camera_only import save_pair
        color = cv2.imread(str(capture['color_path']))
        raw = (np.asarray(capture['depth']) if capture.get('depth') is not None else
               cv2.imread(str(capture['depth_path']), cv2.IMREAD_UNCHANGED))
        scale = capture.get('depth_unit_m_per_raw_unit')
        if (color is None or raw is None or capture.get('depth_aligned_to_color') is not True
                or scale is None or not np.isfinite(scale) or scale <= 0):
            raise ValueError('Explicit aligned RGB-D and metres-per-raw-unit scale required')
        if raw.dtype != np.uint16:
            raise ValueError('Expected original uint16 depth, with its recorded multiplier')
        root = self.output_dir/uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=False)
        frame = dict(capture, color=color, depth=raw, depth_unit_m_per_raw_unit=float(scale))
        save_pair({camera: frame}, root, 0., .10)
        frozen = json_capture(capture)
        frozen.update(color_path=str((root/camera/'color.png').resolve()),
                      depth_path=str((root/camera/'depth.png').resolve()),
                      depth_npy_path=str((root/camera/'depth_m.npy').resolve()),
                      camera_name=camera, depth_scale=float(scale))
        if capture.get('points_path'):
            points = root/camera/'points.npy'
            shutil.copyfile(capture['points_path'], points)
            frozen['points_path'] = str(points.resolve())
        return frozen, root

    def analyze(self, capture, root, selected=None):
        from perception_v2.proposal_refinement import (
            DEFAULT_OCCLUSION_PROMPTS, refine_proposals, select_visible_hypotheses)
        from occlusion_reasoning.real_single_view import save_view
        from occlusion_reasoning.real_primary_aux import _view_summary
        self._ensure_backend()
        camera, root = capture['camera_name'], Path(root)
        color = cv2.imread(capture['color_path'])
        candidates, backend, raw = refine_proposals(color, DEFAULT_OCCLUSION_PROMPTS, self.backend)
        chosen, selection = select_visible_hypotheses(candidates)
        backend['visible_hypothesis_selection'] = selection
        directory = root/'evidence'
        save_view(root/camera, chosen, backend, directory, raw_proposals=raw)
        view = _view_summary(directory, 'primary' if camera == 'wrist' else 'auxiliary')
        with np.load(directory/'observation/observation_arrays.npz', allow_pickle=False) as arrays:
            mask_dir = directory/'masks'
            mask_dir.mkdir()
            for index, node in enumerate(view['relation_graph']['nodes']):
                path = mask_dir/f'instance_{index:03d}.npy'
                np.save(path, arrays[node['raw_mask_key']].astype(bool))
                node['visible_mask_path'] = str(path.resolve())
                node['visible_mask_sha256'] = sha(path)
        association = associate_target(view['relation_graph']['nodes'], selected) if camera == 'wrist' else None
        result = dict(schema='agent_wrist_occlusion_v1', camera_id=camera,
                      primary=view, target_association=association,
                      source_binding=file_binding(capture),
                      observation_provenance=capture.get('observation_cache'),
                      capture_geometry=capture.get('capture_geometry'), timestamp=capture['timestamp'],
                      cross_view_geometry_used=False, confirmed=False,
                      report_path=str((root/'report.json').resolve()))
        Path(result['report_path']).write_text(json.dumps(result, indent=2, allow_nan=False))
        result['report_sha256'] = sha(result['report_path'])
        return result

    @serialized_resource('gpu')
    def _ensure_backend(self):
        if self.backend is None:
            from perception_v2.sam3_proposals import SAM3Proposals
            self.backend = SAM3Proposals(self.checkpoint, device=self.device)


def blocking_quality_issues(node):
    """Require usable input, leaving visible-surface graspability to the planner.

    Coverage, image-border truncation and mask-overlap warnings are diagnostic,
    including in historical reports. Do not turn every quality annotation into
    an execution veto or change the original, file-bound evidence.
    """
    issues = [issue for issue in node.get('quality_issues', [])
              if issue in ('empty_mask', 'no_depth_support')]
    quality = node['quality']
    if quality.get('empty_mask') or quality.get('raw_mask_pixels') == 0:
        issues.append('empty_mask')
    surface = node.get('observed_surface_camera') or {}
    if quality.get('depth_supported_pixels') == 0 or surface.get('point_count') == 0:
        issues.append('no_depth_support')
    return list(dict.fromkeys(issues))


def choose_blocker(report, *, allow_search=False, max_image_fraction=.5, min_depth_fraction=None,
                   preferred_node_id=None):
    """Choose an exposed graph ancestor of the target, never an unrelated pair.

    With explicit search enabled, an unlocalized target may use an exposed root
    of an observed relation as a bounded exploratory action, not proof of hiding.
    min_depth_fraction is a retired compatibility keyword and has no effect.
    """
    view = report['primary']
    nodes = {n['node_id']: n for n in view['relation_graph']['nodes']}
    pairs = view['possible_pairs']
    target = report['target_association']['node_id']
    if report['target_association']['status'] == 'target_association_ambiguous':
        return dict(status='target_association_ambiguous', blocker=None)
    predecessors = {key: set() for key in nodes}
    for pair in pairs:
        predecessors[pair['target']].add(pair['source'])
    exploratory = target is None
    if exploratory and not allow_search:
        return dict(status=report['target_association']['status'], blocker=None)
    if target is not None and not predecessors[target]:
        quality = blocking_quality_issues(nodes[target])
        status = 'target_quality_insufficient' if quality else 'no_occlusion_evidence'
        return dict(status=status, blocker=None, target_node_id=target)
    ancestors = set()
    pending = list({p['source'] for p in pairs} if exploratory else predecessors[target])
    while pending:
        node = pending.pop()
        if node in ancestors:
            continue
        ancestors.add(node)
        pending.extend(predecessors[node])
    # A cycle in the relevant subgraph is unresolved evidence, even if a
    # separate root points into that cycle. Do not clear around the ambiguity.
    visited, active = set(), set()
    def cyclic(node):
        if node in active:
            return True
        if node in visited:
            return False
        active.add(node)
        if any(cyclic(parent) for parent in predecessors[node]):
            return True
        active.remove(node)
        visited.add(node)
        return False
    if any(cyclic(node) for node in ancestors):
        return dict(status='ambiguous_or_cyclic_relations', blocker=None)
    roots = [nodes[key] for key in ancestors if not predecessors[key] and key != target]
    if not roots:
        return dict(status='ambiguous_or_cyclic_relations' if ancestors else 'no_search_evidence', blocker=None)
    eligible = []
    for node in roots:
        q = node['quality']
        image_area = view['frame']['width']*view['frame']['height']
        if (blocking_quality_issues(node)
                or q['raw_mask_pixels']/image_area > max_image_fraction):
            continue
        eligible.append(node)
    if not eligible:
        return dict(status='blocker_quality_insufficient', blocker=None)
    eligible.sort(key=lambda n: (-n['detection_score'], n['node_id']))
    if preferred_node_id is not None:
        if preferred_node_id not in {n['node_id'] for n in eligible}:
            raise ValueError('Preferred blocker is not an eligible exposed ancestor/search candidate')
        eligible.sort(key=lambda n: n['node_id'] != preferred_node_id)
    blocker = eligible[0]
    return dict(status='exploratory_clear_candidate' if exploratory else 'target_occluded',
                blocker=blocker, target_node_id=target, exploratory=exploratory,
                basis='single_view_possible_occlusion_not_grasp_blocking_proof', confirmed=False,
                eligible_node_ids=[n['node_id'] for n in eligible])
