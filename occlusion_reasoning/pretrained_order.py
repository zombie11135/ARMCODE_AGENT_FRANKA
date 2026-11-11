"""Frozen InstaOrderNet_o adapter; visible masks are never modified.

Official backbone and weights are supplied externally. Preprocessing and swapped
pair averaging follow POSTECH-CVLab/InstaOrder, commit dd88eceea300f60722b763f66335ca0099007aea.
Scores are uncalibrated, and predictions describe occlusion in one camera view.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import cv2
import numpy as np


def crop_padding(image, roi):
    x, y, width, height = roi
    h, w = image.shape[:2]
    result = np.zeros((height, width, *image.shape[2:]), dtype=image.dtype)
    result[max(-y, 0):min(h-y, height), max(-x, 0):min(w-x, width)] = image[
        max(y, 0):min(y+height, h), max(x, 0):min(x+width, w)]
    return result


def prepare_pair(rgb, mask_a, mask_b, box_a, box_b):
    """Official square context crop, RGB normalization, and nearest mask resize."""
    x, y = np.minimum(box_a[:2], box_b[:2])
    right, bottom = np.maximum(box_a[2:], box_b[2:])
    w, h = right-x, bottom-y
    size = max(np.sqrt(w*h*2), w*1.1, h*1.1)
    roi = [int(x+w/2-size/2), int(y+h/2-size/2), int(size), int(size)]
    image = cv2.resize(crop_padding(rgb, roi), (256, 256), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)/255
    image = (image-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
    masks = [cv2.resize(crop_padding(m.astype(np.uint8), roi), (256, 256),
                        interpolation=cv2.INTER_NEAREST).astype(np.float32) for m in (mask_a, mask_b)]
    return np.concatenate([np.stack(masks), image.transpose(2, 0, 1)])


class InstaOrderPredictor:
    def __init__(self, source_dir, checkpoint, *, device='cuda:0', batch_size=8):
        import torch
        self.torch = torch
        self.device, self.batch_size = device, batch_size
        spec = importlib.util.spec_from_file_location(
            'instaorder_official_resnet', Path(source_dir)/'models/backbone/resnet_cls.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.resnet50_cls(in_channels=5, num_classes=2)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
        self.model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        self.model.to(device).eval()

    def predict(self, rgb, nodes, masks, pairs):
        """Return both directed scores; accepts an empty list without inference."""
        torch = self.torch
        by_id = {n['node_id']: n for n in nodes}
        result = []
        with torch.inference_mode():
            for offset in range(0, len(pairs), self.batch_size):
                chunk = pairs[offset:offset+self.batch_size]
                inputs = [prepare_pair(rgb, masks[p['a']], masks[p['b']],
                                       by_id[p['a']]['bbox_xyxy'], by_id[p['b']]['bbox_xyxy']) for p in chunk]
                batch = torch.from_numpy(np.stack(inputs)).to(self.device)
                forward = self.model(batch).sigmoid()
                reverse = self.model(batch[:, [1, 0, 2, 3, 4]]).sigmoid()
                scores = torch.stack(((forward[:, 1]+reverse[:, 0])/2,
                                      (forward[:, 0]+reverse[:, 1])/2), dim=1).cpu().numpy()
                for pair, score in zip(chunk, scores):
                    result.append(dict(pair_id=pair['pair_id'], a=pair['a'], b=pair['b'],
                                       a_over_b=float(score[0]), b_over_a=float(score[1])))
        return result


def exclusion_reason(pair, config):
    """Reuse recorded geometry limits; no new scene-specific distance thresholds."""
    if pair['mask_overlap']['fraction_of_smaller_mask'] > config['max_mask_overlap_fraction']:
        return 'mask_assignment_conflict'
    if pair['boundary']['near_pair_count'] == 0:
        return pair['status']
    return None


def supplement_relations(boundary, baseline_edges, scores, *, threshold=.5):
    """Add learned hypotheses at valid boundaries, keeping geometric provenance.

    A resolved opposing depth order or opposing baseline edge stays a conflict;
    two model directions are recorded for review rather than choosing an argmax.
    Learned additions also require visible boundary contact. A segmentation gap
    defers the candidate; it does not prove that the objects do not occlude.
    """
    result = deepcopy(boundary)
    result['schema'] = 'pretrained_occlusion_relations_v1'
    result['model'] = dict(name='InstaOrderNet_o', threshold=threshold,
                           score_kind='uncalibrated_sigmoid', masks_modified=False,
                           boundary_support='unblocked_contact_required_for_learned_additions')
    result['relation_graph']['edges'] = deepcopy(baseline_edges)
    by_score = {p['pair_id']: p for p in scores}
    existing = {(e['source'], e['target']) for e in baseline_edges if e['relation'] == 'possible_occludes'}
    result['model_decisions'] = []
    result['deferred_candidates'] = []
    for pair in result['pairs']:
        pair['edges'] = [deepcopy(e) for e in baseline_edges
                         if {e['source'], e['target']} == {pair['a'], pair['b']}]
        score = by_score.get(pair['pair_id'])
        if score is None:
            continue
        reason = exclusion_reason(pair, boundary['config'])
        if reason is None and pair['boundary']['contact_fraction'] == 0:
            reason = 'no_visible_boundary_contact'
        selected = []
        if score['a_over_b'] > threshold:
            selected.append((pair['a'], pair['b'], score['a_over_b']))
        if score['b_over_a'] > threshold:
            selected.append((pair['b'], pair['a'], score['b_over_a']))
        if len(selected) == 2 and reason is None:
            reason = 'bidirectional_model_prediction'
        for source, target, value in selected:
            rejection = reason
            depth_front = {'a_closer': pair['a'], 'b_closer': pair['b']}.get(pair['depth']['order'])
            if rejection is None and depth_front is not None and source != depth_front:
                rejection = 'opposing_resolved_depth_order'
            if rejection is None and (target, source) in existing:
                rejection = 'opposing_baseline_edge'
            decision = dict(source=source, target=target, pair_id=pair['pair_id'], model_score=value,
                            accepted=rejection is None, rejection_reason=rejection)
            result['model_decisions'].append(decision)
            if rejection is not None:
                if rejection == 'no_visible_boundary_contact':
                    result['deferred_candidates'].append({
                        **decision, 'status': 'needs_boundary_verification',
                        'contact_distance_px': boundary['config']['contact_distance_px'],
                        'boundary': deepcopy(pair['boundary']),
                        'interpretation': 'insufficient_boundary_support_not_proven_absence',
                    })
                continue
            if (source, target) in existing:
                decision['effect'] = 'supports_baseline'
                continue
            edge = dict(source=source, target=target, pair_id=pair['pair_id'], relation='possible_occludes',
                        scope='current_camera', evidence_source='InstaOrderNet_o', model_score=value,
                        score_kind='uncalibrated_sigmoid', calibrated_probability=None, confirmed=False)
            result['relation_graph']['edges'].append(edge)
            pair['edges'].append(edge)
            decision['effect'] = 'adds_candidate'
    return result
