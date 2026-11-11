"""Geometric pair proposals, bidirectional validation, and short-lived group IDs.

No medicine identity is inferred. A group consists only of current visible
pixels; its individual masks are retained. Memory never fills missing pixels.
"""
from itertools import combinations, permutations
import time
import uuid

import cv2
import numpy as np
from occlusion_reasoning.planar import _planes, PlanarConfig
from runtime.observation_memory import pose_distance


def iou(a, b):
    return float(np.count_nonzero(a & b)/max(1,np.count_nonzero(a | b)))


def validate_directions(masks, negatives, prediction):
    a, b = masks
    fwd, rev = prediction['masks']
    coverage = [[float(np.count_nonzero(p & m)/max(1,m.sum())) for m in masks] for p in (fwd,rev)]
    contamination = [float(np.count_nonzero(p & negatives)/max(1,p.sum())) for p in (fwd,rev)]
    extra = [float(np.count_nonzero(p & ~(a | b))/max(1,p.sum())) for p in (fwd,rev)]
    agreement = iou(fwd,rev)
    accepted = (min(coverage[0][0],coverage[1][1]) >= .75
                and min(coverage[0][1],coverage[1][0]) >= .7
                and agreement >= .65 and max(contamination) <= .05 and max(extra) <= .25)
    return dict(accepted=bool(accepted), direction_coverage=coverage,
                blocker_contamination=contamination, outside_visible_fragments=extra,
                direction_iou=agreement)


class FragmentAssociationMemory:
    def __init__(self, max_age_s=30., max_entries=16):
        self.max_age_s, self.max_entries = max_age_s, max_entries
        self.entries = []

    def prepare(self, context):
        if not context or context.get('pose') is None or context.get('scope') is None or context.get('scene_epoch') is None:
            self.entries.clear();return
        now = context['timestamp']
        self.entries = [e for e in self.entries if e['context']['scope'] == context['scope']
                        and e['context']['scene_epoch'] == context['scene_epoch']
                        and 0 <= now-e['context']['timestamp'] <= self.max_age_s
                        and pose_distance(e['context']['pose'],context['pose'])[0] <= .005
                        and pose_distance(e['context']['pose'],context['pose'])[1] <= .035]

    def find(self, masks, depth, image):
        for entry in list(self.entries):
            if entry['masks'][0].shape != depth.shape:
                continue
            for order in permutations((0,1)):
                if min(iou(masks[i],entry['masks'][j]) for i,j in enumerate(order)) < .6:
                    continue
                compatible = True
                for i,j in enumerate(order):
                    valid = masks[i] & np.isfinite(depth) & (depth > 0)
                    mean_color = image[masks[i]].mean(axis=0)
                    if (valid.sum() < 40 or abs(float(np.median(depth[valid]))-entry['depths'][j]) > .008
                            or np.linalg.norm(mean_color-entry['colors'][j]) > 35.):
                        compatible = False;break
                if compatible:
                    return dict(object_id=entry['object_id'], confirmed_at=entry['context']['timestamp'],
                                confirmation_frame=entry['context'].get('frame_id'))
                self.entries = [e for e in self.entries if e is not entry]
                break
        return None

    def remember(self, object_id, masks, depth, image, context):
        if not context or context.get('pose') is None or context.get('scope') is None or context.get('scene_epoch') is None:
            return
        self.entries.append(dict(object_id=object_id, masks=[m.copy() for m in masks],
            depths=[float(np.median(depth[m & np.isfinite(depth) & (depth>0)])) for m in masks],
            colors=[image[m].mean(axis=0) for m in masks], context=dict(context)))
        self.entries = self.entries[-self.max_entries:]


def geometric_pairs(rows, depth, intrinsic, supporting_proposals=()):
    y,x=np.indices(depth.shape)
    cx=intrinsic['ppx'] if 'ppx' in intrinsic else intrinsic['cx']
    cy=intrinsic['ppy'] if 'ppy' in intrinsic else intrinsic['cy']
    rays=np.stack(((x-cx)/intrinsic['fx'],(y-cy)/intrinsic['fy'],np.ones(depth.shape)),axis=-1)
    valid=np.isfinite(depth)&(depth>0);points=rays*depth[...,None]
    config=PlanarConfig(max_planes=1,plane_tolerance_m=.004,min_plane_points=40)
    candidates=[]
    for ia,ib in combinations(range(len(rows)),2):
        a,b=rows[ia]['mask'],rows[ib]['mask'];combined=a|b
        if np.count_nonzero(a&b)/max(1,min(a.sum(),b.sum())) > .05:
            continue
        # Only nearby fragments with an occupied gap are candidates.
        yy,xx=np.nonzero(combined)
        if not len(xx) or (xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)>4*combined.sum():
            continue
        if min(np.count_nonzero(m&valid) for m in (a,b)) < 40:
            continue
        hull=np.zeros(depth.shape,np.uint8)
        cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack((xx,yy)).astype(np.int32)),1)
        gap=hull.astype(bool)&~combined
        others=[r['mask'] for i,r in enumerate(rows) if i not in (ia,ib)]
        occupied=np.logical_or.reduce(others) if others else np.zeros(depth.shape,bool)
        if gap.sum()<40 or np.count_nonzero(gap&occupied)<.5*gap.sum():continue
        surfaces=list(_planes(points[combined&valid],config))
        for proposal in supporting_proposals:
            mask=proposal['mask']
            if all(np.count_nonzero(mask&m)>=.8*m.sum() for m in (a,b)):
                surfaces.extend(_planes(points[mask&valid],config))
        choices=[]
        for normal,offset,_ in surfaces:
            support=[float(np.mean(np.abs(points[m&valid]@normal+offset)<.006)) for m in (a,b)]
            choices.append((min(support),normal,offset,support))
        if not choices:continue
        quality,normal,offset,support=max(choices,key=lambda p:p[0])
        if quality<.6:continue
        denominator=rays@normal
        predicted=np.divide(-offset,denominator,out=np.full(depth.shape,np.nan),where=np.abs(denominator)>1e-8)
        blockers=gap & occupied & valid & (predicted-depth>.008)
        gap_fraction=float(blockers.sum()/max(1,gap.sum()))
        if blockers.sum()<40 or gap_fraction<.5:continue
        candidates.append(dict(indices=(ia,ib),negative_mask=blockers,plane_support=support,
                               occupied_closer_gap_fraction=gap_fraction))
    return sorted(candidates,key=lambda c: (-min(c['plane_support']),-c['occupied_closer_gap_fraction'],c['indices']))


def associate_fragments(rows, depth, intrinsics, image, validator, *, memory=None, context=None, max_pairs=3, supporting_proposals=(), advisor=None):
    started=time.perf_counter()
    if memory is not None:memory.prepare(context)
    candidates=geometric_pairs(rows,depth,intrinsics,supporting_proposals)
    if memory is not None:
        memory.entries = [e for e in memory.entries if any(
            min(iou(rows[c['indices'][i]]['mask'],e['masks'][j]) for i,j in enumerate(order)) >= .6
            for c in candidates for order in permutations((0,1)))]
    consumed=set();groups=[];checks=[];calls=0
    for candidate in candidates:
        ia,ib=candidate['indices']
        if ia in consumed or ib in consumed:continue
        pair=[rows[ia],rows[ib]];masks=[p['mask'] for p in pair]
        cached=memory.find(masks,depth,image) if memory is not None else None
        check={k:v for k,v in candidate.items() if k not in ('indices','negative_mask')}
        check['fragment_source_ids']=[p['source_id'] for p in pair]
        check['fragment_boxes_xyxy']=[p['bbox_xyxy'] for p in pair]
        if cached is not None:
            check.update(accepted=True,source='short_term_memory',**cached)
            check['confirmation_age_s']=context['timestamp']-cached['confirmed_at']
        else:
            if calls >= max_pairs:continue
            prediction=validator(image,masks,candidate['negative_mask']);calls+=1
            check.update(validate_directions(masks,candidate['negative_mask'],prediction))
            check.update({k:v for k,v in prediction.items() if k!='masks'})
            check['source']='sam3_bidirectional'
            if check['accepted']:
                check['object_id']='fragment_'+uuid.uuid4().hex
                if memory is not None:memory.remember(check['object_id'],masks,depth,image,context)
        if not check['accepted'] and advisor is not None:
            check['vlm_advice'] = advisor(masks)
        checks.append(check)
        if not check['accepted']:continue
        consumed.update((ia,ib));visible=masks[0]|masks[1];y,x=np.nonzero(visible)
        object_id=check['object_id']
        groups.append(dict(source_id=object_id,object_id=object_id,mask=visible,
            bbox_xyxy=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],
            confidence=min(p['confidence'] for p in pair),prompt='associated_visible_fragments',
            visible_fragments=[dict(source_id=p['source_id'],object_id=object_id,mask=p['mask']) for p in pair],
            association_source=check['source']))
    result=[r for i,r in enumerate(rows) if i not in consumed]+groups
    result.sort(key=lambda r:(-r['confidence'],r['source_id']))
    return result,dict(schema='bidirectional_fragment_association_v1',candidate_pairs=len(candidates),
        checked_pairs=len(checks),model_pair_calls=calls,max_model_pair_calls=max_pairs,
        checks=checks,groups=len(groups),elapsed_s=time.perf_counter()-started,
        hidden_pixels_filled=False,temporal_memory_is_not_independent_evidence=True)
