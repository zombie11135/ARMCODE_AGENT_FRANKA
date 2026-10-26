"""Original-pixel integrity and duplicate handling, without models or hardware."""

from pathlib import Path

import numpy as np
import pytest

from perception_v2.proposal_refinement import (
    DEFAULT_OCCLUSION_PROMPTS, RefinementConfig, deduplicate, image_windows, refine_proposals, remap_proposal,
)


def row(mask, source="raw", confidence=.8):
    y, x = np.nonzero(mask)
    bbox = [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)] if len(x) else [0,0,1,1]
    return dict(source_id=source, mask=mask, bbox_xyxy=bbox, confidence=confidence, prompt="generic")


def test_crop_mapping_preserves_only_original_predicted_pixels():
    mask = np.zeros((60, 60), bool)
    mask[10:30, 15:25] = True
    original = mask.copy()
    mapped, status = remap_proposal(row(mask), (20,20,80,80), (100,100,3), 2)
    assert status == "eligible"
    assert mapped["bbox_xyxy"] == [35,30,45,50]
    assert mapped["mask"].sum() == mask.sum()
    np.testing.assert_array_equal(mask, original)
    np.testing.assert_array_equal(mapped["mask"][20:80,20:80], mask)


def test_internal_crop_edge_is_rejected_but_real_image_border_is_retained():
    mask = np.zeros((60,60), bool)
    mask[:20, :20] = True
    assert remap_proposal(row(mask), (20,20,80,80), (100,100,3), 2)[1] == "touches_internal_crop_edge"
    assert remap_proposal(row(mask), (0,0,60,60), (100,100,3), 2)[1] == "eligible"


def test_duplicate_masks_are_neither_unioned_nor_confidence_boosted():
    a, b = np.zeros((20,20), bool), np.zeros((20,20), bool)
    a[2:12,2:12], b[2:12,3:13] = True, True
    result, audit = deduplicate([row(a,"a",.8),row(b,"b",.9)], .8)
    assert len(result) == 1
    assert result[0]["confidence"] == .9
    np.testing.assert_array_equal(result[0]["mask"], b)
    assert len(result[0]["duplicate_raw_ids"]) == 2
    assert audit["a"]["status"] == "duplicate_hypothesis"


def test_partial_overlap_chain_cannot_merge_two_different_instances():
    masks=[]
    for start in range(3):
        mask=np.zeros((20,20),bool); mask[2:12,start:start+10]=True; masks.append(mask)
    result,_=deduplicate([row(m,str(i),.9-i*.1) for i,m in enumerate(masks)], .8)
    assert len(result)==2


def test_tiles_recover_a_small_omitted_object_and_keep_duplicates_auditable():
    image=np.zeros((100,100,3),np.uint8)
    image[20:40,20:40,0]=1
    image[75:85,75:85,0]=2
    class Backend:
        checkpoint=Path("synthetic-no-weights")
        def predict(self, crop, prompts, min_mask_pixels):
            rows=[]
            for label in (1,2):
                if crop.shape[:2]==(100,100) and label==2:
                    continue
                mask=crop[...,0]==label
                if mask.sum()>=min_mask_pixels:
                    rows.append(row(mask,str(label),.8))
            return rows,{"proposals_per_prompt":{"generic":len(rows)}}
    proposals, report, raw = refine_proposals(image,["generic"],Backend())
    assert len(proposals)==2
    assert len(raw)>len(proposals)
    assert report["duplicate_proposals"]>0
    for label in (1,2):
        assert any(np.array_equal(p["mask"],image[...,0]==label) for p in proposals)
    assert report["target_identity_used"] is False


def test_empty_backend_does_not_invent_object_nodes():
    class Backend:
        checkpoint=Path("none")
        def predict(self,*a,**k): return [],{}
    proposals, report, raw=refine_proposals(np.zeros((80,80,3),np.uint8),["object"],Backend())
    assert proposals==raw==[]
    assert report["retained_proposals"]==0


class EdgeBlockBackend:
    """The block is missed globally and cut off in a standard tile."""
    checkpoint = Path('synthetic-edge-block')

    def __init__(self, recover=True):
        self.recover = recover
        self.calls = []

    def predict(self, crop, prompts, min_mask_pixels):
        self.calls.append((crop.shape[:2], tuple(prompts)))
        x, y = crop[..., 0], crop[..., 1]
        block = (x >= 60) & (x < 78) & (y >= 35) & (y < 50)
        box = (x >= 20) & (x < 90) & (y >= 20) & (y < 80) & ~block
        rows = [dict(row(box, 'box', .95), prompt='box')] if box.any() else []
        standard_fragment = crop.shape[:2] == (65, 65) and x[0, 0] == y[0, 0] == 0
        local_recovery = crop.shape[:2] not in ((100, 100), (65, 65))
        if block.sum() >= min_mask_pixels and (standard_fragment or (local_recovery and self.recover)):
            rows.append(dict(row(block, 'block', .8), prompt='cube' if local_recovery else 'box'))
        return rows, {}


def edge_block_image():
    y, x = np.indices((100, 100))
    return np.dstack((x, y, np.zeros_like(x))).astype(np.uint8)


def test_touching_fragment_is_resegmented_into_a_complete_anonymous_occluder():
    from occlusion_reasoning.observations import InstanceEvidence, build_observation
    from occlusion_reasoning.relations import extract_relations
    from services.wrist_occlusion_service import choose_blocker
    image = edge_block_image()
    backend = EdgeBlockBackend()
    proposals, report, raw = refine_proposals(image, ['box'], backend)
    block = (image[..., 0] >= 60) & (image[..., 0] < 78) & (image[..., 1] >= 35) & (image[..., 1] < 50)
    recovered = next(p for p in proposals if np.array_equal(p['mask'], block))
    assert recovered['recovery_from_raw_id'] is not None
    assert recovered['confidence'] == .8
    assert report['recovery_windows'] == 1 and report['recovered_proposals'] == 1
    assert backend.calls[-1][1] == ('box', 'cube', 'block')
    assert any(a['status'] == 'touches_internal_crop_edge' for a in report['proposal_audit'])
    # The old fragment stays in raw evidence; only a new measured prediction is accepted.
    fragment = next(p for p in raw if p['source_id'] == recovered['recovery_from_raw_id'])
    assert fragment['mask'].sum() < recovered['mask'].sum()
    depth = np.full((100, 100), .7, np.float32)
    instances = []
    for p in proposals:
        depth[p['mask']] = .45 if p is recovered else .5
        instances.append(InstanceEvidence(p['source_id'], p['mask'], p['bbox_xyxy'], p['confidence']))
    packet, arrays = build_observation(depth_m=depth, intrinsics=dict(fx=100., fy=100., cx=50., cy=50.),
        instances=instances, frame_id='edge-block', query='unidentified target', alignment_status='recorded_aligned')
    relations = extract_relations(packet, arrays)
    nodes = packet['relation_graph']['nodes']
    edges = [e for e in relations['relation_graph']['edges'] if e['relation'] == 'possible_occludes']
    decision = choose_blocker(dict(primary=dict(frame=packet['frame'], relation_graph=dict(nodes=nodes),
        possible_pairs=edges), target_association=dict(status='target_not_identified', node_id=None)), allow_search=True)
    assert decision['status'] == 'exploratory_clear_candidate'
    assert decision['blocker']['source_ids'] == [recovered['source_id']]


def test_failed_resegmentation_keeps_fragment_out_of_graph_without_repeated_attempts():
    backend = EdgeBlockBackend(recover=False)
    proposals, report, raw = refine_proposals(edge_block_image(), ['box'], backend)
    assert len(proposals) == 1
    assert report['recovery_windows'] == 2 and report['recovered_proposals'] == 0
    assert len(backend.calls) == 7
    assert not any(p['recovery_from_raw_id'] is not None for p in proposals)


def test_a_fragment_already_covered_by_a_complete_proposal_needs_no_extra_inference():
    class CompleteBackend(EdgeBlockBackend):
        def predict(self, crop, prompts, min_mask_pixels):
            rows, meta = super().predict(crop, prompts, min_mask_pixels)
            if crop.shape[:2] == (100, 100):
                x, y = crop[..., 0], crop[..., 1]
                rows.append(dict(row((x >= 60) & (x < 78) & (y >= 35) & (y < 50)), prompt='box'))
            return rows, meta
    backend = CompleteBackend()
    proposals, report, _ = refine_proposals(edge_block_image(), ['box'], backend)
    assert len(proposals) == 2
    assert report['recovery_windows'] == 0 and len(backend.calls) == 5


def test_default_prompts_find_small_block_even_without_any_generic_box_fragment():
    class CubeBackend(EdgeBlockBackend):
        def predict(self, crop, prompts, min_mask_pixels):
            rows, metadata = super().predict(crop, prompts, min_mask_pixels)
            rows = [p for p in rows if p['source_id'] == 'box']
            x, y = crop[..., 0], crop[..., 1]
            block = (x >= 60) & (x < 78) & (y >= 35) & (y < 50)
            if 'cube' in prompts and block.sum() >= min_mask_pixels:
                rows.append(dict(row(block, 'cube', .8), prompt='cube'))
            return rows, metadata
    image = edge_block_image()
    generic, old, _ = refine_proposals(image, ['box'], CubeBackend())
    assert len(generic) == 1 and old['recovery_windows'] == 0
    proposals, report, _ = refine_proposals(image, DEFAULT_OCCLUSION_PROMPTS, CubeBackend())
    block = (image[..., 0] >= 60) & (image[..., 0] < 78) & (image[..., 1] >= 35) & (image[..., 1] < 50)
    assert any(np.array_equal(p['mask'], block) for p in proposals)
    assert report['recovery_windows'] == 0  # No dependency on finding a fragment first.


def test_tiled_windows_cover_the_image_with_overlap():
    windows=image_windows(640,480,RefinementConfig())
    assert len(windows)==5 and windows[0]==(0,0,640,480)
    cover=np.zeros((480,640),np.uint8)
    for x0,y0,x1,y1 in windows[1:]:cover[y0:y1,x0:x1]+=1
    assert cover.min()>=1 and cover.max()>1


@pytest.mark.parametrize("kwargs",[{"tile_fraction":.4},{"duplicate_iou":float("nan")},
                                   {"internal_edge_margin_px":1.5}])
def test_invalid_refinement_configuration(kwargs):
    with pytest.raises(ValueError): RefinementConfig(**kwargs)
