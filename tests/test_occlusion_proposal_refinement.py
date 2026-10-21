"""Original-pixel integrity and duplicate handling, without models or hardware."""

from pathlib import Path

import numpy as np
import pytest

from perception_v2.proposal_refinement import (
    RefinementConfig, deduplicate, image_windows, refine_proposals, remap_proposal,
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
