"""Full-image/overlapping-tile proposals without changing visible geometry.

Prompt/crop scores remain uncalibrated. Repeated predictions are not independent
observations: select one measured-image mask per duplicate hypothesis, never
union masks or increase confidence because of repeated prompts/crops.
"""

from dataclasses import asdict, dataclass
import time

import numpy as np

DEFAULT_OCCLUSION_PROMPTS = ("box", "bottle", "cylindrical container", "plate", "cup")


@dataclass(frozen=True)
class RefinementConfig:
    tile_fraction: float = .65
    duplicate_iou: float = .85
    internal_edge_margin_px: int = 2
    minimum_mask_pixels: int = 32

    def __post_init__(self):
        if not np.isfinite(self.tile_fraction) or not .5 < self.tile_fraction < 1:
            raise ValueError("tile_fraction must be in (.5, 1)")
        if not np.isfinite(self.duplicate_iou) or not .5 < self.duplicate_iou <= 1:
            raise ValueError("duplicate_iou must be in (.5, 1]")
        for name in ("internal_edge_margin_px", "minimum_mask_pixels"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "internal_edge_margin_px" else 1):
                raise ValueError(f"invalid {name}")


def image_windows(width, height, config):
    tw, th = max(1, round(width*config.tile_fraction)), max(1, round(height*config.tile_fraction))
    windows = [(0, 0, width, height)]
    windows.extend((x, y, x+tw, y+th) for y in (0, height-th) for x in (0, width-tw))
    return list(dict.fromkeys(windows))


def remap_proposal(proposal, window, image_shape, margin):
    """Reject artificial crop fragments; preserve real original-image borders."""
    height, width = image_shape[:2]
    x0, y0, x1, y1 = window
    local = np.asarray(proposal["mask"], bool)
    if local.shape != (y1-y0, x1-x0):
        raise ValueError("proposal mask must match its source crop")
    y, x = np.nonzero(local)
    if not len(x):
        return None, "empty_mask"
    artificial_edge = ((x0 > 0 and x.min() <= margin)
                       or (y0 > 0 and y.min() <= margin)
                       or (x1 < width and x.max() >= local.shape[1]-1-margin)
                       or (y1 < height and y.max() >= local.shape[0]-1-margin))
    if artificial_edge:
        return None, "touches_internal_crop_edge"
    mask = np.zeros((height, width), bool)
    mask[y0:y1, x0:x1] = local
    row = {**proposal, "mask": mask, "source_window_xyxy": list(window),
           "model_bbox_in_source_window": proposal["bbox_xyxy"],
           "bbox_xyxy": [int(x.min()+x0), int(y.min()+y0), int(x.max()+x0+1), int(y.max()+y0+1)]}
    return row, "eligible"


def _iou(a, b):
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b)/union if union else 0.


def deduplicate(proposals, threshold):
    """Complete-link grouping prevents transitive merging through partial masks."""
    groups = []
    for row in sorted(proposals, key=lambda r: (-r["confidence"], r["source_id"])):
        group = next((g for g in groups if all(_iou(row["mask"], p["mask"]) >= threshold for p in g)), None)
        if group is None:
            groups.append([row])
        else:
            group.append(row)
    retained = []
    audit = {}
    for index, group in enumerate(groups):
        representative = group[0]
        source_id = f"proposal_{index:03d}"
        retained.append({**representative, "source_id": source_id,
                         "representative_raw_id": representative["source_id"],
                         "duplicate_raw_ids": [p["source_id"] for p in group],
                         "grouping": "mask_overlap_duplicate_hypothesis_not_identity_proof"})
        for p in group:
            audit[p["source_id"]] = {"status": "retained" if p is representative else "duplicate_hypothesis",
                                      "retained_source_id": source_id}
    return retained, audit


def refine_proposals(image, prompts, backend, config=None):
    config = config or RefinementConfig()
    height, width = image.shape[:2]
    eligible, raw, audit, window_reports = [], [], [], []
    started = time.perf_counter()
    for window_index, window in enumerate(image_windows(width, height, config)):
        x0, y0, x1, y1 = window
        local_rows, metadata = backend.predict(image[y0:y1, x0:x1], prompts,
                                               min_mask_pixels=config.minimum_mask_pixels)
        window_reports.append({"window_index": window_index, "window_xyxy": list(window),
                               "proposals_per_prompt": metadata.get("proposals_per_prompt", {}),
                               "backend_inference_seconds": metadata.get("inference_seconds")})
        for local in local_rows:
            raw_id = f"raw_{len(raw):04d}"
            full_mask = np.zeros((height, width), bool)
            local_mask = np.asarray(local["mask"], bool)
            if local_mask.shape != (y1-y0, x1-x0):
                raise ValueError("backend mask does not match crop dimensions")
            full_mask[y0:y1, x0:x1] = local_mask
            raw_row = {**local, "source_id": raw_id, "mask": full_mask,
                       "source_window_xyxy": list(window)}
            raw.append(raw_row)
            mapped, status = remap_proposal(local, window, image.shape, config.internal_edge_margin_px)
            if mapped is not None and mapped["mask"].sum() < config.minimum_mask_pixels:
                status, mapped = "small_mask", None
            audit.append({"raw_id": raw_id, "window_index": window_index,
                          "prompt": local.get("prompt"), "confidence": local["confidence"],
                          "status": status, "mask_key": raw_id})
            if mapped is not None:
                eligible.append({**mapped, "source_id": raw_id})
    retained, grouping_audit = deduplicate(eligible, config.duplicate_iou)
    for row in audit:
        row.update(grouping_audit.get(row["raw_id"], {}))
    report = {"backend": "sam3_shape_prompts_and_overlapping_tiles",
              "confidence_threshold": getattr(backend, "confidence", None),
              "prompts": list(prompts), "config": asdict(config), "window_reports": window_reports,
              "raw_proposals": len(raw), "retained_proposals": len(retained),
              "internal_crop_edge_rejections": sum(a["status"] == "touches_internal_crop_edge" for a in audit),
              "duplicate_proposals": sum(a["status"] == "duplicate_hypothesis" for a in audit),
              "proposal_audit": audit, "inference_seconds": time.perf_counter()-started,
              "checkpoint": str(backend.checkpoint), "ocr_used": False, "target_identity_used": False,
              "coverage": "predicted_candidates_incomplete_even_with_multiscale_refinement",
              "confidence_policy": "representative_score_only_no_repeat_vote_boost_no_mask_union"}
    return retained, report, raw
