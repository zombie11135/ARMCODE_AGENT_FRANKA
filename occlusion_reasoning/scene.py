"""Combine measured multiview evidence and optional box-surface hypotheses."""

from .multiview import extract_multiview
from .planar import extract_planar_hypotheses


def infer_scene(observations, *, multiview_config=None, planar_config=None, planar_prior=True):
    result = extract_multiview(observations, multiview_config)
    result["schema"] = "occlusion_scene_v1"
    result["planar_prior_enabled"] = planar_prior
    result["single_view_planar_evidence"] = []
    if planar_prior:
        for packet, arrays in observations:
            local = extract_planar_hypotheses(packet, arrays, planar_config)
            result["single_view_planar_evidence"].append(local)
            for edge in local["edges"]:
                result["relation_graph"]["edges"].append({
                    **edge,
                    "foreground_node": edge["source"], "reprojected_node": edge["target"],
                    "source": result["node_to_entity"][edge["source"]],
                    "target": result["node_to_entity"][edge["target"]],
                })
    result["limitations"].append(
        "Optional planar continuation assumes convex planar surface patches; its edges remain possible_occludes.")
    return result
