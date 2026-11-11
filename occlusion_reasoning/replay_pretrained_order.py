"""Replay frozen InstaOrder on recorded SAM masks; no capture or robot execution.

Example: python -m occlusion_reasoning.replay_pretrained_order \
    --view-dir runs/occlusion_real/<capture>/primary_aux/wrist \
    --output-dir runs/experiments/instaorder_review
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from .extract_relations import write_result
from .pretrained_order import InstaOrderPredictor, exclusion_reason, supplement_relations


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def replay(view_dir, output_dir, predictor, *, audit_all_pairs=False):
    view_dir, output_dir = Path(view_dir).resolve(), Path(output_dir).resolve()
    if output_dir == view_dir or view_dir in output_dir.parents:
        raise ValueError('Store experimental output outside the input view directory')
    output_dir.mkdir(parents=True, exist_ok=False)
    observation_path = view_dir/'observation/scene_observation.json'
    packet = read(observation_path)
    if (view_dir/'view.json').is_file():
        view = read(view_dir/'view.json')
    else:
        view = read(view_dir.parent/'report.json')['views'][packet['frame']['camera_id']]
    boundary = read(view_dir/'boundary/relations.json')
    rgb_path = Path(packet['provenance']['snapshot_dir'])/'color.png'
    if hashlib.sha256(rgb_path.read_bytes()).hexdigest() != packet['provenance']['rgb_sha256']:
        raise ValueError('RGB differs from the recorded observation')
    color = cv2.imread(str(rgb_path))
    with np.load(view_dir/'observation'/packet['artifacts']['arrays'], allow_pickle=False) as stored:
        arrays = dict(stored)
    nodes = packet['relation_graph']['nodes']
    masks = {n['node_id']: arrays[n['raw_mask_key']].astype(bool) for n in nodes}
    eligible = [p for p in boundary['pairs'] if exclusion_reason(p, boundary['config']) is None]
    pairs = boundary['pairs'] if audit_all_pairs else eligible
    # One real pair primes device kernels; separately reported, never included in warm latency.
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    predictor.predict(rgb, nodes, masks, pairs[:1])
    warmup_s = time.perf_counter()-start
    start = time.perf_counter()
    scores = predictor.predict(rgb, nodes, masks, pairs)
    inference_s = time.perf_counter()-start
    result = supplement_relations(boundary, view['relation_graph']['edges'], scores)
    result['input'] = dict(view_dir=str(view_dir), rgb=str(rgb_path),
                           observation_sha256=hashlib.sha256(observation_path.read_bytes()).hexdigest())
    result['timing'] = dict(pair_preprocessing_and_inference_s=inference_s, warmup_s=warmup_s,
                            segmentation_included=False, model_load_included=False)
    result['hardware_accessed'] = False
    result['raw_scores'] = scores
    names = {n['node_id']: f'N{i}' for i, n in enumerate(nodes)}
    def edges(values):
        return sorted({(names[e['source']], names[e['target']]) for e in values if e['relation']=='possible_occludes'})
    original = edges(view['relation_graph']['edges'])
    final = edges(result['relation_graph']['edges'])
    summary = dict(view_dir=str(view_dir), nodes=len(nodes), all_pairs=len(boundary['pairs']),
                   eligible_pairs=len(eligible), inferred_pairs=len(pairs), baseline_edges=original,
                   augmented_edges=final, added_edges=sorted(set(final)-set(original)),
                   deferred_candidates=[dict(source=names[d['source']], target=names[d['target']],
                       model_score=d['model_score'], reason=d['rejection_reason'])
                       for d in result['deferred_candidates']],
                   rejected_model_predictions=dict(Counter(d['rejection_reason'] for d in result['model_decisions'] if not d['accepted'])),
                   timing=result['timing'], accuracy_evaluated=False,
                   limitation='No complete human ground truth; added edges are hypotheses, not measured accuracy gains.')
    write(output_dir/'summary.json', summary)
    write_result(packet, arrays, result, output_dir, color)
    # Existing project renderer, same masks/IDs, so baseline and augmented views are directly comparable.
    baseline = supplement_relations(boundary, view['relation_graph']['edges'], [])
    write_result(packet, arrays, baseline, output_dir/'baseline', color)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view-dir', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=Path('/home/wpy/armcode-runtime/models/instaorder/source'))
    parser.add_argument('--checkpoint', type=Path, default=Path('/home/wpy/armcode-runtime/models/instaorder/InstaOrder_InstaOrderNet_o.pth.tar'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--audit-all-pairs', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    start = time.perf_counter()
    predictor = InstaOrderPredictor(args.source_dir, args.checkpoint, device=args.device)
    model_load_s = time.perf_counter()-start
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for i, view in enumerate(args.view_dir):
        summary = replay(view, args.output_dir/f'{i:02d}_{view.name}', predictor,
                         audit_all_pairs=args.audit_all_pairs)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    write(args.output_dir/'report.json', dict(
        model='InstaOrderNet_o', source=str(args.source_dir),
        source_commit=subprocess.check_output(['git', '-C', str(args.source_dir), 'rev-parse', 'HEAD'], text=True).strip(),
        checkpoint=str(args.checkpoint), weights_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        model_load_s=model_load_s, audit_all_pairs=args.audit_all_pairs, views=summaries,
        camera_accessed=False, robot_accessed=False, live_agent_integration=False))


if __name__ == '__main__':
    main()
