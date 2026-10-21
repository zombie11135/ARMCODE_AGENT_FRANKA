"""Offline two-view inference; consumes observation packets, never hardware."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .multiview import MultiViewConfig
from .planar import PlanarConfig
from .scene import infer_scene


def load_observation(path):
    path = Path(path)
    packet = json.loads(path.read_text())
    arrays_path = path.parent / packet["artifacts"]["arrays"]
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    provenance = {
        "packet": str(path.resolve()),
        "packet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
    }
    return (packet, arrays), provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-planar-prior", action="store_true")
    parser.add_argument("--config", type=Path, help='JSON with "multiview" and "planar" configuration dictionaries')
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in args.observations}:
        parser.error("output must not replace an input packet")
    config = json.loads(args.config.read_text()) if args.config else {}
    loaded = [load_observation(p) for p in args.observations]
    start = time.perf_counter()
    result = infer_scene(
        [x[0] for x in loaded], planar_prior=not args.no_planar_prior,
        multiview_config=MultiViewConfig(**config.get("multiview", {})),
        planar_config=PlanarConfig(**config.get("planar", {})))
    result["runtime_s"] = time.perf_counter() - start
    result["input_provenance"] = [x[1] for x in loaded]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "edges": len(result["relation_graph"]["edges"]),
                      "runtime_s": result["runtime_s"]}))


if __name__ == "__main__":
    main()
