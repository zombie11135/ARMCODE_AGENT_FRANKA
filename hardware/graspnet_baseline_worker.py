"""Original GraspNet-baseline PointNet2 inference, isolated from Agent models."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def decode_with_width_limit(module, endpoints, max_width):
    """Configure upstream decoding, then restore its default in this child process.

    This caps decoded jaw opening; it does not condition/retrain the network or
    change grasp centres, orientations or scores. Full scene checks follow.
    """
    from hardware.graspnet_worker_common import checked_gripper_width
    limit = checked_gripper_width(max_width)
    original = module.GRASP_MAX_WIDTH
    try:
        module.GRASP_MAX_WIDTH = limit
        return module.pred_decode(endpoints)
    finally:
        module.GRASP_MAX_WIDTH = original


def load_model_adapter(root):
    root = Path(root).resolve()
    sys.path[:0] = [str(root / child) for child in ('models', 'utils', 'pointnet2')]
    sys.path.insert(0, str(root))
    import open3d as o3d
    import torch
    import graspnet as baseline
    from graspnet import GraspNet
    from collision_detector import ModelFreeCollisionDetector

    def read_cloud(args):
        cloud = o3d.io.read_point_cloud(str(args.cloud))
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors)
        if len(points) < 200 or not np.isfinite(points).all():
            raise ValueError('GraspNet-baseline requires a finite scene cloud with at least 200 points')
        return points, colors, {'kind': 'point_cloud', 'path': str(args.cloud)}

    def infer(args, points):
        if not torch.cuda.is_available():
            raise RuntimeError('GraspNet-baseline PointNet2 requires CUDA')
        model = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=.05, hmin=-.02, hmax_list=[.01, .02, .03, .04], is_training=False)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        epoch = checkpoint.get('epoch')
        del checkpoint
        model.cuda().eval()
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(points), args.num_point, replace=len(points) < args.num_point)
        batch = {'point_clouds': torch.from_numpy(points[indices].astype(np.float32)[None]).cuda()}
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            raw = decode_with_width_limit(baseline, model(batch), args.max_width)[0].cpu().numpy()
        torch.cuda.synchronize()
        if raw.ndim != 2 or raw.shape[1] != 17 or not np.isfinite(raw).all():
            raise ValueError('Invalid GraspNet-baseline Nx17 output')
        return raw, {'implementation': 'graspnet/graspnet-baseline PointNet2',
            'model_root': str(root), 'checkpoint_epoch': int(epoch) if epoch is not None else None,
            'decoder_max_gripper_width_m': args.max_width,
            'width_policy': 'upstream decoder clamp followed by full-scene collision and empty checks',
            'sampled_points': args.num_point, 'model_forward_s': time.perf_counter() - start}

    return SimpleNamespace(read_cloud=read_cloud, infer=infer,
                           ModelFreeCollisionDetector=ModelFreeCollisionDetector)


def view_saved(root, stage='selected'):
    """Use the upstream demo's native GraspNet/Open3D geometry, without overlays."""
    import open3d as o3d
    from graspnetAPI import GraspGroup
    root = Path(root)
    cloud = o3d.io.read_point_cloud(str(root / 'capture/scene.ply'))
    files = {'raw': 'grasps_raw.npy', 'filtered': 'filtered_candidates.npy', 'selected': 'grasps.npy'}
    group = GraspGroup(str(root / 'graspnet_baseline' / files[stage]))
    if stage == 'selected' and len(group) != 1:
        raise ValueError('Expected the one selected grasp')
    o3d.visualization.draw_geometries([cloud, *group.to_open3d_geometry_list()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--stage', choices=('raw', 'filtered', 'selected'), default='selected',
                        help='Native viewer candidate set (default: selected)')
    args = parser.parse_args()
    if args.view:
        view_saved(args.path, args.stage)
        return
    from hardware.graspnet_worker_common import run
    request = json.loads(args.path.read_text())
    run(request, load_model_adapter(request['graspnet_root']))


if __name__ == '__main__':
    main()
