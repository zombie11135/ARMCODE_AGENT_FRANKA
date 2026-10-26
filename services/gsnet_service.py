"""GSNet implementation of the legacy grasp service contract, isolated in a child process."""
import json
import hashlib
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
from runtime.perception_pipeline import serialized_resource

from hardware.agent_geometry import capture_pose
from hardware.config import load_config, PROJECT_ROOT
from hardware.transforms import load_transforms


def write_cloud(path, points, rgb):
    valid = np.isfinite(points).all(axis=2) & (points[:, :, 2] > .02) & (points[:, :, 2] < 1.5)
    xyz, colors = points[valid], rgb[valid]
    if len(xyz) < 200:
        raise ValueError('Not enough valid SDK cloud points')
    rows = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, key in enumerate(('x', 'y', 'z')):
        rows[key] = xyz[:, i]
    for i, key in enumerate(('red', 'green', 'blue')):
        rows[key] = colors[:, i]
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex %d\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n') % len(xyz)
    with Path(path).open('wb') as stream:
        stream.write(header.encode())
        rows.tofile(stream)


class GSNetService:
    backend = 'gsnet'

    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.transforms = load_transforms(self.config)
        self.last_run = None
        self._candidate_cache = OrderedDict()

    def _model_request(self, **options):
        root = Path(self.config.perception.gsnet_root)
        return {'gsnet_root': str(root), 'checkpoint': str(root / 'checkpoints/graspness_realsense.tar')}

    def _run_inference(self, request_path, log):
        subprocess.run([sys.executable, str(PROJECT_ROOT / 'hardware/gsnet_worker.py'), str(request_path)],
                       check=True, timeout=180, stdout=log, stderr=subprocess.STDOUT)

    @serialized_resource('gpu')
    def plan_grasp(self, *, color_path, intrinsics, capture_geometry=None,
                   points_path=None, mask_path=None, mask_npy_path=None, bbox=None,
                   bbox_filter_margin_px=0, bbox_filter_inner_margin_ratio=0,
                   output_json_path='runs/current_capture/gsnet_result.json',
                   retain_all_candidates=False, **kwargs):
        import cv2
        observation = {'capture_geometry': capture_geometry}
        pose = capture_pose(observation, self.transforms)
        if not points_path:
            raise ValueError(f'{self.backend} needs points_path from the same calibrated RGB-D observation')
        points = np.load(points_path, allow_pickle=False)
        bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if bgr is None or points.shape != (*bgr.shape[:2], 3):
            raise ValueError('RGB and SDK cloud dimensions differ')
        height, width = bgr.shape[:2]
        root = PROJECT_ROOT / f'runs/agent_{self.backend}' / uuid.uuid4().hex
        capture = root / 'capture'
        capture.mkdir(parents=True)
        write_cloud(capture / 'scene.ply', points, bgr[:, :, ::-1])
        k = dict(intrinsics)
        k.update(width=width, height=height, ppx=k.get('cx', k.get('ppx')),
                 ppy=k.get('cy', k.get('ppy')), coeffs=k.get('coeffs', [0.] * 5))
        serial = capture_geometry['serial']
        meta = {'device': {'serial_number': serial}, 'frame': f'realsense_{serial}_color_optical',
                'color_intrinsics': k, 'workspace': {'roi_xyxy': None},
                'deprojection': 'rs.pointcloud.calculate on aligned depth'}
        (capture / 'capture.json').write_text(json.dumps(meta, indent=2))
        (root / 'capture_pose.json').write_text(json.dumps({
            'before': {'pose': pose.values}, 'after': {'pose': capture_geometry['after']},
            'calibration': capture_geometry['calibration']}, indent=2))
        workcell = json.loads((Path(self.config.arm.workspace) / 'config/workcell.json').read_text())
        roi = [0, 0, width, height]
        if bbox is not None:
            x0, y0, x1, y1 = map(float, bbox)
            margin = float(bbox_filter_margin_px)
            inner = float(bbox_filter_inner_margin_ratio)
            if not 0 <= inner < .5:
                raise ValueError('bbox inner margin ratio must be in [0,.5)')
            dx, dy = (x1 - x0) * inner, (y1 - y0) * inner
            roi = [max(0, x0 + dx - margin), max(0, y0 + dy - margin),
                   min(width, x1 - dx + margin), min(height, y1 - dy + margin)]
            if roi[0] >= roi[2] or roi[1] >= roi[3]:
                raise ValueError('Target bbox is outside the image or empty')
        request = {**self._model_request(**kwargs),
                   'capture_dir': str(capture), 'output_dir': str(root / self.backend), 'roi': roi,
                   'base_T_camera': self.transforms.base_T_camera('wrist', pose).tolist(),
                   'table_z_m': workcell['table_z_m'], 'defer_preview': True,
                   'grasp_table_clearance_check': workcell.get('grasp_table_clearance_check', True),
                   'export_filtered_candidates': bool(retain_all_candidates)}
        if mask_path or mask_npy_path:
            mask = (np.load(mask_npy_path, allow_pickle=False) if mask_npy_path
                    else cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))
            if mask is None or mask.shape != (height, width) or not np.any(mask):
                raise ValueError('Target SAM mask must be nonempty and match the captured image')
            np.save(root / 'target_mask.npy', mask.astype(bool))
            request['target_mask_path'] = str(root / 'target_mask.npy')
        request_path = root / 'request.json'
        request_path.write_text(json.dumps(request, indent=2))
        # Cache model output only. Final yaw selection, frame validation and
        # motion planning still run at consumption time. Content hashes prevent
        # a rewritten file or a changed ROI/mask from borrowing old candidates.
        cache_input = {k: v for k, v in request.items()
                       if k not in ('capture_dir', 'output_dir', 'target_mask_path')}
        cache_input.update(cloud_sha256=hashlib.sha256((capture/'scene.ply').read_bytes()).hexdigest(),
            mask_sha256=(hashlib.sha256(mask.astype(bool).tobytes()).hexdigest()
                         if request.get('target_mask_path') else None),
            color_intrinsics=k, capture_geometry=capture_geometry, workcell=workcell,
            scene_token=kwargs.get('scene_token'))
        cache_key = hashlib.sha256(json.dumps(cache_input, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if cache_key in self._candidate_cache:
            self._candidate_cache.move_to_end(cache_key)
            result = deepcopy(self._candidate_cache[cache_key])
            result.update(result_json_path=str(output_json_path), inference_cache_hit=True)
            output = Path(output_json_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2))
            self.last_run = Path(result['run_dir'])
            print(f'[perception_pipeline] {self.backend} candidate cache hit; inference skipped')
            return result
        with (root / 'inference.log').open('w') as log:
            try:
                self._run_inference(request_path, log)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(f'{self.backend} 推理进程失败，详情：{root / "inference.log"}') from error
        if retain_all_candidates:
            rows = np.load(root / self.backend / 'filtered_candidates.npy', allow_pickle=False)
            if rows.ndim != 2 or rows.shape[1] != 17 or not np.isfinite(rows).all():
                raise ValueError(f'Invalid {self.backend} filtered candidate pool')
            if not len(rows):
                report_path = root / self.backend / 'report.json'
                report = json.loads(report_path.read_text())
                raise RuntimeError(f"{self.backend} 筛选后无可用候选: {report['counts']}; report: {report_path}")
            candidates = [dict(index=i, score=float(row[0]), width=float(row[1]),
                height=float(row[2]), depth=float(row[3]), rotation_matrix=row[4:13].reshape(3, 3).tolist(),
                translation=row[13:16].tolist()) for i, row in enumerate(rows)]
            selection = {'selection_pending': True}
        else:
            selected = json.loads((root / self.backend / 'grasps.json').read_text())['grasps'][0]
            candidates = [{'index': 0, 'score': selected['score'], 'width': selected['width_m'],
                'depth': selected['depth_m'], 'height': selected['height_m'],
                'translation': selected['translation_m'], 'rotation_matrix': selected['rotation_matrix']}]
            selection = {'selected': selected, 'best_grasp_camera': candidates[0]}
        result = {'success': True, 'backend': self.backend, 'num_candidates': len(candidates),
                  'candidate_grasps_camera': candidates,
                  'run_dir': str(root), 'workcell': workcell, **selection,
                  'roi_source': 'sam_mask' if request.get('target_mask_path') else 'bbox',
                  'result_json_path': str(output_json_path), 'inference_cache_hit': False}
        output = Path(output_json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        self.last_run = root
        self._candidate_cache[cache_key] = deepcopy(result)
        while len(self._candidate_cache) > 4:
            self._candidate_cache.popitem(last=False)
        return result

    def save_geometry(self, plan):
        root = Path(plan['run_dir'])
        selected = plan['selected']
        np.save(root / self.backend / 'grasps.npy', np.asarray([[selected['score'], selected['width_m'],
            selected['height_m'], selected['depth_m'], *np.asarray(selected['rotation_matrix']).ravel(),
            *selected['translation_m'], -1.]], dtype=float))
        meta = json.loads((root / 'capture/capture.json').read_text())
        (root / self.backend / 'grasps.json').write_text(json.dumps({'frame': meta['frame'],
            'pose_definition': 'Raw GraspNet frame: X approach, Y jaw line; not flange/TCP',
            'grasps': [selected]}, indent=2))
        (root / 'pose_selection.json').write_text(json.dumps(plan['pose_selection'], indent=2))
        # Keep the public result and viewer consistent with the re-ranked choice.
        Path(plan['result_json_path']).write_text(json.dumps(plan, indent=2))
        (root / 'targets.json').write_text(json.dumps({**plan['geometry'], 'selected': plan['selected'],
            'workcell': plan['workcell'], 'calibration': self.transforms.summary()}, indent=2))

    def visualize_last_grasps_by_indices(self, indices, window_name=None):
        if list(indices) != [0] or self.last_run is None:
            raise ValueError('GSNet exposes one selected grasp, index 0')
        subprocess.run([sys.executable, '-m', 'hardware.gsnet_o3d', str(self.last_run)],
                       cwd=PROJECT_ROOT, check=True)
