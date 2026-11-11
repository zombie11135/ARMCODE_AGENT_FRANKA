"""Original GraspNet-baseline with the existing calibrated Agent contract."""
import os
from pathlib import Path
import subprocess

from hardware.config import PROJECT_ROOT
from runtime.timing import timed_stage
from .gsnet_service import GSNetService


def validate_graspnet_paths(perception):
    for name in ('graspnet_root', 'graspnet_ckpt', 'graspnet_python'):
        value = getattr(perception, name)
        if not value or not Path(value).is_absolute():
            raise ValueError(f'perception.{name} must be an absolute path')
    for path in (Path(perception.graspnet_root) / 'models/graspnet.py',
                 Path(perception.graspnet_ckpt), Path(perception.graspnet_python)):
        if not path.is_file():
            raise FileNotFoundError(f'GraspNet-baseline dependency missing: {path}')
    if not os.access(perception.graspnet_python, os.X_OK):
        raise ValueError('graspnet_python is not executable')


class GraspNetBaselineService(GSNetService):
    # The parent supplies capture validation/cache/pose output, not GSNet inference.
    backend = 'graspnet_baseline'

    def __init__(self, config_path):
        super().__init__(config_path)
        validate_graspnet_paths(self.config.perception)

    def _model_request(self, **options):
        p = self.config.perception
        return {'graspnet_root': p.graspnet_root, 'checkpoint': p.graspnet_ckpt,
                'num_point': 20000, 'backend': self.backend,
                'max_gripper_width': p.graspnet_max_gripper_width_m}

    def _run_inference(self, request_path, log):
        env = os.environ.copy()
        env.setdefault('OMP_NUM_THREADS', '8')
        subprocess.run([self.config.perception.graspnet_python, '-u', '-m',
                        'hardware.graspnet_baseline_worker', str(request_path)],
                       cwd=PROJECT_ROOT, env=env, check=True, timeout=180,
                       stdout=log, stderr=subprocess.STDOUT)

    @timed_stage('抓取位姿预览窗口等待')
    def visualize_last_grasps_by_indices(self, indices, window_name=None):
        if list(indices) != [0] or self.last_run is None:
            raise ValueError('GraspNet-baseline exposes one selected grasp, index 0')
        subprocess.run([self.config.perception.graspnet_python, '-m',
                        'hardware.graspnet_baseline_worker', '--view', str(self.last_run)],
                       cwd=PROJECT_ROOT, check=True)
