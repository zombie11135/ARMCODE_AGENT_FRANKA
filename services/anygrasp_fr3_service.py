"""Licensed AnyGrasp with the calibrated FR3 service contract, in its own Python."""
import os
from pathlib import Path
import subprocess

from hardware.config import PROJECT_ROOT
from .gsnet_service import GSNetService


def validate_anygrasp_paths(perception):
    for name in ('anygrasp_dir', 'anygrasp_ckpt', 'anygrasp_python'):
        value = getattr(perception, name)
        if not value or not Path(value).is_absolute():
            raise ValueError(f'perception.{name} must be an absolute path for FR3 AnyGrasp')
    sdk = Path(perception.anygrasp_dir)
    for path in (Path(perception.anygrasp_python), Path(perception.anygrasp_ckpt), sdk / 'gsnet.so',
                 sdk / 'lib_cxx.so', sdk / 'license/licenseCfg.json'):
        if not path.is_file():
            raise FileNotFoundError(f'AnyGrasp dependency missing: {path}')
    if not os.access(perception.anygrasp_python, os.X_OK):
        raise ValueError('anygrasp_python is not executable')


class AnyGraspFR3Service(GSNetService):
    """Share capture validation, full candidate ranking and result saving with FR3."""
    backend = 'anygrasp'

    def __init__(self, config_path):
        super().__init__(config_path)
        validate_anygrasp_paths(self.config.perception)

    def _model_request(self, **options):
        perception = self.config.perception
        return dict(sdk_dir=perception.anygrasp_dir, checkpoint=perception.anygrasp_ckpt,
                    max_gripper_width=.085, gripper_height=.03, top_down_grasp=False,
                    apply_object_mask=options.get('apply_object_mask', True),
                    dense_grasp=options.get('dense_grasp', False),
                    collision_detection=options.get('collision_detection', True))

    def _run_inference(self, request_path, log):
        environment = os.environ.copy()
        compat = Path(self.config.perception.anygrasp_dir).parent / 'compat_libs'
        environment['LD_LIBRARY_PATH'] = str(compat) + ':' + environment.get('LD_LIBRARY_PATH', '')
        environment.setdefault('OMP_NUM_THREADS', '8')
        subprocess.run([self.config.perception.anygrasp_python, '-u', '-m',
                        'hardware.anygrasp_worker', str(request_path)],
                       cwd=PROJECT_ROOT, env=environment, check=True, timeout=180,
                       stdout=log, stderr=subprocess.STDOUT)

    def visualize_last_grasps_by_indices(self, indices, window_name=None):
        if list(indices) != [0] or self.last_run is None:
            raise ValueError('AnyGrasp exposes one selected grasp, index 0')
        subprocess.run([self.config.perception.anygrasp_python, '-m', 'hardware.anygrasp_offline',
                        'view', str(self.last_run), '--stage', 'selected'],
                       cwd=PROJECT_ROOT, check=True)
