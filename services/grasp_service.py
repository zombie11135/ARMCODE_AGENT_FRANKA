"""Backend selection without importing the two incompatible gsnet packages together."""
from hardware.config import load_config
from .runtime_service_cache import get_or_create_service


def detector_model_path(config):
    path = getattr(config, 'hardware_config_path', None)
    if path:
        configured = load_config(path).perception.grounding_dino_model
        if configured:
            return configured
    return config.grounding_dino_model


def build_grasp_service(config):
    path = getattr(config, 'hardware_config_path', None)
    if path and load_config(path).perception.grasp_backend == 'gsnet':
        from .gsnet_service import GSNetService
        return get_or_create_service('open_gsnet', {'config': str(path)}, lambda: GSNetService(path))
    if path:
        from .anygrasp_fr3_service import AnyGraspFR3Service
        return get_or_create_service('fr3_anygrasp', {'config': str(path)}, lambda: AnyGraspFR3Service(path))
    from .anygrasp_service import AnyGraspService
    return AnyGraspService(grasp_detection_dir=config.anygrasp_dir,
                          checkpoint_path=config.anygrasp_ckpt)


class LazySAM3:
    def __init__(self, model_dir=None, prompt_mode="geometric"):
        self.model_dir = model_dir
        self.prompt_mode = prompt_mode

    def segment_by_bbox(self, *args, **kwargs):
        from .sam3_bbox_segmenter_service import SAM3BBoxSegmenterService
        params = {'model_dir': self.model_dir} if self.model_dir else {}
        params['prompt_mode'] = self.prompt_mode
        service = get_or_create_service('sam3', params, lambda: SAM3BBoxSegmenterService(**params))
        return service.segment_by_bbox(*args, **kwargs)


def build_segmenter(config):
    path = getattr(config, 'hardware_config_path', None)
    return LazySAM3(load_config(path).perception.sam3_model_dir if path else None,
                    prompt_mode='instance' if path else 'geometric')
