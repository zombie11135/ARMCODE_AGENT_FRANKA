"""Share frozen SAM3 weights; processors keep their per-image/prompt state."""
from pathlib import Path

from runtime.perception_pipeline import serialized_resource
from services.runtime_service_cache import get_or_create_service


@serialized_resource('gpu')
def image_model(checkpoint, *, device='cuda', interactive=True, compile_model=False):
    path = str(Path(checkpoint).resolve())
    params = dict(checkpoint=path, device=str(device), interactive=bool(interactive),
                  compile_model=bool(compile_model))
    def build():
        from sam3.model_builder import build_sam3_image_model
        # Some SAM3 releases only move weights for the exact string 'cuda'.
        # Load on CPU first, then honor indexed devices such as 'cuda:1'.
        model = build_sam3_image_model(checkpoint_path=path, load_from_HF=False,
            device='cpu', eval_mode=True, enable_segmentation=True,
            enable_inst_interactivity=interactive, compile=compile_model)
        return model.to(device=device)
    return get_or_create_service('sam3_image_weights', params, build)
