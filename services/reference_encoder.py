"""Lazy image encoders for reference identity matching; no hardware imports."""
from pathlib import Path


# Include the model and preprocessing contract in every persistent feature key.
ENCODER_REVISIONS = {
    "mobileclip_blt": "mobileclip_b_rgb_crop_v1",
    "mobileclip2_b": "mobileclip2_b_openclip_dfndr2b_rgb224_center_crop_reparam_v1",
}
MOBILECLIP2_PREPROCESSING = (
    "Compose(\n"
    "    Resize(size=224, interpolation=bilinear, max_size=None, antialias=True)\n"
    "    CenterCrop(size=(224, 224))\n"
    "    MaybeConvertMode()\n"
    "    MaybeToTensor()\n"
    "    Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))\n"
    ")"
)


class MobileCLIP2Encoder:
    """Use the same local weights and preprocessing as the wrist-photo evaluation."""

    def __init__(self, checkpoint, device="cuda:0"):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MobileCLIP2 checkpoint not found: {checkpoint}")
        import open_clip
        from timm.utils import reparameterize_model

        cfg = open_clip.get_pretrained_cfg("MobileCLIP2-B", "dfndr2b")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP2-B", pretrained=str(checkpoint), device=device,
            image_mean=cfg["mean"], image_std=cfg["std"],
            image_interpolation=cfg["interpolation"], image_resize_mode=cfg["resize_mode"],
        )
        if str(self.preprocess) != MOBILECLIP2_PREPROCESSING:
            raise ValueError("MobileCLIP2 preprocessing differs from the validated gallery; check dependencies")
        self.model.eval()
        self.model = reparameterize_model(self.model, inplace=True)
        self.device = device

    def encode(self, images, batch_size=16):
        import numpy as np
        import torch

        images = list(images)
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        parts = []
        with torch.inference_mode():
            for offset in range(0, len(images), batch_size):
                batch = torch.stack([self.preprocess(im.convert("RGB"))
                                     for im in images[offset:offset + batch_size]]).to(self.device)
                encoded = self.model.encode_image(batch, normalize=True)
                parts.append(encoded.float().cpu().numpy())
        return np.concatenate(parts).astype(np.float32, copy=False)


def build_reference_encoder(model, checkpoint, device):
    if model == "mobileclip2_b":
        return MobileCLIP2Encoder(checkpoint, device)
    if model == "mobileclip_blt":
        # Keep old deployments usable without requiring both packages at startup.
        from perception_v2.retrieval import MobileCLIPGallery
        return MobileCLIPGallery(Path(checkpoint), device)
    raise ValueError(f"Unsupported reference encoder: {model}")
