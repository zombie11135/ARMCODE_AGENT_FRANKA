"""Build a portable MobileCLIP2 reference bank from registered photos, without hardware."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from services.object_reference_service import ObjectReferenceService, atomic_json, image_digest, load_settings
from services.reference_encoder import ENCODER_REVISIONS, MOBILECLIP2_PREPROCESSING


def rebuild_bank(settings, output, *, encoder=None, expected_checkpoint_sha256=None):
    if settings.encoder_model != 'mobileclip2_b':
        raise ValueError('This command requires encoder_model: mobileclip2_b')
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Bank already exists; choose a new output directory: {output}')
    checkpoint_hash = image_digest(settings.checkpoint)
    if expected_checkpoint_sha256 and checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError('Checkpoint SHA-256 differs from the expected weights')
    service = ObjectReferenceService(settings, encoder=encoder)
    manifest = service.gallery.manifest
    manifest_hash = image_digest(manifest)
    data = service.gallery.read()
    rows, images = [], []
    for obj in data['objects']:
        for ref in obj.get('references', []):
            path = service.gallery.directory/ref['path']
            if image_digest(path) != ref['sha256']:
                raise ValueError(f'Reference image changed: {path}')
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
            rows.append(dict(ref, object_id=obj['object_id'], name=obj['name']))
    if not rows:
        raise ValueError('Reference gallery is empty')
    print(f'Encoding {len(rows)} reference photos with {settings.encoder_model}', flush=True)
    features = service._encode(images)
    # A concurrently edited gallery must not produce a bank describing another snapshot.
    if image_digest(manifest) != manifest_hash:
        raise ValueError('Gallery manifest changed during encoding; retry with a stable gallery')
    for row in rows:
        if image_digest(service.gallery.directory/row['path']) != row['sha256']:
            raise ValueError('Reference image changed during encoding')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.building-', dir=output.parent) as directory:
        staging = Path(directory)/'bank'
        staging.mkdir()
        index = staging/'index.npz'
        np.savez_compressed(index, features=features,
            object_ids=np.asarray([r['object_id'] for r in rows]),
            image_sha256=np.asarray([r['sha256'] for r in rows]))
        metadata = dict(schema_version=1, model_key=settings.encoder_model,
            encoder_revision=ENCODER_REVISIONS[settings.encoder_model],
            checkpoint_sha256=checkpoint_hash, preprocessing=MOBILECLIP2_PREPROCESSING,
            gallery_sha256=manifest_hash, index_sha256=image_digest(index),
            created_at=datetime.now(timezone.utc).isoformat(), references=rows)
        atomic_json(staging/'metadata.json', metadata)
        validator = ObjectReferenceService(replace(settings, vector_library_dir=str(staging)))
        restored = validator._prepared_features(data)
        if restored is None or not np.array_equal(restored, features):
            raise ValueError('Exported bank failed the production loader round-trip')
        if output.exists():
            raise FileExistsError(f'Bank appeared during encoding: {output}')
        staging.rename(output)
    return dict(model=settings.encoder_model, reference_count=len(rows),
        class_count=len({r['object_id'] for r in rows}), feature_shape=list(features.shape),
        checkpoint_sha256=checkpoint_hash, index_sha256=metadata['index_sha256'],
        output=str(output), production_loader_verified=True, hardware_accessed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/perception/references.example.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-checkpoint-sha256')
    args = parser.parse_args()
    settings = load_settings(args.config)
    output = args.output or settings.vector_library_dir
    if not output:
        parser.error('Provide --output or vector_library_dir in the configuration')
    result = rebuild_bank(settings, output, expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
