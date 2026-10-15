"""Local image-reference identity matching. No camera, robot or LLM imports.

Cosine scores are similarity measures, not calibrated probabilities. An accepted
identity still needs segmentation, depth, grasp and motion validation downstream.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

import numpy as np
from PIL import Image
import yaml

from .object_names import legacy_object_id, normalize_name

REFERENCE_VIEWPOINTS = ("front", "side", "back", "top", "bottom", "oblique", "unspecified")


@dataclass(frozen=True)
class ReferenceSettings:
    enabled: bool = True
    gallery_dir: str = ""
    checkpoint: str = ""
    device: str = "cuda:0"
    min_similarity: float = 0.86
    min_class_margin: float = 0.05
    top_k_references: int = 3
    batch_size: int = 16
    cache_frames: int = 8
    log_dir: str = ""


def load_settings(path):
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text())
    settings = ReferenceSettings(**raw)
    for key in ("min_similarity", "min_class_margin"):
        value = getattr(settings, key)
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    for key in ("top_k_references", "batch_size", "cache_frames"):
        value = getattr(settings, key)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(settings.enabled) is not bool:
        raise ValueError("enabled must be boolean")
    raw = asdict(settings)
    for key in ("gallery_dir", "checkpoint", "log_dir"):
        if not raw[key]:
            raise ValueError(f"{key} is required")
        p = Path(raw[key]).expanduser()
        raw[key] = str((p if p.is_absolute() else path.parent / p).resolve())
    return ReferenceSettings(**raw)


def image_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def crop_image(image, bbox=None):
    image = image.convert("RGB")
    if bbox is None:
        return image.copy()
    a = np.asarray(bbox, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError("bbox must be four finite XYXY pixel coordinates")
    x1, y1, x2, y2 = [int(round(x)) for x in a]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        raise ValueError("reference/candidate crop is too small or outside image")
    return image.crop((x1, y1, x2, y2))


class ReferenceGallery:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest = self.directory / "gallery.json"

    def read(self):
        if not self.manifest.exists():
            return {"schema_version": 2, "objects": []}
        data = json.loads(self.manifest.read_text())
        if data.get("schema_version") not in (1, 2) or not isinstance(data.get("objects"), list):
            raise ValueError("Invalid reference gallery manifest")
        # A read-only migration view preserves old galleries until the next explicit write.
        legacy_schema = data['schema_version'] == 1
        data['schema_version'] = 2
        ids = set()
        for obj in data['objects']:
            if legacy_schema:
                obj.setdefault('object_id', legacy_object_id(obj['name']))
            elif not obj.get('object_id'):
                raise ValueError('Schema-2 objects must keep their stable object_id')
            obj.setdefault('attributes', {})
            obj.setdefault('descriptors', [])
            obj.setdefault('aliases', [])
            key = normalize_name(obj['object_id'])
            if not key.startswith('ref_') or key in ids:
                raise ValueError('Invalid or duplicate object_id')
            ids.add(key)
            if not isinstance(obj['attributes'], dict) or not all(isinstance(v, str) for v in obj['attributes'].values()):
                raise ValueError('Object attributes must be string values')
            if not isinstance(obj['descriptors'], list) or not all(isinstance(v, str) and v.strip() for v in obj['descriptors']):
                raise ValueError('Object descriptors must be nonempty strings')
        names = set()
        for obj in data["objects"]:
            for name in [obj["name"], *obj.get("aliases", [])]:
                token = normalize_name(name)
                if not token or token in names or token in ids:
                    raise ValueError(f"Duplicate/empty gallery name or alias: {name}")
                names.add(token)
            for ref in obj.get("references", []):
                p = (self.directory / ref["path"]).resolve()
                if not p.is_relative_to(self.directory):
                    raise ValueError("Reference path must stay inside gallery")
        return data

    def resolve(self, query):
        token = normalize_name(query)
        return next((o for o in self.read()["objects"]
                     if token in [normalize_name(x) for x in [o['object_id'], o["name"], *o.get("aliases", [])]]), None)

    def catalog(self):
        return [{k: obj.get(k) for k in ("object_id", "name", "category", "aliases", "attributes", "descriptors", "detector_prompt")}
                | {"reference_count": len(obj.get("references", []))}
                for obj in self.read()["objects"] if obj.get("references")]

    @contextmanager
    def _lock(self):
        import fcntl
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def register(self, *, name, image_path, category, detector_prompt, bbox=None,
                 aliases=(), source_group=None, provenance="user_registered", source_image_sha256=None,
                 attributes=None, descriptors=(), viewpoint=None, capture_metadata=None):
        if viewpoint is not None and viewpoint not in REFERENCE_VIEWPOINTS:
            raise ValueError(f"Unknown reference viewpoint: {viewpoint}")
        if capture_metadata is not None:
            if not isinstance(capture_metadata, dict):
                raise ValueError("capture_metadata must be a JSON object")
            capture_metadata = json.loads(json.dumps(capture_metadata, allow_nan=False))
        name, category, detector_prompt = (str(x).strip() for x in (name, category, detector_prompt))
        aliases = self._unique_aliases(name, aliases)
        if not all((name, category, detector_prompt)):
            raise ValueError("name, category and an English detector_prompt are required")
        source_hash = source_image_sha256 or image_digest(image_path)
        with Image.open(image_path) as im:
            crop = crop_image(im, bbox)
        # Stored references are immutable, independent of mutable runs/current_capture files.
        import io
        buf = io.BytesIO(); crop.save(buf, format="PNG")
        digest = hashlib.sha256(buf.getvalue()).hexdigest()
        with self._lock():
            data = self.read()
            target = next((o for o in data["objects"] if normalize_name(o["name"]) == normalize_name(name)), None)
            other_tokens = {normalize_name(s) for o in data["objects"] if o is not target
                            for s in [o['object_id'], o["name"], *o.get("aliases", [])]}
            if any(normalize_name(s) in other_tokens or normalize_name(s).startswith('ref_') for s in [name, *aliases]):
                raise ValueError("Name/alias is already assigned to another object")
            if target is None:
                target = dict(object_id='ref_' + uuid.uuid4().hex[:20], name=name, category=category, detector_prompt=detector_prompt,
                              aliases=aliases, attributes={}, descriptors=[], references=[])
                data["objects"].append(target)
            elif target["category"] != category or target["detector_prompt"] != detector_prompt:
                raise ValueError("Existing object category/prompt differ; use its registered settings")
            if attributes and any(target['attributes'].get(k) and target['attributes'][k] != str(v).strip()
                                  for k, v in attributes.items()):
                raise ValueError('Existing brand/specification differs; register a distinct name or explicitly update metadata')
            target["aliases"] = self._unique_aliases(target["name"], [*target.get("aliases", []), *aliases])
            self._update_attributes(target, attributes, descriptors)
            existing = next((r for r in target["references"] if r["sha256"] == digest), None)
            added = existing is None
            if added:
                rel = f"images/{digest}.png"
                path = self.directory / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(buf.getvalue())
                existing = dict(path=rel, sha256=digest,
                    source_image_sha256=source_hash, source_group=source_group,
                    provenance=provenance)
                if viewpoint is not None:
                    existing["viewpoint"] = viewpoint
                if capture_metadata is not None:
                    existing["capture"] = capture_metadata
                target["references"].append(existing)
            atomic_json(self.manifest, data)
            return dict(object_id=target['object_id'], name=target["name"], reference_count=len(target["references"]),
                        sha256=digest, added=added, reference_path=existing["path"],
                        viewpoint=existing.get("viewpoint", "unspecified"))

    def update(self, query, *, name=None, aliases=(), attributes=None, descriptors=(), remove_aliases=()):
        """Explicit metadata edits, with stable IDs and no need to re-import an image."""
        with self._lock():
            data = self.read()
            obj = next((o for o in data['objects'] if normalize_name(query) in
                        [normalize_name(s) for s in [o['object_id'], o['name'], *o['aliases']]]), None)
            if obj is None:
                raise ValueError(f'Unknown registered object: {query}')
            old_name = obj['name']
            obj['name'] = str(name).strip() if name is not None else old_name
            obj['aliases'] = self._unique_aliases(obj['name'], [*obj['aliases'], old_name, *aliases])
            removed = {normalize_name(s) for s in remove_aliases}
            obj['aliases'] = [s for s in obj['aliases'] if normalize_name(s) not in removed]
            occupied = {normalize_name(s) for other in data['objects'] if other is not obj
                        for s in [other['object_id'], other['name'], *other['aliases']]}
            if any(not normalize_name(s) or normalize_name(s) in occupied or normalize_name(s).startswith('ref_')
                   for s in [obj['name'], *obj['aliases']]):
                raise ValueError('Name/alias is empty, reserved or already assigned to another object')
            self._update_attributes(obj, attributes, descriptors)
            atomic_json(self.manifest, data)
            return copy.deepcopy(obj)

    @staticmethod
    def _update_attributes(obj, attributes, descriptors):
        if attributes is not None:
            if not isinstance(attributes, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in attributes.items()):
                raise ValueError('Object attributes must be string values')
            obj['attributes'].update({k: v.strip() for k, v in attributes.items()})
        obj['descriptors'] = ReferenceGallery._unique_aliases('', [*obj.get('descriptors', []), *descriptors])

    @staticmethod
    def _unique_aliases(name, aliases):
        seen, result = {normalize_name(name)}, []
        for alias in aliases:
            alias = str(alias).strip()
            if alias and normalize_name(alias) not in seen:
                result.append(alias)
                seen.add(normalize_name(alias))
        return result


class ObjectReferenceService:
    def __init__(self, settings, encoder=None):
        self.settings = settings
        self.gallery = ReferenceGallery(settings.gallery_dir)
        self._encoder = encoder
        self._feature_key = None
        self._features = None
        self._refs = []
        self._frame_features = OrderedDict()
        self._lock = threading.RLock()

    def catalog(self):
        return self.gallery.catalog() if self.settings.enabled else []

    def resolve(self, query):
        return self.gallery.resolve(query) if self.settings.enabled else None

    def _encode(self, images):
        if self._encoder is None:
            from perception_v2.retrieval import MobileCLIPGallery
            self._encoder = MobileCLIPGallery(Path(self.settings.checkpoint), self.settings.device)
        values = np.asarray(self._encoder.encode(images, batch_size=self.settings.batch_size), dtype=np.float32)
        if values.ndim != 2 or len(values) != len(images) or not np.isfinite(values).all():
            raise ValueError("Invalid image embeddings")
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Zero image embedding")
        return values / norm

    def _load_features(self):
        data = self.gallery.read()
        refs = [(o["name"], r) for o in data["objects"] for r in o.get("references", [])]
        if not refs:
            raise ValueError("Reference gallery is empty")
        checkpoint = Path(self.settings.checkpoint)
        stat = checkpoint.stat()
        # Include reference file metadata as well as manifest and checkpoint identity.
        signatures = [(o['object_id'], r['sha256'], str((self.gallery.directory / r["path"]).resolve()),
                       (self.gallery.directory / r["path"]).stat().st_mtime_ns)
                      for o in data['objects'] for r in o.get('references', [])]
        # Names/aliases/brands do not change image embeddings; refresh their metadata separately.
        self._refs = refs
        key = hashlib.sha256(json.dumps([signatures, str(checkpoint), stat.st_size,
            stat.st_mtime_ns, "mobileclip_b_rgb_crop_v1"], sort_keys=True).encode()).hexdigest()
        if key == self._feature_key:
            return "memory"
        path = self.gallery.directory / "features" / f"{key}.npz"
        hit = False
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                features = cached["features"]
            hit = (features.ndim == 2 and len(features) == len(refs) and np.isfinite(features).all()
                   and np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4))
        if not hit:
            images = []
            for _, ref in refs:
                p = self.gallery.directory / ref["path"]
                if image_digest(p) != ref["sha256"]:
                    raise ValueError(f"Reference image changed: {p}")
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            features = self._encode(images)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
            try:
                with os.fdopen(fd, "wb") as out:
                    np.savez_compressed(out, features=features)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        self._features, self._refs, self._feature_key = features, refs, key
        self._frame_features.clear()
        return "disk" if hit else "built"

    def match(self, *, image_path, candidates, query, exclude_groups=()):
        started = time.perf_counter()
        obj = self.resolve(query)
        if not obj or not obj.get("references"):
            return dict(accepted=False, reason="unregistered_target", query=query, ranked_candidates=[])
        with self._lock:
            gallery_cache = self._load_features()
            source_hash = image_digest(image_path)
            key = (source_hash, tuple(tuple(c["bbox"]) for c in candidates), self._feature_key)
            frame_hit = key in self._frame_features
            if frame_hit:
                features = self._frame_features.pop(key)
            else:
                with Image.open(image_path) as im:
                    images = [crop_image(im, c["bbox"]) for c in candidates]
                features = self._encode(images) if images else np.empty((0, self._features.shape[1]))
            self._frame_features[key] = features
            while len(self._frame_features) > self.settings.cache_frames:
                self._frame_features.popitem(last=False)
            excluded = set(exclude_groups)
            valid = [i for i, (_, r) in enumerate(self._refs)
                     if r.get("source_group") not in excluded and r.get("source_image_sha256") != source_hash]
            names = sorted({self._refs[i][0] for i in valid})
            rows = []
            for idx, vector in enumerate(features):
                similarities = self._features @ vector
                scores = {n: float(np.mean(sorted([float(similarities[i]) for i in valid
                    if self._refs[i][0] == n], reverse=True)[:self.settings.top_k_references])) for n in names}
                ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                score = scores.get(obj["name"], -1.0)
                rival = max((s for n, s in scores.items() if n != obj["name"]), default=0.0)
                margin = score - rival
                accepted = bool(ranked and ranked[0][0] == obj["name"]
                    and score >= self.settings.min_similarity and margin >= self.settings.min_class_margin)
                rows.append(dict(index=idx, object_id=candidates[idx].get("object_id", f"obj_{idx+1}"),
                    bbox=candidates[idx]["bbox"], score=score, class_margin=margin,
                    predicted_identity=ranked[0][0] if ranked else None,
                    class_scores=scores, accepted=accepted, instance=copy.deepcopy(candidates[idx])))
            rows.sort(key=lambda r: (r["accepted"], r["score"]), reverse=True)
            chosen = rows[0] if rows and rows[0]["accepted"] else None
            return dict(accepted=chosen is not None, reason="reference_match" if chosen else "reference_uncertain",
                query=query, identity=obj["name"], identity_id=obj['object_id'], category=obj["category"], source="reference_gallery",
                score=chosen["score"] if chosen else (rows[0]["score"] if rows else 0.0),
                class_margin=chosen["class_margin"] if chosen else None,
                selected_index=chosen["index"] if chosen else None,
                ranked_candidates=rows, min_similarity=self.settings.min_similarity,
                min_class_margin=self.settings.min_class_margin, score_kind="cosine_similarity_not_probability",
                gallery_revision=self._feature_key, image_sha256=source_hash,
                gallery_cache=gallery_cache, frame_features_cache_hit=frame_hit,
                elapsed_s=time.perf_counter()-started)


def build_reference_service(config):
    from hardware.config import load_config
    from services.runtime_service_cache import get_or_create_service
    path = getattr(config, "hardware_config_path", None)
    if not path:
        return None
    reference_config = load_config(path).perception.reference_config
    if not reference_config:
        return None
    settings = load_settings(reference_config)
    if not settings.enabled:
        return None
    return get_or_create_service("object_references", asdict(settings), lambda: ObjectReferenceService(settings))
