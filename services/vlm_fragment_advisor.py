"""Bounded visual advice for geometry-compatible fragments; never merges masks."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from services.object_reference_service import atomic_json


class VLMFragmentAdvisor:
    def __init__(self, client, max_entries=32):
        self.client = client
        self.max_entries = max_entries
        self.cache = OrderedDict()

    def assess(self, image_path, masks, context, output_dir):
        started = time.perf_counter()
        regions = []
        for mask in masks:
            y, x = np.nonzero(mask)
            distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            py, px = np.unravel_index(np.argmax(distance), mask.shape)
            regions.append(dict(bbox_xyxy=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)],
                                interior_pixel_xy=[int(px), int(py)]))
        image_hash = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        signature = hashlib.sha256(b''.join(np.packbits(m).tobytes() for m in masks)).hexdigest()
        context = context or {}
        # Reuse only identical pixels/masks in the same explicitly recorded scene.
        cacheable = context.get('scope') is not None and context.get('scene_epoch') is not None
        key = (context.get('scope'), context.get('scene_epoch'), image_hash, signature)
        prompt = '''判断原始照片中以下 A、B 两个可见区域是否属于同一个物理物体。
图1是原图，图2是A区域的原始裁剪，图3是B区域的原始裁剪；均未画掩膜或箭头。
坐标为原图像素，bbox 是 xyxy，interior_pixel_xy 位于碎片内部。
裁剪可能包含其他遮挡物，要判断的是对应 interior_pixel_xy 的表面，结合原图和两个裁剪判断。
看表面纹理和轮廓连续性；包装颜色或类别相同不等于同一物体，上下叠放也不等于同一物体。
不要求识别药品名称。遮挡使证据不足时返回 uncertain，不猜测不可见部分。
只输出 JSON：{"relation":"same_object|different_objects|uncertain","reason":"简短可见证据"}。
区域数据：'''+json.dumps(dict(A=regions[0], B=regions[1]), ensure_ascii=False)
        record = dict(schema='vlm_fragment_advice_v1', image_sha256=image_hash,
                      regions=regions, context=context, model=getattr(self.client, 'model_name', None),
                      advisory_only=True, mask_merge_authorized=False)
        try:
            if cacheable and key in self.cache:
                result = dict(self.cache[key])
                record['cache_hit'] = True
                self.cache.move_to_end(key)
            else:
                color = cv2.imread(str(image_path))
                input_dir = Path(output_dir)/'inputs'
                input_dir.mkdir(parents=True, exist_ok=True)
                images = [str(image_path)]
                for label, region in zip(('A', 'B'), regions):
                    x0, y0, x1, y1 = region['bbox_xyxy']
                    crop_path = input_dir/f'{image_hash[:12]}_{signature[:12]}_{label}.png'
                    if not cv2.imwrite(str(crop_path), color[y0:y1, x0:x1]):
                        raise OSError('Unable to save original fragment crop')
                    images.append(str(crop_path))
                record['image_paths'] = images
                raw = self.client.describe_images(images, prompt, max_tokens=300, temperature=0.)
                record['raw_response'] = raw
                clean = raw.strip()
                if clean.startswith('```') and clean.endswith('```'):
                    clean = clean.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                result = json.loads(clean)
                if (not isinstance(result, dict) or set(result) != {'relation', 'reason'}
                        or result['relation'] not in ('same_object', 'different_objects', 'uncertain')
                        or not isinstance(result['reason'], str) or not result['reason'].strip()):
                    raise ValueError('Invalid fragment advice')
                record['cache_hit'] = False
                if cacheable:
                    self.cache[key] = dict(result)
                    while len(self.cache) > self.max_entries:
                        self.cache.popitem(last=False)
            record.update(result, status='ok')
        except Exception as error:
            # Optional remote advice must not discard usable RGB-D geometry.
            record.update(relation='uncertain', reason='VLM advice unavailable', status='unavailable',
                          error_type=type(error).__name__, cache_hit=False)
        record['elapsed_s'] = time.perf_counter()-started
        path = Path(output_dir)/(uuid.uuid4().hex+'.json')
        atomic_json(path, record)
        return {k: record[k] for k in ('relation', 'reason', 'status', 'cache_hit', 'elapsed_s')} | {
            'trace_path': str(path.resolve()), 'advisory_only': True, 'mask_merge_authorized': False}
