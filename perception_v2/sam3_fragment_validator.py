"""Two independent point-prompt decodes sharing one local image encoding."""
from contextlib import nullcontext
import time

import cv2
import numpy as np
from runtime.perception_pipeline import serialized_resource


def interior_point(mask):
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(distances), mask.shape)
    return [float(x), float(y)]


class SAM3FragmentValidator:
    def __init__(self, backend):
        self.backend = backend

    @serialized_resource('gpu')
    def __call__(self, image, masks, negatives):
        from PIL import Image
        backend = self.backend
        torch = backend.torch
        started = time.perf_counter()
        union = masks[0] | masks[1]
        yy, xx = np.nonzero(union)
        x0, y0 = max(0, int(xx.min())-24), max(0, int(yy.min())-24)
        x1, y1 = min(image.shape[1], int(xx.max())+25), min(image.shape[0], int(yy.max())+25)
        local = [m[y0:y1, x0:x1] for m in masks]
        neg = negatives[y0:y1, x0:x1] & ~union[y0:y1, x0:x1]
        count, components, stats, _ = cv2.connectedComponentsWithStats(neg.astype(np.uint8), 8)
        neg_points = [interior_point(components == i) for i in
                      sorted(range(1, count), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:4]
                      if stats[i, cv2.CC_STAT_AREA] >= 16]
        box = np.array([[xx.min()-x0, yy.min()-y0, xx.max()+1-x0, yy.max()+1-y0]], np.float32)
        autocast = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                    if backend.device.startswith('cuda') else nullcontext())
        outputs, prompts, scores = [], [], []
        with torch.inference_mode(), autocast:
            state = backend.processor.set_image(Image.fromarray(cv2.cvtColor(image[y0:y1,x0:x1], cv2.COLOR_BGR2RGB)))
            for fragment in local:
                positive = interior_point(fragment)
                coordinates = np.array([positive, *neg_points], dtype=np.float32)
                labels = np.array([1, *([0]*len(neg_points))], dtype=np.int32)
                predicted, quality, _ = backend.model.predict_inst(state,
                    point_coords=coordinates, point_labels=labels, box=box, multimask_output=True)
                quality = np.asarray(quality).reshape(-1)
                # Choose by the model's own score, never by how well it agrees
                # with the proposed association we are trying to test.
                chosen = np.asarray(predicted).reshape(-1,y1-y0,x1-x0)[int(np.argmax(quality))].astype(bool)
                full = np.zeros(union.shape, bool);full[y0:y1,x0:x1] = chosen
                outputs.append(full);scores.append(float(np.max(quality)))
                prompts.append(dict(positive_xy=[positive[0]+x0, positive[1]+y0],
                    negative_xy=[[p[0]+x0,p[1]+y0] for p in neg_points]))
        return dict(masks=outputs, scores=scores, prompts=prompts, crop_xyxy=[x0,y0,x1,y1],
                    image_encodings=1, decoder_calls=2, elapsed_s=time.perf_counter()-started)
