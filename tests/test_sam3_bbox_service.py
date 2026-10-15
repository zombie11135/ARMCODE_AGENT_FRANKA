"""Box-prompt API contracts, without loading weights or accessing devices."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from hardware.config import DEFAULT_CONFIG
from services.grasp_service import build_segmenter
from services.sam3_bbox_segmenter_service import SAM3BBoxSegmenterService


class SAM3BoxTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.image = self.root / 'input.png'
        cv2.imwrite(str(self.image), np.full((60, 80, 3), 180, np.uint8))
        self.service = SAM3BBoxSegmenterService.__new__(SAM3BBoxSegmenterService)
        self.service.model = Mock()
        self.service.processor = Mock()
        self.service.processor.set_image.return_value = {'image': 'fresh'}
        self.service._autocast_context = nullcontext
        self.mask = np.zeros((1, 60, 80), bool)
        self.mask[:, 10:40, 20:50] = True
        self.scores = np.array([.95])

    def test_fr3_uses_instance_prompts_legacy_keeps_geometric(self):
        self.assertEqual(build_segmenter(SimpleNamespace(hardware_config_path=str(DEFAULT_CONFIG))).prompt_mode, 'instance')
        self.assertEqual(build_segmenter(SimpleNamespace()).prompt_mode, 'geometric')

    def test_instance_receives_pixel_xyxy_and_preserves_full_resolution_mask(self):
        self.service.prompt_mode = 'instance'
        self.service.model.predict_inst.return_value = self.mask, self.scores, None
        result = self.service.segment_by_bbox(str(self.image), [20, 10, 50, 40], str(self.root / 'instance'))
        call = self.service.model.predict_inst.call_args
        np.testing.assert_array_equal(call.kwargs['box'], [[15, 5, 55, 45]])
        self.assertFalse(call.kwargs['multimask_output'])
        self.assertIsNone(call.kwargs['point_coords'])
        self.service.processor.add_geometric_prompt.assert_not_called()
        np.testing.assert_array_equal(np.load(result['mask_npy_path']) > 0, self.mask[0])
        self.assertEqual(result['mask_area'], 900)
        self.assertEqual(result['prompt_mode'], 'instance')

    def test_legacy_geometric_receives_normalized_cxcywh(self):
        self.service.prompt_mode = 'geometric'
        self.service.processor.add_geometric_prompt.return_value = {
            'masks': self.mask, 'scores': self.scores, 'boxes': np.array([[20, 10, 50, 40]])}
        result = self.service.segment_by_bbox(str(self.image), [20, 10, 50, 40], str(self.root / 'geometric'))
        np.testing.assert_allclose(self.service.processor.add_geometric_prompt.call_args.kwargs['box'],
                                   [35/80, 25/60, 40/80, 40/60])
        self.service.model.predict_inst.assert_not_called()
        self.assertEqual(result['mask_area'], 900)

    def test_instance_without_interactive_decoder_is_rejected_before_load(self):
        with self.assertRaisesRegex(ValueError, 'enable_inst_interactivity'):
            SAM3BBoxSegmenterService(prompt_mode='instance', enable_inst_interactivity=False)


if __name__ == '__main__':
    unittest.main()
