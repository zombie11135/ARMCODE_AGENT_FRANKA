"""Small adapter for the official pre/post July 2026 AnyGrasp detector APIs.

Imported only by inference workers. Keeps full-scene collision input and returns
raw GraspGroup candidates; downstream FR3 filtering and selection stay unchanged.
"""
from importlib import import_module
import math


class AnyGrasp:
    def __init__(self, options):
        self.options = options
        self.sdk = import_module('gsnet')
        self.sdk_api = 'factory_20260704' if hasattr(self.sdk, 'create_detector') else 'legacy'
        self.detector = None

    def load_net(self):
        if self.sdk_api == 'legacy':
            self.detector = self.sdk.AnyGrasp(self.options)
            self.detector.load_net()
        else:
            self.detector = self.sdk.create_detector(self.options)
            if self.detector is None:
                raise RuntimeError('AnyGrasp detector initialization failed; check the SDK license diagnostics above')

    def get_grasp(self, points, colors, *, lims, apply_object_mask=True,
                  dense_grasp=False, collision_detection=True):
        if self.sdk_api == 'legacy':
            return self.detector.get_grasp(points, colors, lims=lims,
                apply_object_mask=apply_object_mask, dense_grasp=dense_grasp,
                collision_detection=collision_detection)
        if not apply_object_mask:
            raise ValueError('New AnyGrasp SDK has no apply_object_mask=False option; use its explicit region steering API')
        # Existing workers pass bounds of the whole input cloud. Preserve those
        # points for collision filtering; ROI/SAM membership is checked afterward.
        params = dict(dense_grasp=dense_grasp, collision_detection=collision_detection)
        if self.options.top_down_grasp:
            params.update(approach_steering=[0, 0, 1], approach_thresh=math.pi/6)
        group = self.detector.get_grasp(points, params)
        if group is None:
            from graspnetAPI import GraspGroup
            group = GraspGroup()  # Official API returns None when no candidate survives.
        return group, None
