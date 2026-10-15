"""Camera views used by a task; device construction lives in runtime.services."""
from __future__ import annotations


class TaskCachedCamera:
    """
    One-task image cache used by the simple runtime.

    The first home-view global/wrist captures are often the cleanest view of
    containers and medicines. The runtime enables this cache only for the first
    attempt of container pre-observation and medicine grasp localization, then
    falls back to live captures if that attempt fails.
    """

    def __init__(self, camera):
        self.camera = camera
        self.cached_global = None
        self.cached_wrist = None
        self.enabled = False

    def set_task_captures(self, global_capture=None, wrist_capture=None):
        self.cached_global = dict(global_capture or {}) if global_capture else None
        self.cached_wrist = dict(wrist_capture or {}) if wrist_capture else None

    def enable_cache(self):
        self.enabled = True

    def disable_cache(self):
        self.enabled = False

    def capture_global(self):
        if self.enabled and self.cached_global is not None:
            return dict(self.cached_global)
        return self.camera.capture_global()

    def capture_wrist(self):
        if self.enabled and self.cached_wrist is not None:
            return dict(self.cached_wrist)
        return self.camera.capture_wrist()

    def capture(self, camera_name: str):
        if camera_name == "global":
            return self.capture_global()
        if camera_name == "wrist":
            return self.capture_wrist()
        return self.camera.capture(camera_name)

    def __getattr__(self, name):
        return getattr(self.camera, name)
