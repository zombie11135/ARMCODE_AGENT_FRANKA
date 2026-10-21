"""Camera views used by a task; device construction lives in runtime.services."""
from __future__ import annotations

from runtime.observation_memory import memory_for


class TaskCachedCamera:
    """
    Legacy opt-in task snapshots; calibrated FR3 delegates to pose-aware memory.

    Initial task snapshots have no freshness or pose checks of their own. They
    are never used for a camera with calibrated observation memory, including
    when its policy is disabled (which means live captures).
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
        if memory_for(self.camera) is not None:
            return self.camera.capture_global()
        if self.enabled and self.cached_global is not None:
            return dict(self.cached_global)
        return self.camera.capture_global()

    def capture_wrist(self):
        if memory_for(self.camera) is not None:
            return self.camera.capture_wrist()
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
