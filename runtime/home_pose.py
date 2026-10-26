"""One FR3 home per CLI session/profile; no devices or disk persistence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HomePoseSession:
    profile: str | None = None
    pose_mrad: tuple | None = None
    source: str | None = None

    def known_pose(self, profile):
        if self.profile == str(Path(profile).resolve()) and self.pose_mrad is not None:
            return list(self.pose_mrad)
        return None

    def resolve(self, profile, configured_pose, read_current_pose):
        """Return (pose, newly_recorded); never replace home after a task fails."""
        from hardware.interfaces import Pose6D
        known = self.known_pose(profile)
        if known is not None:
            return known, False
        source = 'hardware_yaml' if configured_pose is not None else 'first_actual_flange_pose'
        values = configured_pose if configured_pose is not None else read_current_pose()
        pose = Pose6D(*values).values
        self.profile = str(Path(profile).resolve())
        self.pose_mrad, self.source = tuple(pose), source
        return list(pose), True


def home_pose_session(owner):
    session = getattr(owner, 'home_pose_session', None)
    if session is None:
        session = HomePoseSession()
        owner.home_pose_session = session
    return session
