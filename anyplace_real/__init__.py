"""Offline-only AnyPlace stage-1 integration helpers.

This package deliberately contains no robot or gripper command path.  It turns
recorded RGB-D into point clouds, invokes the upstream AnyPlace inference entry,
and exports a reviewable object-motion proposal.
"""

__all__ = ["geometry"]

