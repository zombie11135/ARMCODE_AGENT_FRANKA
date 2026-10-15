"""Compare physical jaw directions without planning or sending motion.

TCP axes are DEFINED parallel to flange axes; the straight tool approaches +Fz.
GSNet Gx approaches the object and Gy spans the jaws. R_G_TCP is derived from
the observed jaw direction, not a guessed extra Euler-angle correction.
"""
import numpy as np

from .transforms import pose_matrix


def _unit_direction(value):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or not np.any(vector):
        raise ValueError("Expected a finite nonzero jaw direction")
    # Scale first to avoid overflow for otherwise finite input vectors.
    vector = vector / np.max(np.abs(vector))
    return vector / np.linalg.norm(vector)


def rotation_grasp_to_tcp(closing_axis_flange):
    closing = _unit_direction(closing_axis_flange)
    approach = np.array([0., 0., 1.])
    if abs(float(closing @ approach)) > 1e-6:
        raise ValueError("For this straight mount, jaw direction must lie in the flange XY plane")
    # Each column is a GSNet axis expressed in the flange frame.
    flange_R_grasp = np.column_stack((approach, closing, np.cross(approach, closing)))
    # R_F_TCP = I by our TCP frame definition, so R_G_TCP = inverse(R_F_G).
    return flange_R_grasp.T


def axes_report(flange_pose, transforms, closing_axis_flange=None, *, closing_axis_base=None):
    if closing_axis_flange is not None and closing_axis_base is not None:
        raise ValueError("Supply jaw direction in either base or flange coordinates, not both")
    base_T_flange = pose_matrix(flange_pose)
    rotation = base_T_flange[:3, :3]
    tcp = transforms.flange_T_tcp
    if tcp is None or not np.allclose(tcp[:3, :3], np.eye(3), atol=1e-8, rtol=0):
        raise ValueError("Axes comparison expects configured TCP axes parallel to the flange")

    def candidate(direction):
        vector = _unit_direction(direction)
        align = rotation_grasp_to_tcp(vector)
        return dict(jaw_direction_flange=vector.tolist(),
                    jaw_direction_base=(rotation @ vector).tolist(),
                    R_grasp_tcp=align.tolist())

    result = dict(
        base_frame="fr3_link0", flange_frame="fr3_link8", flange_pose_xyz_rpy=flange_pose.values,
        unit="m/rad; direction vectors are unitless",
        tcp_axes="parallel to flange axes by definition",
        tcp_position_base_m=(base_T_flange @ tcp)[:3, 3].tolist(),
        approach_direction_base=rotation[:, 2].tolist(),
        flange_axes_in_base={axis: rotation[:, i].tolist() for i, axis in enumerate(("X", "Y", "Z"))},
        candidates={"jaw_parallel_to_flange_X": candidate([1., 0., 0.]),
                    "jaw_parallel_to_flange_Y": candidate([0., 1., 0.])},
        note="Compare the line joining the two fingers, not finger length. X/Y candidates are alternatives, not detected results. Opposite vector signs describe the same line but may give different robot poses. Diagonal mounting is also possible.",
        configuration_written=False, motion_command_sent=False,
    )
    if closing_axis_flange is not None:
        result["supplied_direction_candidate"] = candidate(closing_axis_flange)
    if closing_axis_base is not None:
        observed_base = _unit_direction(closing_axis_base)
        observed_flange = rotation.T @ observed_base
        plane_norm = float(np.linalg.norm(observed_flange[:2]))
        if plane_norm < 1e-8:
            raise ValueError("Observed jaw line is parallel to tool approach; cannot determine the jaw axis")
        # Human descriptions may be approximate. Expose the projection error;
        # never silently treat a tilted observation as an exact calibration.
        projected = np.array([observed_flange[0]/plane_norm, observed_flange[1]/plane_norm, 0.])
        error = float(np.degrees(np.arctan2(abs(observed_flange[2]), plane_norm)))
        result["observed_base_line"] = dict(
            direction_base=observed_base.tolist(),
            direction_flange_unprojected=observed_flange.tolist(),
            projected_direction_flange=projected.tolist(),
            deviation_from_flange_xy_plane_deg=error,
            line_angle_to_flange_X_deg=float(np.degrees(np.arccos(np.clip(abs(projected[0]), 0, 1)))),
            line_angle_to_flange_Y_deg=float(np.degrees(np.arccos(np.clip(abs(projected[1]), 0, 1)))),
            sign_ambiguous_candidates=[candidate(projected), candidate(-projected)],
            note="The observed line is undirected and must match the supplied/read flange pose. Nonzero plane deviation indicates an approximate observation or incompatible straight-mount assumption. Candidates are not applied automatically.",
        )
    return result
