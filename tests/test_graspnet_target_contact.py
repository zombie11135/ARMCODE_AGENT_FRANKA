"""Contact-filter regressions for buried fingers and incomplete target evidence."""
import numpy as np
from hardware.graspnet_target_contact import filter_contacts, jaw_clearance, mask_membership, target_surface


def grasp(width=.085):
    return np.array([.9, width, .02, .02, *np.eye(3).ravel(), 0., 0., 0., -1.])


def surface(low=-.03, high=.03):
    return np.array([[x, y, z] for x in (-.005, 0., .005)
                     for y in np.linspace(low, high, 41) for z in (-.004, .004)])


def test_single_finger_collision_is_not_diluted_by_whole_sweep_volume():
    rows = np.array([grasp(), grasp()])
    parts = np.array([[.002, .002], [.104, 0.], [0., 0.], [0., 0.], [0., 0.]])
    finger, target, records = filter_contacts(rows, parts, target=surface())
    assert finger.tolist() == [False, True]
    assert target.all()
    assert not records[0]['accepted']
    assert records[1]['accepted']


def test_missing_side_surface_cannot_hide_finger_inside_target():
    # No samples near the left finger itself, yet the section extends past it.
    points = surface(-.055, .02)
    points = points[(points[:, 1] < -.05) | (points[:, 1] > -.035)]
    finger, target, records = filter_contacts(np.array([grasp()]), np.zeros((5, 1)), target=points)
    assert finger[0]
    assert not target[0]
    assert 'target_not_between_open_fingers' in records[0]['target_section']['reasons']


def test_wider_object_is_not_made_graspable_by_clamping_width():
    evidence = jaw_clearance(grasp(), surface(-.048, .048), finger_length_m=.06)
    assert not evidence['accepted']
    assert evidence['required_opening_at_current_center_m'] > .085


def test_centered_local_section_passes_and_remote_object_parts_do_not_count():
    points = np.vstack([surface(), surface(-.10, .10) + [0., 0., .08]])
    result = jaw_clearance(grasp(), points, finger_length_m=.06)
    assert result['accepted']
    assert result['left_clearance_m'] > .01


def test_off_center_or_insufficient_observation_rejected():
    result = jaw_clearance(grasp(), surface(.005, .035), finger_length_m=.06)
    assert 'target_not_straddling_grasp_center' in result['reasons']
    result = jaw_clearance(grasp(), surface()[:5], finger_length_m=.06)
    assert result['reasons'] == ['insufficient_target_section']


def test_contact_geometry_is_invariant_under_camera_rigid_transform():
    row = grasp(); points = surface()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([.2, .3, .7])
    moved = row.copy(); moved[4:13] = rotation.ravel(); moved[13:16] = translation
    before = jaw_clearance(row, points, finger_length_m=.06)
    after = jaw_clearance(moved, points @ rotation.T + translation, finger_length_m=.06)
    assert after['accepted'] == before['accepted']
    np.testing.assert_allclose(after['target_y_bounds_m'], before['target_y_bounds_m'])


def test_sparse_outlier_does_not_define_object_boundary():
    points = np.vstack([surface(), [[0., -.2, 0.]]])
    assert jaw_clearance(grasp(), points, finger_length_m=.06)['accepted']


def test_target_extraction_excludes_other_objects_table_and_invalid_projection():
    intr = dict(fx=100., fy=100., ppx=2., ppy=2.)
    mask = np.zeros((5, 5), bool); mask[2, 2] = True
    points = np.array([[0., 0., 1.], [.1, 0., 1.], [0., 0., 0.], [np.nan, 0., 1.], [0., 0., .005]])
    assert mask_membership(points, mask, intr).tolist() == [True, False, False, False, True]
    np.testing.assert_allclose(target_surface(points, mask, intr, np.eye(4), 0.), [[0., 0., 1.]])


def test_no_mask_still_checks_fingers_and_empty_pool_is_not_filled():
    finger, target, records = filter_contacts(np.array([grasp()]), np.array([[0.], [.1], [0.], [0.], [0.]]))
    assert not finger[0] and target[0]
    assert 'target_section' not in records[0]
    finger, target, records = filter_contacts(np.empty((0, 17)), np.empty((5, 0)), target=np.empty((0, 3)))
    assert len(finger) == len(target) == len(records) == 0
