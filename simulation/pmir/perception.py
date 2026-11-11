"""Known-color, axis-aligned boxes on a known support plane; no neural model.

Only copied RGB-D/intrinsics/extrinsics enter these functions. Simulator object
poses are deliberately excluded. This is a controlled perception baseline.
"""
import time
import numpy as np
from scipy import ndimage


def geometry(camera):
    depth, k = camera['depth_m'], camera['K']
    y, x = np.indices(depth.shape)
    optical = np.stack(((x-k[0, 2])*depth/k[0, 0], (y-k[1, 2])*depth/k[1, 1], depth), axis=-1)
    transform = camera['T_world_from_camera_optical']
    return optical @ transform[:3, :3].T + transform[:3, 3]


def mask_for(camera, target, table_height=.8):
    rgb = camera['rgb'].astype(float)
    r, g, b = np.moveaxis(rgb, -1, 0)
    if target == 'red_cube':
        mask = (r > 70) & (r > 1.6*g) & (r > 1.6*b)
    elif target == 'blue_cube':
        mask = (b > 70) & (b > 1.6*r) & (b > 1.4*g)
    else:
        raise ValueError(f'No color model for {target}')
    world = geometry(camera)
    # Exclude colored goal markers on the table and lifted objects. Only resting
    # boxes belong to this first experiment's perception/grasp contract.
    mask &= (world[..., 2] > table_height+.008) & (world[..., 2] < table_height+.06)
    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == sizes.argmax()
    return mask, world


def estimate(snapshot, target, *, delay_s=0., table_height=.8, half_size=.02):
    mask, world = mask_for(snapshot['camera'], target, table_height)
    pixels = int(mask.sum())
    injected_start = time.perf_counter()
    if delay_s:
        time.sleep(delay_s)  # Recorded latency injection, never called model time.
    injected_end = time.perf_counter()
    result = dict(_timing={'injected_wait': [injected_start, injected_end]}, success=pixels >= 25, visible_pixels=pixels, source='rgbd_color_known_box_geometry',
                  read_versions=dict(snapshot['read_versions']), snapshot_id=snapshot['snapshot_id'])
    if pixels < 25:
        return result
    points = world[mask]
    low, high = np.quantile(points, [.02, .98], axis=0)
    center = (low+high)/2
    eye = snapshot['camera']['T_world_from_camera_optical'][:3, 3]
    # A near frontal image exposes only one face on the viewing axis. Known box
    # size supplies its centre offset; this is not general 6D pose estimation.
    for axis in (0, 1):
        if high[axis]-low[axis] < half_size:
            center[axis] += np.sign(center[axis]-eye[axis])*half_size
    center[2] = table_height+half_size
    result['position_m'] = center.tolist()
    return result


def compute(kind, target, inputs, settings):
    if kind == 'PERCEIVE':
        return estimate(inputs[0], target, delay_s=settings['compute_delay_s'])
    if kind == 'GRASP_PLAN':
        evidence = inputs[0]
        if not evidence['success']:
            raise ValueError(f'No visible grasp evidence for {target}')
        return dict(evidence, grasp_type='fixed_vertical_known_box',
                    collision_free_trajectory_verified=False)
    raise ValueError(f'Unsupported compute kind: {kind}')
