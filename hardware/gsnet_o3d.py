"""Read-only Open3D comparison of saved GSNet, calibrated tool and feedback.

Only recorded files are read. No camera, robot, gripper or current calibration
is accessed. All geometry uses fr3_link0, metres, at the saved capture time.
"""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import sys

import numpy as np

from .interfaces import Pose6D
from .transforms import pose_matrix, rigid_matrix

GREEN = [0.1, .9, .25]
ORANGE = [1., .5, .05]
BLUE = [.1, .65, 1.]
MAGENTA = [1., .15, .8]

HELP = """Open3D 对比（所有数据在 fr3_link0 中，单位 m）：
  绿色：最终选中的 GSNet 抓取；通用平行夹爪示意，不是 Robotiq CAD
  橙色：抓取目标处的整套工具碰撞包络，含配置中的 padding
  蓝色：预抓取参考位置；连接线仅表示目标间关系，不是规划轨迹
  紫色：日志中的实际抓取到达位置/工具包络（有执行记录时）
  坐标轴：红 X、绿 Y、蓝 Z；GSNet X=接近，Y=开合；TCP 轴按保存的标定
快捷键：G 抓取 / E 工具包络 / P 预抓取 / O 实际到达 / T 桌面网格
        A 切换 TCP+法兰坐标轴、GSNet 坐标轴、隐藏坐标轴
        C 相机观察视角 / B 基座斜视角 / S 保存截图
鼠标拖动旋转、滚轮缩放。Q / Esc / 关闭窗口结束本次查看。
点云始终是拍摄时的静态场景；实际到达数据来自机器人 FK 反馈，不是视觉测量。
"""


def read_json(path):
    return json.loads(Path(path).read_text())


def transform_points(points, transform):
    points = np.asarray(points, dtype=float)
    return points @ transform[:3, :3].T + transform[:3, 3]


def load_scene(root, robot_log=None):
    root = Path(root).expanduser().resolve()
    workflow = read_json(root / 'run.json') if (root / 'run.json').is_file() else {}
    if (root / 'selection.json').is_file():
        ranked = read_json(root / 'selection.json')
        observation = Path(ranked['source_capture'])
        selected, calibration, workcell = ranked['selected'], ranked['calibration'], ranked['workcell']
        flange = pose_matrix(Pose6D(*ranked['flange_xyz_rpy']))
        geometry = None
    else:
        observation = root / 'observation' if (root / 'observation').is_dir() else root
        geometry = read_json(observation / 'targets.json')
        selected, calibration, workcell = geometry['selected'], geometry['calibration'], geometry['workcell']
        flange = pose_matrix(Pose6D(*geometry['targets']['grasp']['flange_xyz_rpy']))
    record = read_json(observation / 'capture_pose.json')
    meta = read_json(observation / 'capture/capture.json')
    if record['calibration'] != calibration:
        raise ValueError('Capture and target calibration snapshots differ')
    camera = calibration['cameras']['wrist']
    if (meta['device']['serial_number'] != camera['serial'] or
            meta['frame'] != f"realsense_{camera['serial']}_color_optical"):
        raise ValueError('Saved point cloud is not the calibrated wrist COLOR frame')
    base_camera = (pose_matrix(Pose6D(*record['before']['pose'])) @
                   rigid_matrix(camera['T_flange_camera'], 'flange_T_camera'))
    base_grasp = base_camera @ rigid_matrix(selected['T_input_grasp'], 'camera_T_grasp')
    flange_tcp = rigid_matrix(calibration['T_flange_tcp'], 'flange_T_tcp')
    grasp_tcp = rigid_matrix(calibration['T_grasp_tcp'], 'grasp_T_tcp')
    base_tcp = flange @ flange_tcp
    if not np.allclose(base_tcp, base_grasp @ grasp_tcp, atol=1e-6, rtol=0):
        raise ValueError('Saved grasp and flange target disagree; refusing misleading overlay')
    if geometry is not None:
        pre_flange = pose_matrix(Pose6D(*geometry['targets']['pregrasp']['flange_xyz_rpy']))
    else:
        pre_flange = flange.copy()
        pre_flange[:3, 3] -= .08 * base_grasp[:3, 0]
    actual = []
    for step in workflow.get('steps', []):
        if step['stage'] not in ('move_pregrasp', 'move_grasp', 'lift'):
            continue
        result = step.get('result') or {}
        endpoint = result.get('endpoint', {})
        if 'actual_model_pose_xyz_rpy' in endpoint:
            actual.append({'stage': step['stage'], 'pose': endpoint['actual_model_pose_xyz_rpy'],
                           'position_error_m': endpoint.get('position_error_m'),
                           'orientation_error_rad': endpoint.get('orientation_error_rad'),
                           'execution_ok': result.get('ok', False)})
    if robot_log is not None:
        events = [json.loads(s) for s in Path(robot_log).expanduser().read_text().splitlines() if s.strip()]
        plans = [e for e in events if e.get('event') == 'plan']
        endpoints = [e for e in events if e.get('event') == 'endpoint']
        if not plans or not endpoints:
            raise ValueError('Comparison log needs a plan and an actual endpoint')
        requested = pose_matrix(Pose6D(*plans[-1]['summary']['target_pose_xyz_rpy']))
        if not np.allclose(requested, flange, atol=1e-6, rtol=0):
            raise ValueError('Robot comparison log targets a different grasp pose')
        endpoint = endpoints[-1]
        actual.append({'stage': 'move_grasp', 'pose': endpoint['actual_model_pose_xyz_rpy'],
                       'position_error_m': endpoint.get('position_error_m'),
                       'orientation_error_rad': endpoint.get('orientation_error_rad'),
                       'source_log': str(Path(robot_log).expanduser().resolve())})
    return dict(root=root, cloud_path=observation / 'capture/scene.ply', selected=selected,
                workcell=workcell, base_camera=base_camera, base_grasp=base_grasp,
                base_flange=flange, base_tcp=base_tcp, flange_tcp=flange_tcp,
                pre_flange=pre_flange, actual=actual)


def build_geometry(scene):
    import open3d as o3d
    from graspnetAPI import Grasp

    cloud = o3d.io.read_point_cloud(str(scene['cloud_path']))
    if len(cloud.points) == 0:
        raise ValueError('Saved point cloud is empty')
    cloud.transform(scene['base_camera'])
    # Visualization only; inference and collision input remain full resolution.
    cloud = cloud.voxel_down_sample(.002)

    def frame(transform, size):
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size).transform(transform)

    def line(points, pairs, color):
        lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points),
                                     o3d.utility.Vector2iVector(pairs))
        lines.paint_uniform_color(color)
        return lines

    def marker(point, color, radius=.004):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh.translate(point)

    def envelope(transform, color):
        cell = scene['workcell']
        width = cell['tool_max_width_m'] + 2 * cell['tool_padding_m']
        length = cell['tool_length_m'] + cell['tool_padding_m']
        corners = np.array(list(itertools.product((-width/2, width/2),
                                                   (-width/2, width/2), (0., length))))
        edges = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]
        return line(transform_points(corners, transform), edges, color)

    selected, grasp = scene['selected'], scene['base_grasp']
    row = np.r_[selected['score'], selected['width_m'], selected['height_m'], selected['depth_m'],
                grasp[:3, :3].ravel(), grasp[:3, 3], -1.]
    gripper = Grasp(row).to_open3d_geometry(color=GREEN)
    gripper.compute_vertex_normals()
    pre_tcp = scene['pre_flange'] @ scene['flange_tcp']
    groups = {'cloud': [cloud], 'grasp': [gripper, marker(grasp[:3, 3], GREEN)],
              'envelope': [envelope(scene['base_flange'], ORANGE)],
              'tcp_axes': [frame(scene['base_tcp'], .04), frame(scene['base_flange'], .06)],
              'grasp_axes': [frame(grasp, .04)],
              'pregrasp': [marker(pre_tcp[:3, 3], BLUE),
                          line([pre_tcp[:3, 3], scene['base_tcp'][:3, 3]], [[0, 1]], BLUE)],
              'actual': [], 'table': []}
    for actual in scene['actual']:
        actual_flange = pose_matrix(Pose6D(*actual['pose']))
        actual_tcp = actual_flange @ scene['flange_tcp']
        groups['actual'].append(marker(actual_tcp[:3, 3], MAGENTA))
        if actual['stage'] == 'move_grasp':
            groups['actual'].append(envelope(actual_flange, MAGENTA))
    # A finite grid illustrates the infinite table; it is not new collision geometry.
    points, pairs = [], []
    x, y = scene['base_tcp'][:2, 3]
    z = scene['workcell']['table_z_m']
    for d in np.linspace(-.3, .3, 13):
        for segment in ([(x+d, y-.3, z), (x+d, y+.3, z)],
                        [(x-.3, y+d, z), (x+.3, y+d, z)]):
            start = len(points); points.extend(segment); pairs.append([start, start+1])
    groups['table'] = [line(points, pairs, [.45, .45, .45])]
    return groups


def view(root, *, robot_log=None, screenshot=None, check_only=False, continuation='offline'):
    scene = load_scene(root, robot_log)
    groups = build_geometry(scene)
    output = scene['root'] / 'o3d'
    output.mkdir(exist_ok=True)
    summary = {'base_frame': 'fr3_link0', 'units': 'm', 'cloud': str(scene['cloud_path']),
               'score': scene['selected']['score'], 'width_m': scene['selected']['width_m'],
               'displayed_points': len(groups['cloud'][0].points), 'actual_endpoints': scene['actual'],
               'no_hardware_access': True, 'guide_line_is_trajectory': False,
               **{key: scene[key].tolist() for key in
                  ('base_camera', 'base_grasp', 'base_flange', 'base_tcp', 'pre_flange')}}
    (output / 'scene.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(HELP, flush=True)
    for actual in scene['actual']:
        print('执行反馈：' + json.dumps(actual, ensure_ascii=False), flush=True)
    if check_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    import open3d as o3d
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    labels = {'offline': 'OFFLINE / NO ROBOT', 'execute': 'CLOSE: CONTINUE GRASP EXECUTION',
              'plan': 'CLOSE: CONTINUE PLAN ONLY', 'hold': 'CLOSE: RETURN TO GRIPPER CONSOLE',
              'done': 'CLOSE: END GRIPPER SESSION'}
    title = 'GSNet FR3 | ' + labels[continuation] + ' | G E P O T / A axes / C B view / S screenshot'
    if not visualizer.create_window(window_name=title, width=1280, height=900, visible=screenshot is None):
        raise RuntimeError('Open3D could not create a window; run from the Ubuntu graphical desktop')
    visible = {'cloud', 'grasp', 'envelope', 'tcp_axes', 'pregrasp', 'actual'}
    axes_mode = 'tcp_axes'
    try:
        for name in visible:
            for obj in groups[name]:
                visualizer.add_geometry(obj)
        options = visualizer.get_render_option()
        options.background_color = np.array([.06, .075, .09])
        options.point_size = 2.
        options.line_width = 2.

        def camera_view(vis, camera=True):
            control = vis.get_view_control()
            control.set_lookat(scene['base_tcp'][:3, 3])
            if camera:
                rotation = scene['base_camera'][:3, :3]
                control.set_front(-rotation[:, 2])
                control.set_up(-rotation[:, 1])
            else:
                control.set_front([1., -1., .8])
                control.set_up([0., 0., 1.])
            control.set_zoom(.35)
            return False

        def toggle(name):
            def callback(vis):
                removing = name in visible
                for obj in groups[name]:
                    if removing:
                        vis.remove_geometry(obj, reset_bounding_box=False)
                    else:
                        vis.add_geometry(obj, reset_bounding_box=False)
                visible.discard(name) if removing else visible.add(name)
                return False
            return callback

        def axes(vis):
            nonlocal axes_mode
            if axes_mode:
                toggle(axes_mode)(vis)
            axes_mode = {'tcp_axes': 'grasp_axes', 'grasp_axes': '', '': 'tcp_axes'}[axes_mode]
            if axes_mode:
                toggle(axes_mode)(vis)
            print('坐标轴：' + (axes_mode or '隐藏'), flush=True)
            return False

        def save(vis, path=None):
            path = Path(path) if path else output / (datetime.now().strftime('view-%Y%m%d-%H%M%S-%f') + '.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            vis.capture_screen_image(str(path), do_render=True)
            path.with_suffix('.view.json').write_text(vis.get_view_status())
            print('截图：' + str(path), flush=True)
            return False

        for key, name in [('G', 'grasp'), ('E', 'envelope'), ('P', 'pregrasp'), ('O', 'actual'), ('T', 'table')]:
            visualizer.register_key_callback(ord(key), toggle(name))
        visualizer.register_key_callback(ord('A'), axes)
        visualizer.register_key_callback(ord('C'), lambda vis: camera_view(vis, True))
        visualizer.register_key_callback(ord('B'), lambda vis: camera_view(vis, False))
        visualizer.register_key_callback(ord('S'), save)
        camera_view(visualizer)
        if screenshot is not None:
            for _ in range(4):
                visualizer.poll_events(); visualizer.update_renderer()
            save(visualizer, screenshot)
        else:
            print({'execute': '关闭窗口后继续已经启用的抓取动作。',
                   'plan': '关闭窗口后仅继续规划，不执行动作。',
                   'hold': '关闭窗口后返回夹爪通信控制台。',
                   'done': '关闭窗口后结束夹爪通信会话。',
                   'offline': '只查看保存的数据，不连接任何硬件。'}[continuation], flush=True)
            visualizer.run()
    finally:
        visualizer.destroy_window()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='gsnet_pick, gsnet_single or gsnet_ranked run directory')
    parser.add_argument('--robot-log', type=Path, help='Optional execution JSONL targeting the same grasp')
    parser.add_argument('--check-only', action='store_true', help='Build geometry and manifest without a window')
    parser.add_argument('--screenshot', type=Path, help='Render one PNG in an invisible window, then exit')
    parser.add_argument('--continuation', choices=('offline', 'plan', 'execute', 'hold', 'done'), default='offline',
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        view(args.run_dir, robot_log=args.robot_log, screenshot=args.screenshot,
             check_only=args.check_only, continuation=args.continuation)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'Open3D stopped: {error!r}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
