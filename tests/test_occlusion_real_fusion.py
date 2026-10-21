"""Measured-relative-pose binding and geometry direction without devices."""

import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from occlusion_reasoning.observations import InstanceEvidence, build_observation
from occlusion_reasoning.real_fusion import bind_static_pair, relative_pose_from_board, fuse


def calibration():
    pose=np.eye(4);pose[0,3]=.2
    return {"schema":"static_camera_pair_v1","status":"measured",
            "camera_serials":{"global":"g","wrist":"w"},"translation_unit":"m",
            "matrix_convention":"p_global = T_global_from_wrist @ p_wrist",
            "validity":"both_camera_poses_unchanged",
            "T_global_optical_from_wrist_optical":pose.tolist()}


def observations():
    output=[]
    for camera in ("global","wrist"):
        mask=np.ones((10,12),bool)
        packet,arrays=build_observation(
            depth_m=np.ones((10,12)),intrinsics=dict(fx=100,fy=100,cx=6,cy=5),
            instances=[InstanceEvidence("visible",mask,[0,0,12,10],.8)],
            frame_id=camera,query="unused",alignment_status="recorded_aligned",
            timestamp_s=1.,camera_id=camera,capture_group_id="same",timestamp_domain="same",
            reference_frame=camera+"_local",T_reference_from_camera_optical=np.eye(4))
        output.append((packet,arrays))
    return output


def test_relative_board_composition_uses_the_correct_transform_direction():
    expected=np.eye(4)
    expected[:3,:3]=Rotation.from_euler("xyz",[.2,-.1,.3]).as_matrix()
    expected[:3,3]=[.2,-.3,.5]
    global_from_board=np.eye(4)
    global_from_board[:3,:3]=Rotation.from_euler("xyz",[-.4,.3,-.1]).as_matrix()
    global_from_board[:3,3]=[.1,.2,1.2]
    wrist_from_board=np.linalg.inv(expected)@global_from_board
    np.testing.assert_allclose(relative_pose_from_board(global_from_board,wrist_from_board),expected,atol=1e-12)


def test_binding_preserves_sources_and_updates_only_reference_coordinates():
    source=observations()
    original=copy.deepcopy(source)
    bound=bind_static_pair(source,calibration(),{"global":"g","wrist":"w"})
    assert all(p["frame"]["reference_frame"]=="measured_static_global_optical" for p,_ in bound)
    assert source[1][0]["frame"]["reference_frame"]=="wrist_local"
    node=bound[1][0]["relation_graph"]["nodes"][0]
    reference=bound[1][1][node["observed_points_reference_key"]]
    camera=bound[1][1][node["observed_points_camera_key"]]
    np.testing.assert_allclose(reference-camera,np.tile([.2,0,0],(len(camera),1)),atol=1e-12)
    for index in (0,1):
        np.testing.assert_array_equal(source[index][1]["depth_m"],original[index][1]["depth_m"])
    assert bound[1][0]["provenance"]["robot_state_queried"] is False


@pytest.mark.parametrize("field,value",[("status","pending_measurement"),
    ("translation_unit","mm"),("validity","any_robot_pose"),
    ("matrix_convention","inverse"),("camera_serials",{"global":"other","wrist":"w"}),
    ("T_global_optical_from_wrist_optical",np.zeros((4,4)).tolist())])
def test_incomplete_or_misbound_calibration_is_rejected(field,value):
    c=calibration();c[field]=value
    with pytest.raises(ValueError):bind_static_pair(observations(),c,{"global":"g","wrist":"w"})


def test_offline_fusion_exports_generic_packets_without_loading_any_hardware(tmp_path):
    import cv2,hashlib
    analysis=tmp_path/"analysis"
    for packet,arrays in observations():
        camera=packet["frame"]["camera_id"]
        snapshot=tmp_path/camera;snapshot.mkdir()
        cv2.imwrite(str(snapshot/"color.png"),np.zeros((10,12,3),np.uint8))
        np.save(snapshot/"depth_m.npy",arrays["depth_m"])
        (snapshot/"grasp_input.json").write_text(json.dumps({"device_serial":"g" if camera=="global" else "w"}))
        packet["provenance"]={"snapshot_dir":str(snapshot),
            "rgb_sha256":hashlib.sha256((snapshot/"color.png").read_bytes()).hexdigest(),
            "depth_sha256":hashlib.sha256((snapshot/"depth_m.npy").read_bytes()).hexdigest()}
        directory=analysis/camera/"observation";directory.mkdir(parents=True)
        packet["artifacts"]={"arrays":"observation_arrays.npz"}
        np.savez_compressed(directory/"observation_arrays.npz",**arrays)
        (directory/"scene_observation.json").write_text(json.dumps(packet))
    path=tmp_path/"calibration.json";path.write_text(json.dumps(calibration()))
    result=fuse(analysis,path,tmp_path/"fusion")
    assert not result["real_camera_fusion"]["robot_control_initialized"]
    assert len(result["single_view_boundary_evidence"])==2
    assert (tmp_path/"fusion/evidence.png").is_file()
    assert (tmp_path/"fusion/wrist/observation_arrays.npz").is_file()
