# skill_lab/test_plan_grasp_with_anygrasp_base.py

import argparse
from pprint import pprint

from core.context import RuntimeContext

from services.anygrasp_service import AnyGraspService
from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill


def print_event(event: dict):
    event_type = event.get("type")

    if event_type == "image":
        print(f"[Image] {event.get('content')}: {event.get('image_path')}")
    elif event_type == "text":
        print(f"[Text] {event.get('content')}")
    elif event_type == "warning":
        print(f"[Warning] {event.get('content')}")
    elif event_type == "error":
        print(f"[Error] {event.get('content')}")
    else:
        print(event)


def build_grasp_input(args):
    """
    直接构造 plan_grasp_with_anygrasp 所需的 grasp_input。

    注意：
    - color/depth/mask 必须来自同一次腕部相机采集。
    - mask 应该和 color 对齐。
    - depth 最好是 align 到 color 后的 depth。
    """

    if args.depth_npy_path:
        depth_npy_path = args.depth_npy_path
        depth_path = None
    else:
        depth_npy_path = None
        depth_path = args.depth_path

    if args.mask_npy_path:
        mask_npy_path = args.mask_npy_path
        mask_path = None
    else:
        mask_npy_path = None
        mask_path = args.mask_path

    intrinsics = {
        "fx": args.fx,
        "fy": args.fy,
        "cx": args.cx,
        "cy": args.cy,
    }

    grasp_input = {
        "color_path": args.color_path,
        "depth_path": depth_path,
        "depth_npy_path": depth_npy_path,
        "mask_path": mask_path,
        "mask_npy_path": mask_npy_path,
        "depth_intrinsics": intrinsics,
        "color_intrinsics": intrinsics,
        "depth_scale": args.depth_scale,
        "bbox": None,
        "object_id": "manual_test_object",
        "target_name": "manual_test",
    }

    return grasp_input


def main():
    parser = argparse.ArgumentParser()

    # AnyGrasp
    parser.add_argument(
        "--anygrasp_dir",
        required=True,
        help="AnyGrasp SDK 的 grasp_detection 目录，也就是 demo.py 所在目录。",
    )
    parser.add_argument(
        "--anygrasp_ckpt",
        required=True,
        help="AnyGrasp checkpoint 路径，例如 grasp_detection/log/checkpoint_detection.tar",
    )

    # RGB-D + mask
    parser.add_argument(
        "--color_path",
        default="runs/current_capture/wrist_color.png",
    )
    parser.add_argument(
        "--depth_path",
        default="runs/current_capture/wrist_depth.png",
    )
    parser.add_argument(
        "--depth_npy_path",
        default=None,
        help="可选。如果有米制 depth npy，优先用这个。",
    )
    parser.add_argument(
        "--mask_path",
        default="runs/current_capture/sam3_gdino_segment_01_obj_1_mask.png",
    )
    parser.add_argument(
        "--mask_npy_path",
        default=None,
        help="可选。如果有 mask npy，优先用这个。",
    )

    # 相机内参。默认先放你之前 GraspNet 代码中的一组 1280x720 内参。
    # 如果你的 realsense service 已经输出了真实内参，请用真实值覆盖。
    parser.add_argument("--fx", type=float, default=909.59)
    parser.add_argument("--fy", type=float, default=909.80)
    parser.add_argument("--cx", type=float, default=658.97)
    parser.add_argument("--cy", type=float, default=355.42)

    # 如果 depth_path 是 uint16 mm，则 depth_scale=1000.0。
    # 如果 depth_npy_path 已经是米，一般不用管，service 会自动判断。
    parser.add_argument("--depth_scale", type=float, default=1000.0)

    # xArm + hand-eye
    parser.add_argument("--xarm_ip", required=True)
    parser.add_argument(
        "--handeye_config",
        default="config/calibration/wrist_handeye.yaml",
    )

    # AnyGrasp 参数
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--lims_margin", type=float, default=0.03)
    parser.add_argument("--min_mask_points", type=int, default=200)

    # 姿态选择参数
    parser.add_argument("--max_rx_tilt_deg", type=float, default=35.0)
    parser.add_argument("--max_ry_tilt_deg", type=float, default=35.0)
    parser.add_argument("--gripper_length", type=float, default=0.12)

    # 不执行机械臂，只输出规划位姿
    args = parser.parse_args()

    print("\n=== Test Config ===")
    pprint(vars(args))

    print("\n=== Loading AnyGraspService ===")
    anygrasp_service = AnyGraspService(
        grasp_detection_dir=args.anygrasp_dir,
        checkpoint_path=args.anygrasp_ckpt,
        max_gripper_width=0.10,
        gripper_height=0.03,
        top_down_grasp=False,
        debug=False,
    )

    context = RuntimeContext(
        anygrasp=anygrasp_service,
        event_callback=print_event,
    )

    # 如果你的 RuntimeContext 不支持 xarm 字段，动态挂上去也可以。
    # plan_grasp_with_anygrasp 内部会在需要时创建 XArmAPI。
    context.xarm = None

    skill = PlanGraspWithAnyGraspSkill(context)

    grasp_input = build_grasp_input(args)

    print("\n=== Grasp Input ===")
    pprint(grasp_input)

    print("\n=== Running plan_grasp_with_anygrasp ===")

    result = skill.run(
        grasp_input=grasp_input,

        # xArm 当前位姿 + 手眼标定
        xarm_ip=args.xarm_ip,
        handeye_config_path=args.handeye_config,

        # 使用 SAM3 mask 约束点云
        use_mask_points=True,
        min_mask_points=args.min_mask_points,

        # 根据 mask 内点云自动计算 tight lims
        auto_lims_from_mask=True,
        lims_margin=args.lims_margin,

        # AnyGrasp 设置
        top_k=args.top_k,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True,

        # 相机系 grasp -> base 系 grasp
        gripper_length=args.gripper_length,
        max_rx_tilt_deg=args.max_rx_tilt_deg,
        max_ry_tilt_deg=args.max_ry_tilt_deg,
        enable_pose_fix=True,

        # 只规划，不执行
        approach_clearance_mm=80.0,
        lift_after_grasp_mm=120.0,
    )

    print("\n=== SkillResult ===")
    print("success:", result.success)
    print("message:", result.message)
    print("error:", result.error)

    if not result.success:
        print("\n=== Error Data ===")
        pprint(result.data)
        return

    data = result.data

    print("\n=== Current EE Pose, m/rad ===")
    pprint(data.get("current_ee_pose_mrad"))

    print("\n=== Best Grasp Camera ===")
    pprint(data.get("best_grasp_camera"))

    print("\n=== Best Grasp Base, m/rad ===")
    pprint(data.get("best_grasp_base_mrad"))

    print("\n=== Best Grasp xArm Pose, mm/rad ===")
    pprint(data.get("best_grasp_xarm_mmrad"))

    print("\n=== Approach Pose, mm/rad ===")
    pprint(data.get("approach_pose_mmrad"))

    print("\n=== Lift Pose, mm/rad ===")
    pprint(data.get("lift_pose_mmrad"))

    print("\n=== Choose Reason ===")
    print(data.get("choose_reason"))

    print("\n=== AnyGrasp Info ===")
    print("lims:", data.get("lims"))
    print("num_candidates:", data.get("num_candidates"))
    print("result_json_path:", data.get("result_json_path"))

    print("\n=== Top Converted Candidates ===")
    for i, cand in enumerate(data.get("converted_candidates", [])[:5]):
        print(f"\n[{i}]")
        print("raw_score:", cand.get("raw_score"))
        print("effective_score:", cand.get("effective_score"))
        print("tilt_rx_deg:", cand.get("tilt_rx_deg"))
        print("tilt_ry_deg:", cand.get("tilt_ry_deg"))
        print("xarm_pose_mmrad:", cand.get("xarm_pose_mmrad"))


if __name__ == "__main__":
    main()