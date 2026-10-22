"""Run from the project root: python -m hardware.cli --help."""
import argparse
from dataclasses import replace
import json
import sys

from .config import DEFAULT_CONFIG, load_config
from .fr3 import FrankaMoveItArm
from .interfaces import Pose6D
from .robotiq import Robotiq2F85


def main(argv=None):
    parser = argparse.ArgumentParser(description="FR3 flange PTP + Robotiq 2F-85; no automatic stack startup")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="device", required=True)
    sub.add_parser("config", help="Validate and print settings; no device access")
    calibration = sub.add_parser("calibration", help="Offline coordinate conversion; no device access")
    calibration_sub = calibration.add_subparsers(dest="operation", required=True)
    calibration_sub.add_parser("inspect")
    axes = calibration_sub.add_parser("axes", help="Compare jaw-axis candidates; no planning/motion/config writes")
    axes_pose = axes.add_mutually_exclusive_group(required=True)
    axes_pose.add_argument("--flange-pose", nargs=6, type=float, help="Offline flange XYZ/RPY, metres/radians")
    axes_pose.add_argument("--current-pose", action="store_true", help="Read flange FK from the already-running stack")
    closing_axis = axes.add_mutually_exclusive_group()
    closing_axis.add_argument("--closing-axis-flange", nargs=3, type=float,
                              help="Observed jaw direction in flange coordinates; prints candidate only")
    closing_axis.add_argument("--closing-axis-base", nargs=3, type=float,
                              help="Observed jaw line in base coordinates at this same pose, e.g. 0 1 0")
    point = calibration_sub.add_parser("point", help="Camera XYZ in metres -> base XYZ in metres")
    point.add_argument("point", nargs=3, type=float)
    point.add_argument("--flange-pose", nargs=6, type=float, required=True,
                       help="Capture-time flange XYZ/RPY in metres/radians; never read automatically")
    point.add_argument("--camera", default="wrist")
    point.add_argument("--serial", required=True, help="Actual capture device serial")
    point.add_argument("--frame", required=True, help="Point's optical frame")
    arm = sub.add_parser("arm")
    arm_sub = arm.add_subparsers(dest="operation", required=True)
    for name in ("pose", "plan", "move", "home"):
        p = arm_sub.add_parser(name)
        p.add_argument("--stack", choices=("real", "fake"))
        if name in ("plan", "move"):
            p.add_argument("pose", nargs=6, type=float, metavar="VALUE",
                           help="X Y Z RX RY RZ: fr3_link8 in fr3_link0, metres and RPY radians")
        if name in ("move", "home"):
            p.add_argument("--execute", action="store_true", help="Authorize execution of this command")
    grip = sub.add_parser("gripper")
    grip_sub = grip.add_subparsers(dest="operation", required=True)
    for name in ("ports", "status", "check", "console", "activate", "open", "close", "move", "stop"):
        p = grip_sub.add_parser(name)
        if name != "ports":
            p.add_argument("--port", help="Explicit /dev/serial/by-id/... override")
        if name not in ("ports", "status", "check"):
            p.add_argument("--execute", action="store_true")
        if name == "check":
            p.add_argument("--duration", type=float, default=60, help="Read-only check duration in seconds, >=3")
            p.add_argument("--interval", type=float, default=.1, help="Polling interval in seconds")
            p.add_argument("--output-dir", help="Directory for JSON report and JSONL samples")
        if name == "console":
            p.add_argument("--output-dir", help="Directory for session JSONL logs")
        if name in ("open", "close", "move", "console"):
            p.add_argument("--speed-raw", type=int, help="Override raw speed, 0..255")
            p.add_argument("--force-raw", type=int, help="Override raw force, 0..255")
            p.add_argument("--close-current-limit-ma", type=int,
                           help="Calibrated closing-current stop threshold, 10..2550 mA in steps of 10; not newtons")
        if name == "move":
            p.add_argument("position_raw", type=int)
    args = parser.parse_args(argv)
    if getattr(args, "operation", None) in ("move", "home", "activate", "open", "close", "stop") and not args.execute:
        parser.error("This command requires --execute; arm plan is planning only")
    try:
        cfg = load_config(args.config)
        if args.device == "config":
            result = cfg.to_dict()
        elif args.device == "calibration":
            from .transforms import load_transforms
            if cfg.calibration_file is None:
                raise RuntimeError("calibration_file 尚未配置")
            transforms = load_transforms(cfg)
            if args.operation == "inspect":
                result = transforms.summary()
            elif args.operation == "axes":
                from .grasp_alignment import axes_report
                if args.current_pose:
                    robot = FrankaMoveItArm(cfg.arm, execution_enabled=False)
                    try:
                        pose = robot.get_pose()
                    finally:
                        robot.close()
                else:
                    pose = Pose6D(*args.flange_pose)
                result = axes_report(pose, transforms, args.closing_axis_flange,
                                     closing_axis_base=args.closing_axis_base)
                result["pose_source"] = "stack_read_only" if args.current_pose else "supplied_offline_pose"
                result["hardware_accessed"] = bool(args.current_pose)
            else:
                result = dict(base_frame="fr3_link0", unit="m", hardware_accessed=False,
                              point=transforms.camera_point_to_base(
                                  args.point, args.camera, Pose6D(*args.flange_pose),
                                  device_serial=args.serial, optical_frame=args.frame))
        elif args.device == "arm":
            robot = FrankaMoveItArm(cfg.arm, stack=args.stack,
                                   execution_enabled=getattr(args, "execute", False))
            print(f"FR3 stack={robot.config.stack}, operation={args.operation}; metres / radians", file=sys.stderr)
            if args.operation == "pose":
                robot.get_pose()
                result = robot.last_result
            elif args.operation == "home":
                result = robot.home()
            else:
                pose = Pose6D(*args.pose)
                result = robot.plan_pose(pose) if args.operation == "plan" else robot.move_pose(pose)
        elif args.operation == "ports":
            from serial.tools.list_ports import comports
            result = [dict(device=p.device, serial_number=p.serial_number, description=p.description,
                           hwid=p.hwid) for p in comports()]
        else:
            config = replace(cfg.gripper, port=args.port) if args.port else cfg.gripper
            for key in ("speed_raw", "force_raw"):
                value = getattr(args, key, None)
                if value is not None:
                    if not 0 <= value <= 255:
                        raise ValueError(f"{key} must be in [0, 255]")
                    config = replace(config, **{key: value})
            current_limit = getattr(args, "close_current_limit_ma", None)
            if current_limit is not None:
                from .config import validate_gripper_feedback
                config = replace(config, close_current_limit_ma=current_limit)
                validate_gripper_feedback(config)
            if args.operation == "check":
                from .robotiq_diagnostics import check_communication
                result = check_communication(config, duration_s=args.duration, interval_s=args.interval,
                                             output_dir=args.output_dir,
                                             progress=lambda data: print(json.dumps(data), file=sys.stderr, flush=True))
            elif args.operation == "console":
                from .gripper_console import run_console
                result = run_console(config, execution_enabled=args.execute, output_dir=args.output_dir)
            else:
                with Robotiq2F85(config, execution_enabled=getattr(args, "execute", False)) as gripper:
                    state = gripper.move_raw(args.position_raw) if args.operation == "move" else getattr(gripper, args.operation)()
                    result = state.to_dict()
                    result["session"] = gripper.session_info()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.device == "gripper" and args.operation == "check":
            return 130 if result["interrupted"] else 0 if result["communication_passed"] else 1
        return 0
    except KeyboardInterrupt:
        print("Interrupted; no recovery or retry.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
