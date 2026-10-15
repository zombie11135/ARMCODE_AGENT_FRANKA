"""Interactive, read-only recorder for manually taught xArm flange poses."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record manually positioned xArm poses; never enables or moves the robot"
    )
    parser.add_argument("--xarm-ip", default="192.168.1.237")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.count < 4:
        parser.error("--count must be at least 4")

    from xarm import __version__ as xarm_sdk_version
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(args.xarm_ip, is_radian=True)
    try:
        tcp_offset = [float(value) for value in arm.tcp_offset]
        translation_nonzero = any(abs(value) > 0.1 for value in tcp_offset[:3])
        rotation_nonzero = any(abs(value) > 1e-4 for value in tcp_offset[3:])
        if translation_nonzero or rotation_nonzero:
            raise RuntimeError(
                "controller TCP offset must be zero while recording flange poses; observed {}".format(
                    tcp_offset
                )
            )
        state = arm.get_state()
        errors = arm.get_err_warn_code()
        serial = arm.get_robot_sn()
        print("Connected read-only. state={}, error/warn={}, tcp_offset={}".format(state, errors, tcp_offset))
        print("Move only with xArm Studio/manual controls. This process sends no motion commands.")
        poses = []
        for index in range(args.count):
            input(
                "Pose {}/{}: keep the improvised tip on the same mark, stop the robot, then press Enter... ".format(
                    index + 1, args.count
                )
            )
            code, pose = arm.get_position(is_radian=True)
            if code != 0:
                raise RuntimeError("get_position failed with code {}".format(code))
            poses.append(
                {
                    "name": "teach_{:02d}".format(index + 1),
                    "pose": [float(value) for value in pose],
                    "recorded_unix_time": time.time(),
                }
            )
            print("  recorded {}".format(poses[-1]["pose"]))
        manifest = {
            "schema_version": 1,
            "example_only": False,
            "capture_mode": "manual_teaching_read_only_get_position",
            "robot_ip": args.xarm_ip,
            "robot_serial_response": str(serial),
            "xarm_sdk_version": str(xarm_sdk_version),
            "controller_tcp_offset_during_capture": tcp_offset,
            "pose_semantics": "T_base_flange_mmrad",
            "teaching_poses_mmrad": poses,
            "tcp_orientation_rpy_rad": None,
            "acceptance": {
                "maximum_rms_residual_m": 0.002,
                "maximum_single_residual_m": 0.004,
                "minimum_orientation_spread_deg": 30.0,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("Saved {}".format(args.output.resolve()))
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
