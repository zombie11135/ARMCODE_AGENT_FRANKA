#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from xarm.wrapper import XArmAPI


def check_code(arm, code, label=""):
    """
    检查 xArm SDK 返回码
    code == 0 一般表示成功
    """
    if code != 0:
        print(f"[ERROR] {label} failed, code={code}")
        print(f"        arm state={arm.state}, error={arm.error_code}, warn={arm.warn_code}")
        return False
    return True


def init_arm(ip: str, is_radian: bool = True):
    """
    初始化机械臂
    """
    arm = XArmAPI(ip, is_radian=is_radian)

    time.sleep(0.5)

    print("[INFO] Connected:", arm.connected)
    print("[INFO] State:", arm.state)
    print("[INFO] Error code:", arm.error_code)
    print("[INFO] Warn code:", arm.warn_code)

    # 清除警告和错误
    arm.clean_warn()
    arm.clean_error()

    # 使能机械臂
    code = arm.motion_enable(enable=True)
    check_code(arm, code, "motion_enable")

    # 设置为位置控制模式
    code = arm.set_mode(0)
    check_code(arm, code, "set_mode")

    # 设置为 ready 状态
    code = arm.set_state(0)
    check_code(arm, code, "set_state")

    time.sleep(0.5)

    return arm


def move_to_pose(
    arm,
    x: float,
    y: float,
    z: float,
    rx: float,
    ry: float,
    rz: float,
    speed: float = 50,
    acc: float = 500,
    wait: bool = True,
):
    """
    控制机械臂末端移动到指定位姿

    x, y, z: mm
    rx, ry, rz: rad, because XArmAPI(..., is_radian=True)
    """

    print("\n[INFO] Target pose:")
    print(f"       x={x:.3f}, y={y:.3f}, z={z:.3f}")
    print(f"       rx={rx:.6f}, ry={ry:.6f}, rz={rz:.6f}")

    # 打印当前位姿
    code, current_pose = arm.get_position(is_radian=True)
    if code == 0:
        print("[INFO] Current pose:")
        print("      ", current_pose)
    else:
        print("[WARN] Failed to get current pose")

    # 执行末端笛卡尔空间运动
    code = arm.set_position(
        x=x,
        y=y,
        z=z,
        roll=rx,
        pitch=ry,
        yaw=rz,
        speed=speed,
        mvacc=acc,
        wait=wait,
    )

    if not check_code(arm, code, "set_position"):
        return False

    print("[INFO] Move finished")

    code, final_pose = arm.get_position(is_radian=True)
    if code == 0:
        print("[INFO] Final pose:")
        print("      ", final_pose)

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", type=str, required=True, help="xArm controller IP, e.g. 192.168.1.185")

    parser.add_argument(
        "--pose",
        type=float,
        nargs=6,
        required=True,
        metavar=("x", "y", "z", "rx", "ry", "rz"),
        help="target pose: x y z rx ry rz, xyz in mm, rpy in rad",
    )

    parser.add_argument("--speed", type=float, default=50, help="TCP speed, default 50 mm/s")
    parser.add_argument("--acc", type=float, default=500, help="TCP acceleration, default 500 mm/s^2")

    args = parser.parse_args()

    x, y, z, rx, ry, rz = args.pose

    arm = None

    try:
        arm = init_arm(args.ip, is_radian=True)

        success = move_to_pose(
            arm,
            x=x,
            y=y,
            z=z,
            rx=rx,
            ry=ry,
            rz=rz,
            speed=args.speed,
            acc=args.acc,
            wait=True,
        )

        if success:
            print("\n[SUCCESS] xArm moved to target pose")
        else:
            print("\n[FAILED] xArm move failed")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, stopping arm...")
        if arm is not None:
            arm.set_state(4)

    finally:
        if arm is not None:
            arm.disconnect()
            print("[INFO] Disconnected")


if __name__ == "__main__":
    main()