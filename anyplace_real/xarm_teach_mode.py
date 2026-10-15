"""Explicitly enter, inspect, or leave xArm joint teaching mode.

This helper never sends a Cartesian or joint-position command and never clears
controller errors.  Entering teaching mode does enable the motors and release
the arm for gravity-compensated manual guidance, so an operator confirmation is
required at the terminal.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict


TEACH_MODE = 2
POSITION_MODE = 0
READY_STATE = 0
STOP_STATE = 4


def _require_success(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError("{} failed with xArm code {}".format(operation, code))


def _read_status(arm: Any) -> Dict[str, Any]:
    state_code, state = arm.get_state()
    _require_success(state_code, "get_state")
    error_code, error_warn = arm.get_err_warn_code()
    _require_success(error_code, "get_err_warn_code")
    return {
        "mode": int(arm.mode),
        "state": int(state),
        "is_moving": bool(arm.get_is_moving()),
        "error": int(error_warn[0]),
        "warning": int(error_warn[1]),
        "motor_enable_states": list(arm.motor_enable_states),
    }


def _print_status(status: Dict[str, Any], label: str) -> None:
    print(
        "{}: mode={}, state={}, moving={}, error={}, warning={}, motors={}".format(
            label,
            status["mode"],
            status["state"],
            status["is_moving"],
            status["error"],
            status["warning"],
            status["motor_enable_states"],
        )
    )


def enter_teach_mode(arm: Any, confirmed: bool = False) -> Dict[str, Any]:
    before = _read_status(arm)
    _print_status(before, "Before")
    if before["is_moving"]:
        raise RuntimeError("xArm is moving; wait for it to stop before entering teach mode")
    if before["error"] != 0 or before["warning"] != 0:
        raise RuntimeError(
            "controller error/warning is non-zero; resolve it in xArm Studio first "
            "(this script will not clear it)"
        )
    if before["mode"] == TEACH_MODE:
        print("xArm is already in joint teaching mode (mode 2).")
        return before

    if not confirmed:
        print("\nEntering teaching mode enables the motors and allows manual dragging.")
        print("Remove obstacles, support the arm if necessary, and keep E-stop reachable.")
        answer = input('Type TEACH to continue: ').strip()
        if answer != "TEACH":
            raise RuntimeError("operator confirmation not received; no mode change made")

    motors_enabled = False
    try:
        _require_success(arm.motion_enable(enable=True), "motion_enable(True)")
        motors_enabled = True
        _require_success(
            arm.set_mode(TEACH_MODE, detection_param=0),
            "set_mode(2)",
        )
        _require_success(arm.set_state(READY_STATE), "set_state(0)")
        time.sleep(0.5)
        after = _read_status(arm)
        _print_status(after, "After")
        if after["mode"] != TEACH_MODE:
            raise RuntimeError(
                "controller did not report joint teaching mode; observed mode {}".format(
                    after["mode"]
                )
            )
        print("Joint teaching mode is active. Disconnecting does not command motion.")
        return after
    except Exception:
        if motors_enabled:
            # Stop without clearing faults or commanding a pose.
            arm.set_state(STOP_STATE)
        raise


def leave_teach_mode(arm: Any, confirmed: bool = False) -> Dict[str, Any]:
    before = _read_status(arm)
    _print_status(before, "Before")
    if before["is_moving"]:
        raise RuntimeError("xArm is moving; stop manual motion before leaving teach mode")
    if not confirmed:
        print("\nThe arm will leave gravity-compensated teaching mode and enter stop state.")
        print("Hold clear of pinch points and support any unstable payload.")
        answer = input('Type STOP to continue: ').strip()
        if answer != "STOP":
            raise RuntimeError("operator confirmation not received; no mode change made")

    _require_success(arm.set_state(STOP_STATE), "set_state(4)")
    _require_success(arm.set_mode(POSITION_MODE), "set_mode(0)")
    time.sleep(0.5)
    after = _read_status(arm)
    _print_status(after, "After")
    print("Teaching mode is inactive. No position command was sent.")
    return after


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enter/status/leave xArm joint teaching mode without sending a pose command"
    )
    parser.add_argument("action", choices=("enter", "status", "leave"))
    parser.add_argument("--xarm-ip", default="192.168.1.237")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the typed confirmation; use only with an operator physically present",
    )
    args = parser.parse_args()

    from xarm.wrapper import XArmAPI

    arm = XArmAPI(args.xarm_ip, is_radian=True)
    try:
        if not arm.connected:
            raise RuntimeError("failed to connect to xArm at {}".format(args.xarm_ip))
        if args.action == "status":
            _print_status(_read_status(arm), "Status")
        elif args.action == "enter":
            enter_teach_mode(arm, confirmed=args.yes)
        else:
            leave_teach_mode(arm, confirmed=args.yes)
    finally:
        arm.disconnect()
        print("Disconnected from xArm.")


if __name__ == "__main__":
    main()
