"""Explicit integration test against the fake Docker service only.

Run from the project root after starting config/hardware/fake.override.yaml.
No real service is started or addressed. The existing Session also verifies that
the robot description uses mock_components/GenericSystem before any operation.
"""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware.config import load_config
from hardware.fr3 import FrankaMoveItArm, MotionError
from hardware.interfaces import Pose6D


def events(path):
    result = []
    for line in Path(path).read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # Last line may be in flight during the cancellation test.
    return result


def stopped(rows):
    return any(e["event"] == "control_stopped" and e["verified_state"] == "inactive" for e in rows)


def main():
    cfg = load_config()
    arm = FrankaMoveItArm(cfg.arm, stack="fake", execution_enabled=True)
    report = {"hardware": "fake", "checks": [], "logs": []}
    original = arm.get_pose()
    target = Pose6D(original.x + .005, original.y, original.z, original.rx, original.ry, original.rz)
    plan = arm.plan_pose(target)
    rows = events(plan["host_log_path"])
    assert not any(e["event"] in ("execute_requested", "controller_switch") for e in rows)
    assert plan["plan"]["collision_checked_samples"] > 0
    report["checks"].append("plan only; collision scene checked; controller untouched")
    report["logs"].append(plan["host_log_path"])

    move = arm.move_pose(target)
    assert move["endpoint"]["within_pose_tolerance"] and move["endpoint"]["robot_stopped"]
    assert stopped(events(move["host_log_path"]))
    report["endpoint"] = move["endpoint"]
    report["checks"].append("mock movement; endpoint accepted; controller inactive after cleanup")
    report["logs"].append(move["host_log_path"])

    # The existing attached tool extends below the flange; this goal intersects table.
    try:
        arm.plan_pose(Pose6D(target.x, target.y, .05, target.rx, target.ry, target.rz))
    except MotionError as error:
        assert "IK" in str(error) or "collision" in str(error).lower() or "table" in str(error).lower(), error
        rows = events(error.result["host_log_path"])
        assert not any(e["event"] == "controller_switch" for e in rows)
        report["checks"].append("tool/table-intersecting target rejected before activation")
        report["logs"].append(error.result["host_log_path"])
    else:
        raise AssertionError("Table-intersecting target unexpectedly accepted")

    # Interrupt host CLI after its own mock trajectory has actually started.
    log_dir = Path(cfg.arm.workspace) / "logs/agent"
    before = set(log_dir.glob("*.jsonl"))
    next_target = [target.x + .04, target.y, target.z, target.rx, target.ry, target.rz]
    command = [sys.executable, "-m", "hardware.cli", "arm", "move",
               *map(str, next_target), "--stack", "fake", "--execute"]
    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        owned_log = None
        deadline = time.monotonic() + 60
        try:
            while time.monotonic() < deadline and process.poll() is None:
                for path in set(log_dir.glob("*.jsonl")) - before:
                    rows = events(path)
                    if any(e["event"] == "start" and e["hardware"] == "mock" for e in rows) and \
                       any(e["event"] == "execute_requested" for e in rows):
                        owned_log = path
                        break
                if owned_log:
                    time.sleep(.3)  # Allow action acceptance before cancellation.
                    process.send_signal(signal.SIGINT)
                    break
                time.sleep(.1)
            if owned_log is None:
                raise AssertionError("Mock command did not reach execution")
            process.wait(timeout=45)
            output.seek(0)
            transcript = output.read()
            assert process.returncode == 130, transcript
            rows = events(owned_log)
            assert stopped(rows), rows[-5:]
            report["checks"].append("host Ctrl-C cancels owned mock trajectory and deactivates controller")
            report["logs"].append(str(owned_log))
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=45)

    # Prove the interrupted command released its process lock and controller.
    after = arm.get_pose()
    reset_plan = arm.plan_pose(original)
    report["checks"].append("new request works after interruption; no automatic recovery")
    report["pose_after_cancel"] = after.values
    report["logs"].append(reset_plan["host_log_path"])
    destination = ROOT / "runs/hardware_validation/fr3_mock_smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
