"""Executed inside the existing Docker container, not imported on the host.

Reuse the tested Session, including collision scene, trajectory checks, endpoint
acceptance and owned-controller cleanup. This file is sent through docker exec;
the old agent directory need not be mounted in the container.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys


def main():
    request = json.loads(sys.argv[1])
    request_id = request["request_id"]
    if len(request_id) != 32 or any(c not in "0123456789abcdef" for c in request_id):
        raise ValueError("Invalid request id")
    result = {"ok": False, "request_id": request_id, "operation": request["operation"],
              "stack": request["stack"], "cleanup_ok": True}
    pidfile = Path(f"/tmp/armcode-agent-{request_id}.pid")
    session = None
    initialized = False
    lock = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"FR3 request interrupted by signal {signum}")

    def deadline(signum, frame):
        raise TimeoutError("FR3 request deadline exceeded")

    try:
        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, request["timeout_s"])
        lock = open("/tmp/armcode-agent-fr3.lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pidfile.write_text(str(os.getpid()))
        sys.path.insert(0, "/project/scripts")
        import real_ptp as ptp
        operation = request["operation"]
        if operation not in ("pose", "plan", "move", "plan_grasps") or request["stack"] not in ("fake", "real"):
            raise ValueError("Unsupported operation or hardware mode")
        screening = request.get('singularity_screening', 'endpoint')
        if screening not in ('endpoint', 'advisory'):
            raise ValueError('Unknown singularity screening mode')
        if operation == 'plan_grasps' and screening != 'endpoint':
            raise ValueError('Ranked grasp screening requires endpoint mode')
        if not math.isfinite(request["speed_scaling"]) or not 0 < request["speed_scaling"] <= 1:
            raise ValueError("Invalid speed scaling")
        for key in ("position_tolerance_m", "orientation_tolerance_rad"):
            if not math.isfinite(request[key]) or request[key] <= 0:
                raise ValueError("Invalid endpoint tolerance")
        if request["orientation_tolerance_rad"] > math.pi:
            raise ValueError("Orientation tolerance must be <= pi")
        if operation in ('plan', 'move') and (len(request.get("pose", [])) != 6 or
                                    not all(math.isfinite(v) for v in request["pose"])):
            raise ValueError("Expected six finite pose values")
        args = argparse.Namespace(
            pose=request.get("pose"), offset=None, show_pose=operation == "pose",
            execute=operation == "move", mock=request["stack"] == "fake",
            speed=request["speed_scaling"], position_tolerance=request["position_tolerance_m"],
            orientation_tolerance=request["orientation_tolerance_rad"],
            max_translation=None, max_rotation=None, max_joint_delta=None,
            goal_joints=request.get('goal_joints'),
            singularity_screening=screening,
        )
        log_path = Path("/project/logs/agent") / f"{request_id}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        result["log_path"] = str(log_path)
        ptp.rclpy.init(signal_handler_options=ptp.SignalHandlerOptions.NO)
        initialized = True
        with log_path.open("x") as log:
            session = ptp.Session(args, log)
            original_emit = session.emit

            def emit(event, **data):
                original_emit(event, **data)
                if event == "plan":
                    result["plan"] = data["summary"]
                elif event in ("current_pose", "endpoint", "action_result", 'grasp_selection'):
                    result[event] = data

            session.emit = emit
            try:
                if operation == "pose":
                    session.show_pose()
                elif operation == 'plan_grasps':
                    from grasp_screen import plan_grasps
                    plan_grasps(session, request['grasp_candidates'])
                else:
                    initial, target, trajectory = session.prepare()
                    if operation == "move":
                        session.execute(initial, target, trajectory)
                result["ok"] = True
            except (Exception, KeyboardInterrupt) as error:
                result["error"] = repr(error)
                session.emit("failure", error=repr(error))
            finally:
                # Repeated interrupts must not interrupt controller cancellation.
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                result["cleanup_ok"] = session.cleanup()
                result["ok"] = result["ok"] and result["cleanup_ok"]
                if not result["cleanup_ok"]:
                    result["error"] = result.get("error", "") + " Controller cleanup incomplete"
    except (Exception, KeyboardInterrupt) as error:
        result["ok"] = False
        result["error"] = repr(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if initialized:
            try:
                ptp.rclpy.shutdown()
            except Exception as error:
                result["ok"] = False
                result["error"] = result.get("error", "") + f" ROS shutdown failed: {error!r}"
        pidfile.unlink(missing_ok=True)
        if lock is not None:
            lock.close()
    print("ARM_AGENT_RESULT=" + json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
