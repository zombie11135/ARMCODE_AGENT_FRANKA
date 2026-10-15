"""Host adapter for the already deployed ROS 2 + MoveIt 2 PTP client."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from .config import ArmConfig
from .interfaces import Pose6D


class MotionError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(result.get("error", "FR3 request failed") +
                         (f"; log: {result['host_log_path']}" if result.get("host_log_path") else ""))


class FrankaMoveItArm:
    def __init__(self, config: ArmConfig, *, stack=None, execution_enabled=False, motion_guard=None):
        self.config = replace(config, stack=stack) if stack is not None else config
        if self.config.stack not in ("real", "fake"):
            raise ValueError("stack must be real or fake")
        self.execution_enabled = execution_enabled
        self.motion_guard = motion_guard
        if self.config.singularity_screening not in ('endpoint', 'advisory'):
            raise ValueError('Unknown singularity screening mode')
        self.set_speed_scaling(self.config.speed_scaling)
        self.workspace = Path(self.config.workspace)
        self.last_result = None

    def set_speed_scaling(self, value):
        import math
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("arm speed scaling must be finite and in (0, 1]")
        self.config = replace(self.config, speed_scaling=float(value))

    def _command(self, source, argument):
        entrypoint = "/fr3-real-entrypoint.sh" if self.config.stack == "real" else "/fr3-entrypoint.sh"
        return ["docker", "compose", "-f", str(self.workspace / "compose.yaml"),
                "exec", "-T", self.config.stack, "/bin/bash", entrypoint,
                "python3", "-u", "-c", source, argument]

    def _cancel(self, request_id):
        # Signal only this request's process, never another controller client.
        source = """import os, pathlib, signal, sys
rid = sys.argv[1]
p = pathlib.Path('/tmp/armcode-agent-' + rid + '.pid')
if p.exists():
    pid = int(p.read_text())
    try:
        cmd = pathlib.Path('/proc/' + str(pid) + '/cmdline').read_bytes()
        if rid.encode() not in cmd:
            raise RuntimeError('Worker PID ownership mismatch')
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        pass
"""
        subprocess.run(self._command(source, request_id), cwd=self.workspace,
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)

    def _wait_motion(self, process, timeout):
        if self.motion_guard is None:
            return process.wait(timeout=timeout)
        deadline = time.monotonic() + timeout
        while True:
            self.motion_guard()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired('FR3 motion worker', timeout)
            try:
                result = process.wait(timeout=min(.1, remaining))
            except subprocess.TimeoutExpired:
                continue
            self.motion_guard()
            return result

    def _request(self, operation, pose=None, *, goal_joints=None, grasp_candidates=None):
        if operation == "move" and not self.execution_enabled:
            raise RuntimeError("Movement requires execution_enabled=True (CLI: --execute)")
        if operation == 'move' and self.motion_guard is not None:
            self.motion_guard()
        request_id = uuid.uuid4().hex
        request = dict(request_id=request_id, operation=operation, stack=self.config.stack,
                       timeout_s=self.config.request_timeout_s,
                       speed_scaling=self.config.speed_scaling,
                       position_tolerance_m=self.config.position_tolerance_m,
                       orientation_tolerance_rad=self.config.orientation_tolerance_rad)
        request['singularity_screening'] = self.config.singularity_screening
        if pose is not None:
            if not isinstance(pose, Pose6D):
                raise TypeError("Expected Pose6D in metres/radians")
            request["pose"] = pose.values
        if goal_joints is not None:
            import math
            if len(goal_joints) != 7 or not all(math.isfinite(v) for v in goal_joints):
                raise ValueError('Expected seven finite pinned joint angles')
            request['goal_joints'] = list(goal_joints)
        if grasp_candidates is not None:
            request['grasp_candidates'] = grasp_candidates
        source = Path(__file__).with_name("fr3_worker.py").read_text()
        interrupted = None
        with tempfile.TemporaryFile(mode="w+") as output:
            process = subprocess.Popen(self._command(source, json.dumps(request, allow_nan=False)),
                                       cwd=self.workspace, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                # Worker has its own deadline, then time to cancel/deactivate.
                if operation == 'move':
                    self._wait_motion(process, self.config.request_timeout_s + 45)
                else:
                    process.wait(timeout=self.config.request_timeout_s + 45)
            except (KeyboardInterrupt, Exception) as error:
                interrupted = error
                try:
                    self._cancel(request_id)
                    process.wait(timeout=40)
                except (Exception, KeyboardInterrupt) as cancel_error:
                    raise MotionError({"error": f"Cannot verify remote cleanup: {cancel_error!r}. "
                                       "Check Desk and existing stack status before another request.",
                                       "host_log_path": str(self.workspace / "logs/agent" / f"{request_id}.jsonl")}) from error
            output.seek(0)
            transcript = output.read()
        rows = [s for s in transcript.splitlines() if s.startswith("ARM_AGENT_RESULT=")]
        if not rows:
            raise MotionError({"error": "No worker result. Ensure the selected Docker service is running. "
                               + transcript[-2500:]})
        result = json.loads(rows[-1].split("=", 1)[1])
        if result.get("request_id") != request_id:
            raise MotionError({"error": "Unexpected worker request id"})
        if result.get("log_path"):
            result["host_log_path"] = str(self.workspace / Path(result["log_path"]).relative_to("/project"))
        self.last_result = result
        if interrupted is not None:
            if not result.get("cleanup_ok"):
                raise MotionError(result) from interrupted
            raise interrupted
        if process.returncode != 0 or not result.get("ok") or not result.get("cleanup_ok"):
            raise MotionError(result)
        return result

    def get_pose(self):
        report = self._request("pose")["current_pose"]
        return Pose6D(*report["pose_xyz_rpy"], base_frame=report["base_frame"], target_link=report["target_link"])

    def plan_pose(self, pose):
        return self._request("plan", pose)

    def move_pose(self, pose):
        """Plan from the current state, execute once, check endpoint, deactivate."""
        return self._request("move", pose)

    def plan_grasps(self, candidates):
        """Parallel kinematic screening + a single selected MoveIt plan. No motion."""
        if not candidates:
            raise ValueError('No grasp candidates')
        return self._request('plan_grasps', grasp_candidates=candidates)

    def move_joint_goal(self, pose, goal_joints):
        """Explicit operator use: replan to the selected joint configuration."""
        return self._request('move', pose, goal_joints=goal_joints)

    def home(self):
        if self.config.home_pose is None:
            raise RuntimeError("arm.home_pose 未配置；不会使用旧 xArm 的 home 坐标")
        return self.move_pose(Pose6D(*self.config.home_pose))

    def close(self):
        # Each blocking request owns and cleans up its own ROS session.
        pass
