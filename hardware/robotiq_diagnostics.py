"""Bounded read-only communication check; never activates or moves the gripper."""
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import struct
import time
from zoneinfo import ZoneInfo

from .config import PROJECT_ROOT
from .robotiq import GripperError, Robotiq2F85


class StatusOnlyGripper(Robotiq2F85):
    """Restrict even the transport to one read-only register request."""
    def __init__(self, config, *, serial_factory=None):
        super().__init__(config, execution_enabled=False, serial_factory=serial_factory)

    def _exchange(self, payload, function):
        if function != 3 or payload != struct.pack(">HH", 0x07D0, 3):
            raise GripperError("Communication check permits only FC03 status reads at 0x07D0")
        return super()._exchange(payload, function)


def summarize(samples, *, warmup_s, duration_s, interval_s, completed):
    good = [s for s in samples if "status" in s]
    errors = [s for s in samples if "error" in s]
    steady = [s for s in good if s["elapsed_s"] >= warmup_s]
    faults = Counter(s["status"]["fault_hex"] for s in good)
    steady_faults = [s for s in steady if s["status"]["fault"] != 0]
    unexpected = [s for s in good if s["status"]["fault"] not in (0, 9)]
    latencies = sorted(s["latency_ms"] for s in good)
    gaps = [b["elapsed_s"] - a["elapsed_s"] for a, b in zip(good, good[1:])]
    clear = next((s for s in good if s["status"]["fault"] == 0), None)
    return {
        "completed": completed,
        "communication_passed": completed and bool(steady) and not errors and
                                not steady_faults and not unexpected and max(gaps, default=0) < 1.0,
        "all_steady_samples_ready": bool(steady) and all(s["status"]["ready"] for s in steady),
        "configured_duration_s": duration_s, "poll_interval_s": interval_s,
        "warmup_s": warmup_s, "attempted_reads": len(samples), "successful_reads": len(good),
        "failed_reads": len(errors), "steady_samples": len(steady),
        "fault_counts": dict(faults), "steady_nonzero_fault_samples": len(steady_faults),
        "first_fault_free_s": clear["elapsed_s"] if clear else None,
        "response_latency_ms": {
            "median": statistics.median(latencies) if latencies else None,
            "p95": latencies[max(0, math.ceil(len(latencies) * .95) - 1)] if latencies else None,
            "max": max(latencies, default=None),
        },
        "max_read_start_gap_s": max(gaps, default=None),
        "position_raw_min": min((s["status"]["position_raw"] for s in good), default=None),
        "position_raw_max": max((s["status"]["position_raw"] for s in good), default=None),
        "max_current_ma": max((s["status"]["current_ma"] for s in good), default=None),
        "go_to_true_samples": sum(s["status"]["go_to"] for s in good),
        "first_status": good[0]["status"] if good else None,
        "last_status": good[-1]["status"] if good else None,
        "errors": errors,
    }


def check_communication(config, *, duration_s=60.0, interval_s=.1, output_dir=None, progress=None):
    if not math.isfinite(duration_s) or duration_s < 3:
        raise ValueError("duration must be finite and at least 3 seconds")
    if not math.isfinite(interval_s) or not .005 <= interval_s <= .5:
        raise ValueError("interval must be between 0.005 and 0.5 seconds")
    folder = Path(output_dir) if output_dir else PROJECT_ROOT / "runs/hardware_validation"
    folder.mkdir(parents=True, exist_ok=True)
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    stem = "robotiq-comm-" + started.strftime("%Y%m%d-%H%M%S-%f")
    trace_path = folder / (stem + ".jsonl")
    report_path = folder / (stem + ".json")
    samples = []
    begin = time.monotonic()
    completed = False
    interrupted = False
    next_progress = 0.0
    with trace_path.open("x") as trace, StatusOnlyGripper(config) as gripper:
        try:
            while time.monotonic() - begin < duration_s:
                before = time.monotonic()
                sample = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                          "elapsed_s": before - begin}
                try:
                    sample["status"] = gripper.status().to_dict()
                except Exception as error:
                    sample["error"] = repr(error)
                sample["latency_ms"] = (time.monotonic() - before) * 1000
                samples.append(sample)
                trace.write(json.dumps(sample) + "\n")
                trace.flush()
                if progress and sample["elapsed_s"] >= next_progress:
                    progress(dict(elapsed_s=round(sample["elapsed_s"], 1), reads=len(samples),
                                  fault=sample.get("status", {}).get("fault_hex"),
                                  ready=sample.get("status", {}).get("ready"), error=sample.get("error")))
                    next_progress = sample["elapsed_s"] + 10
                if "error" in sample:
                    break  # Preserve first failure; no reconnect/reset or hidden retry.
                time.sleep(max(0, min(begin + duration_s, before + interval_s) - time.monotonic()))
            else:
                completed = True
        except KeyboardInterrupt:
            interrupted = True
    report = {
        "started_at": started.isoformat(), "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "port": config.port, "baudrate": config.baudrate, "slave_id": config.slave_id,
        "mode": "read_only", "allowed_function_codes": [3], "status_register": "0x07D0",
        "registers_per_read": 3, "write_register_requests": 0, "interrupted": interrupted,
        **summarize(samples, warmup_s=2.0, duration_s=duration_s, interval_s=interval_s, completed=completed),
        "trace_path": str(trace_path), "report_path": str(report_path),
        "scope": "Communication only; no activation, open/close, force or grasp validation. "
                 "Polling stops when this command exits; this is not the Agent keepalive service.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
