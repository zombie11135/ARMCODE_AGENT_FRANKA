"""Append-only per-run simulation events, separate from production timing."""
import json
from pathlib import Path
import time


class EventLog:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('x', encoding='utf-8')
        self.started = time.perf_counter()
        self.sequence = 0

    def write(self, event, *, episode, step, sim_time_s, **payload):
        record = dict(schema='robosuite_events_v1', backend='robosuite', simulated=True,
                      sequence=self.sequence, event=event, episode=episode, step=step,
                      sim_time_s=sim_time_s, wall_elapsed_s=time.perf_counter() - self.started, **payload)
        self.file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.file.flush()
        self.sequence += 1

    def close(self):
        self.file.close()
