"""Structured logger with JSON export for experiment tracking."""

import json
import time
import numpy as np
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, log_dir: str = "logs", run_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or f"run_{int(time.time())}"
        self._records: list[dict] = []
        self._start = time.time()

    def log(self, step: int, **kwargs):
        record = {"step": step, "t": time.time() - self._start}
        for k, v in kwargs.items():
            record[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        self._records.append(record)

    def save(self):
        path = self.log_dir / f"{self.run_name}.json"
        with open(path, "w") as f:
            json.dump(self._records, f, indent=2)
        print(f"[Logger] Saved {len(self._records)} records → {path}")
        return path

    def latest(self) -> dict:
        return self._records[-1] if self._records else {}

    def summary(self) -> dict:
        if not self._records:
            return {}
        rmses = [r["rmse"] for r in self._records if "rmse" in r]
        return {
            "run":          self.run_name,
            "total_trials": len(self._records),
            "elapsed_s":    round(time.time() - self._start, 1),
            "initial_rmse": rmses[0] if rmses else None,
            "final_rmse":   rmses[-1] if rmses else None,
        }
