from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_omics_toolkit.application.runs import append_event, update_status


TERMINAL = {"SUCCEEDED", "WARNING", "FAILED", "BLOCKED", "QC_PENDING"}


def finish_controller(run_dir: Path, exit_code: int) -> None:
    status_path = run_dir / "status.json"
    status = "UNKNOWN"
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status", "UNKNOWN")
        except (OSError, json.JSONDecodeError):
            status = "UNKNOWN"
    if status in TERMINAL:
        return
    if exit_code == 0:
        message = "Python workflow worker completed, but did not record a terminal workflow status"
    else:
        message = f"Python workflow worker exited with status {exit_code} before recording a terminal state"
    append_event(run_dir, {"stage_id": "controller", "status": "FAILED", "message": message, "tool_versions": {}, "parameters": {"worker_exit_code": exit_code}, "artifacts": []})
    update_status(run_dir, "FAILED", message, worker_exit_code=exit_code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()
    finish_controller(args.run_dir, args.exit_code)


if __name__ == "__main__":
    main()
