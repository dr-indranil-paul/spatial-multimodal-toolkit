from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from spatial_omics_toolkit.application.contracts import RunRequest
from spatial_omics_toolkit.application.runs import append_event, record_slurm_job, update_status, update_status_if





def submit_controller(request: RunRequest) -> str:
    """Submit one guarded Python worker directly to SLURM."""
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise RuntimeError("sbatch is not available on PATH; workflow was not submitted")
    run_dir = request.run_dir.resolve()
    attempt = int(request.parameters.get("review_attempt", 0))
    resources = request.parameters.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("Resolved SLURM resources must be an object")
    cpus = resources.get("cpus", 1)
    memory_gb = resources.get("memory_gb", 8)
    time_limit = resources.get("time_limit", "72h")
    if not isinstance(cpus, int) or isinstance(cpus, bool) or not 1 <= cpus <= 32:
        raise ValueError("SLURM task CPU request must be from 1 through 32")
    if not isinstance(memory_gb, int) or isinstance(memory_gb, bool) or not 1 <= memory_gb <= 512:
        raise ValueError("SLURM task memory request must be from 1 through 512 GiB")
    time_limits = {"8h": "08:00:00", "24h": "24:00:00", "48h": "48:00:00", "72h": "72:00:00", "168h": "168:00:00"}
    if time_limit not in time_limits:
        raise ValueError("Unsupported SLURM task time limit")
    script = run_dir / f"worker-attempt-{attempt}.sh"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    python_launcher = "/home/groups/EmiliLab/pauli/00_environments/bin/spatial-python"
    quoted_run_dir = shlex.quote(str(run_dir))
    script.write_text(
        "#!/bin/bash\nset -uo pipefail\n"
        "export PYTHONPATH=/home/groups/EmiliLab/pauli/00_environments/python-libs/spatial-toolkit-20261030:/home/groups/EmiliLab/pauli/00_environments/python-packages/sdtfile-2024.5.24:/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_toolkit/src${PYTHONPATH:+:$PYTHONPATH}\n"
        "source /home/groups/EmiliLab/pauli/00_environments/bin/spatial_env.sh || exit $?\n"
        f"{shlex.quote(python_launcher)} -m spatial_omics_toolkit.application.worker --run-dir {quoted_run_dir}\n"
        "worker_rc=$?\n"
        f"{shlex.quote(python_launcher)} -m spatial_omics_toolkit.application.controller --run-dir {quoted_run_dir} --exit-code \"$worker_rc\"\n"
        "controller_rc=$?\n"
        "if [[ \"$controller_rc\" -ne 0 ]]; then exit \"$controller_rc\"; fi\n"
        "exit \"$worker_rc\"\n",
        encoding="utf-8",
    )
    script.chmod(0o750)
    job_resources = {"cpus": cpus, "memory_gb": memory_gb, "time_limit": time_limit, "attempt": attempt}
    env = os.environ.copy()
    update_status(run_dir, "QUEUED", "Submitting guarded Python workflow job to SLURM", resources=job_resources, attempt=attempt)
    result = subprocess.run(
        [
            sbatch,
            "--parsable",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={cpus}",
            f"--time={time_limits[time_limit]}",
            f"--mem={memory_gb}G",
            f"--chdir={run_dir}",
            f"--output={log_dir}/worker-attempt-{attempt}-%j.out",
            f"--error={log_dir}/worker-attempt-{attempt}-%j.err",
            str(script),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch returned {result.returncode}: {(result.stderr or result.stdout).strip()}")
    job_id = result.stdout.strip().split(";")[0]
    if not job_id:
        raise RuntimeError("sbatch returned an empty job ID")
    record_slurm_job(request.run_dir, job_id, "guarded_python_worker", job_resources, attempt)
    append_event(request.run_dir, {"stage_id": "submission", "status": "RUNNING", "message": "Guarded Python workflow job submitted to SLURM", "tool_versions": {}, "parameters": {"slurm_job_id": job_id, "resources": job_resources}, "artifacts": [str(script)]})
    return job_id


def read_run_status(run_dir: Path) -> dict[str, object]:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return {"status": "UNKNOWN", "message": "status.json is not available"}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(status, dict):
            return {"status": "UNKNOWN", "message": "status.json does not contain an object"}
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "UNKNOWN", "message": f"Unable to read status.json: {exc}"}
    if status.get("status") not in {"SUBMITTED", "QUEUED", "RUNNING"}:
        return status
    job_path = run_dir / "job.json"
    sacct = shutil.which("sacct")
    if not job_path.is_file() or sacct is None:
        return status
    try:
        job_metadata = json.loads(job_path.read_text(encoding="utf-8"))
        job_id = job_metadata.get("slurm_job_id") if isinstance(job_metadata, dict) else None
    except (OSError, json.JSONDecodeError):
        return status
    if not isinstance(job_id, str) or not job_id:
        return status
    try:
        accounting = subprocess.run(
            [sacct, "--noheader", "--parsable2", "--jobs", job_id, "--format=JobIDRaw,State,ExitCode"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return status
    if accounting.returncode != 0:
        return status
    job_state: str | None = None
    exit_code: str | None = None
    for line in accounting.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] == job_id:
            job_state = fields[1].split("+", 1)[0].split()[0]
            exit_code = fields[2]
            break
    if job_state is None:
        return status
    terminal_failures = {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "OOM",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
    if job_state == "COMPLETED" or job_state in terminal_failures:
        message = (
            f"SLURM job ended as {job_state} (exit {exit_code}) before the workflow "
            "recorded a terminal run status."
        )
        if update_status_if(
            run_dir,
            {"SUBMITTED", "QUEUED", "RUNNING"},
            "FAILED",
            message,
            expected_job_id=job_id,
            slurm_state=job_state,
            exit_code=exit_code,
        ):
            append_event(
                run_dir,
                {
                    "stage_id": "slurm_reconciliation",
                    "status": "FAILED",
                    "message": message,
                    "tool_versions": {},
                    "parameters": {
                        "slurm_job_id": job_id,
                        "slurm_state": job_state,
                        "exit_code": exit_code,
                    },
                    "artifacts": [],
                },
            )
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"status": "UNKNOWN", "message": "Unable to read reconciled status.json"}
    return status
