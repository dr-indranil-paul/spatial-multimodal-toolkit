from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from spatial_omics_toolkit.application import scheduler
from spatial_omics_toolkit.application.config import (
    WORKSPACE_ROOTS_ENV,
    containing_workspace_root,
    default_output_root,
    load_site_config,
    package_source_dir,
)
from spatial_omics_toolkit.application.contracts import RunRequest
from spatial_omics_toolkit.application.intake import path_is_under_workspace, validate_path_scope

_SITE_VARIABLES = (
    WORKSPACE_ROOTS_ENV,
    "SPATIAL_TOOLKIT_PYTHON",
    "SPATIAL_TOOLKIT_ENV_SCRIPT",
    "SPATIAL_TOOLKIT_EXTRA_PYTHONPATH",
    "SPATIAL_TOOLKIT_SLURM_ACCOUNT",
    "SPATIAL_TOOLKIT_SLURM_PARTITION",
)


@pytest.fixture(autouse=True)
def _clear_site_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SITE_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _request(run_dir: Path) -> RunRequest:
    project = run_dir.parent / "project"
    return RunRequest(
        workflow_id="visium.native_qc",
        modality="Visium",
        project_root=project,
        modality_dir=project / "data",
        annotation_csv=project / "project_samples.csv",
        output_root=run_dir.parent,
        run_id=uuid4(),
        run_dir=run_dir,
        parameters={"resources": {"cpus": 2, "memory_gb": 4, "time_limit": "8h"}},
    )


def _capture_sbatch(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = list(args)
        return subprocess.CompletedProcess(args, 0, stdout="4242\n", stderr="")

    monkeypatch.setattr(scheduler.shutil, "which", lambda name: "/usr/bin/sbatch")
    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
    return captured


def test_unset_variables_reproduce_the_lab_installation(tmp_path: Path) -> None:
    config = load_site_config()
    assert config.workspace_roots == (Path("/home/groups/EmiliLab/pauli").resolve(),)
    assert config.python == Path("/home/groups/EmiliLab/pauli/00_environments/bin/spatial-python")
    assert config.env_script == Path("/home/groups/EmiliLab/pauli/00_environments/bin/spatial_env.sh")
    assert config.slurm_account is None
    assert config.slurm_partition is None
    assert default_output_root() == Path("/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_toolkit/runs")


def test_configured_roots_accept_either_root_and_reject_sibling_and_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lab_a = tmp_path / "labA"
    lab_b = tmp_path / "labB"
    sibling = tmp_path / "labA-other"
    outside = tmp_path / "outside"
    for directory in (lab_a, lab_b, sibling, outside):
        directory.mkdir()
    (lab_a / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv(WORKSPACE_ROOTS_ENV, f"{lab_a}:{lab_b}")

    assert path_is_under_workspace(lab_a / "project")
    assert path_is_under_workspace(lab_b / "project")
    assert not path_is_under_workspace(sibling / "project")
    assert not path_is_under_workspace(lab_a / "escape")
    assert not path_is_under_workspace(Path("/home/groups/EmiliLab/pauli"))


def test_unapproved_path_message_names_the_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    approved = tmp_path / "approved"
    other = tmp_path / "elsewhere"
    approved.mkdir()
    other.mkdir()
    monkeypatch.setenv(WORKSPACE_ROOTS_ENV, str(approved))
    errors = validate_path_scope(other, "Project root")
    assert len(errors) == 1
    assert WORKSPACE_ROOTS_ENV in errors[0]


def test_containing_root_is_the_root_that_holds_the_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lab_a = tmp_path / "labA"
    lab_b = tmp_path / "labB"
    (lab_b / "project").mkdir(parents=True)
    lab_a.mkdir()
    monkeypatch.setenv(WORKSPACE_ROOTS_ENV, f"{lab_a}:{lab_b}")
    assert containing_workspace_root(lab_b / "project") == lab_b.resolve()
    with pytest.raises(ValueError, match=WORKSPACE_ROOTS_ENV):
        containing_workspace_root(tmp_path / "outside")


def test_default_output_root_follows_the_first_configured_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lab = tmp_path / "lab"
    lab.mkdir()
    monkeypatch.setenv(WORKSPACE_ROOTS_ENV, str(lab))
    assert default_output_root() == lab.resolve() / "spatial_toolkit_runs"


def test_job_script_and_sbatch_use_configured_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python = tmp_path / "site env" / "bin" / "python3"
    env_script = tmp_path / "site env" / "activate.sh"
    overlay = tmp_path / "overlay"
    monkeypatch.setenv("SPATIAL_TOOLKIT_PYTHON", str(python))
    monkeypatch.setenv("SPATIAL_TOOLKIT_ENV_SCRIPT", str(env_script))
    monkeypatch.setenv("SPATIAL_TOOLKIT_EXTRA_PYTHONPATH", str(overlay))
    monkeypatch.setenv("SPATIAL_TOOLKIT_SLURM_ACCOUNT", "acct-1")
    monkeypatch.setenv("SPATIAL_TOOLKIT_SLURM_PARTITION", "part-1")
    captured = _capture_sbatch(monkeypatch)
    run_dir = tmp_path / "my runs" / "run-a"
    scheduler.submit_controller(_request(run_dir))

    script = (run_dir.resolve() / "worker-attempt-0.sh").read_text(encoding="utf-8")
    assert f"source {shlex.quote(str(env_script))} || exit $?" in script
    assert shlex.quote(str(python)) in script
    assert str(package_source_dir()) in script
    assert str(overlay) in script
    assert shlex.quote(str(run_dir.resolve())) in script

    args = captured["args"]
    assert "--account=acct-1" in args
    assert "--partition=part-1" in args


def test_job_script_omits_optional_site_settings_when_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPATIAL_TOOLKIT_ENV_SCRIPT", "")
    captured = _capture_sbatch(monkeypatch)
    run_dir = tmp_path / "run-b"
    scheduler.submit_controller(_request(run_dir))

    script = (run_dir.resolve() / "worker-attempt-0.sh").read_text(encoding="utf-8")
    assert "source " not in script
    assert not any(arg.startswith(("--account=", "--partition=")) for arg in captured["args"])
