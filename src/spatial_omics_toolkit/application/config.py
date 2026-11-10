"""Site configuration read from the process environment.

Every value is read on each call to ``load_site_config``; nothing is captured at
import time. The defaults reproduce the Emili lab installation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_ROOTS_ENV = "SPATIAL_TOOLKIT_WORKSPACE_ROOTS"
PYTHON_ENV = "SPATIAL_TOOLKIT_PYTHON"
ENV_SCRIPT_ENV = "SPATIAL_TOOLKIT_ENV_SCRIPT"
EXTRA_PYTHONPATH_ENV = "SPATIAL_TOOLKIT_EXTRA_PYTHONPATH"
SLURM_ACCOUNT_ENV = "SPATIAL_TOOLKIT_SLURM_ACCOUNT"
SLURM_PARTITION_ENV = "SPATIAL_TOOLKIT_SLURM_PARTITION"

_DEFAULT_WORKSPACE_ROOTS = (Path("/home/groups/EmiliLab/pauli"),)
_DEFAULT_PYTHON = Path("/home/groups/EmiliLab/pauli/00_environments/bin/spatial-python")
_DEFAULT_ENV_SCRIPT = Path("/home/groups/EmiliLab/pauli/00_environments/bin/spatial_env.sh")
_DEFAULT_EXTRA_PYTHONPATH = (
    Path("/home/groups/EmiliLab/pauli/00_environments/python-libs/spatial-toolkit-20261030"),
    Path("/home/groups/EmiliLab/pauli/00_environments/python-packages/sdtfile-2024.5.24"),
)


@dataclass(frozen=True)
class SiteConfig:
    """Resolved site settings used by workspace checks and SLURM job scripts."""

    workspace_roots: tuple[Path, ...]
    python: Path
    env_script: Path | None
    extra_pythonpath: tuple[Path, ...]
    slurm_account: str | None
    slurm_partition: str | None

    @property
    def software_root(self) -> Path:
        """Installation root that holds the configured interpreter's ``bin`` directory."""
        return self.python.parent.parent


def _split_paths(variable: str, value: str) -> tuple[Path, ...]:
    entries = [entry.strip() for entry in value.split(os.pathsep) if entry.strip()]
    paths = tuple(Path(entry).expanduser() for entry in entries)
    for path in paths:
        if not path.is_absolute():
            raise ValueError(f"{variable} entries must be absolute paths: {path}")
    return paths


def _optional_text(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def load_site_config() -> SiteConfig:
    """Read site configuration from the environment; unset variables take the lab defaults."""
    roots_text = os.environ.get(WORKSPACE_ROOTS_ENV)
    if roots_text is None:
        workspace_roots = _DEFAULT_WORKSPACE_ROOTS
    else:
        workspace_roots = _split_paths(WORKSPACE_ROOTS_ENV, roots_text)
        if not workspace_roots:
            raise ValueError(f"{WORKSPACE_ROOTS_ENV} is set but names no workspace root")
    workspace_roots = tuple(path.resolve() for path in workspace_roots)

    python_text = os.environ.get(PYTHON_ENV)
    if python_text is None:
        python = _DEFAULT_PYTHON
    elif not python_text.strip():
        raise ValueError(f"{PYTHON_ENV} is set but empty")
    else:
        python = Path(python_text.strip()).expanduser()

    env_text = os.environ.get(ENV_SCRIPT_ENV)
    if env_text is None:
        env_script: Path | None = _DEFAULT_ENV_SCRIPT
    elif not env_text.strip():
        env_script = None
    else:
        env_script = Path(env_text.strip()).expanduser()

    extra_text = os.environ.get(EXTRA_PYTHONPATH_ENV)
    if extra_text is None:
        extra_pythonpath = _DEFAULT_EXTRA_PYTHONPATH
    else:
        extra_pythonpath = tuple(
            Path(entry).expanduser()
            for entry in extra_text.split(os.pathsep)
            if entry.strip()
        )

    return SiteConfig(
        workspace_roots=workspace_roots,
        python=python,
        env_script=env_script,
        extra_pythonpath=extra_pythonpath,
        slurm_account=_optional_text(SLURM_ACCOUNT_ENV),
        slurm_partition=_optional_text(SLURM_PARTITION_ENV),
    )


def workspace_roots() -> tuple[Path, ...]:
    """Return the approved workspace roots, resolved."""
    return load_site_config().workspace_roots


def first_workspace_root() -> Path:
    """Return the first approved workspace root, used as the default location."""
    return workspace_roots()[0]


def containing_workspace_root(path: Path) -> Path:
    """Return the approved root that contains ``path`` (symlinks resolved).

    Raises ``ValueError`` naming the configuration variable when no approved root contains it.
    """
    resolved = path.resolve()
    for root in workspace_roots():
        if resolved == root or resolved.is_relative_to(root):
            return root
    raise ValueError(f"{path} is outside the {describe_workspace_roots()}")


_LAB_DEFAULT_OUTPUT_ROOT = Path("/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_toolkit/runs")


def default_output_root() -> Path:
    """Default results folder: the lab location unless roots are configured, then ``<first root>/spatial_toolkit_runs``."""
    if os.environ.get(WORKSPACE_ROOTS_ENV) is None:
        return _LAB_DEFAULT_OUTPUT_ROOT
    return first_workspace_root() / "spatial_toolkit_runs"


def describe_workspace_roots() -> str:
    """Human-readable approved roots, naming the variable that controls them."""
    roots = ", ".join(str(root) for root in workspace_roots())
    return f"approved workspace roots ({WORKSPACE_ROOTS_ENV}): {roots}"


def package_source_dir() -> Path:
    """Return the directory containing the ``spatial_omics_toolkit`` package (its ``src`` root)."""
    return Path(__file__).resolve().parents[2]
