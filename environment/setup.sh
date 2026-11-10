#!/usr/bin/env bash
# Create a Python environment for spatial-omics-toolkit from the pinned lock.
#
# Usage: environment/setup.sh TARGET_PREFIX
#
# Creates a virtual environment at TARGET_PREFIX (it must not exist or must be empty),
# installs environment/requirements.lock.txt, then installs this repository in editable mode.
# It installs Python packages only. It never installs Java, Nextflow, Apptainer, MCMICRO
# containers, or GPU drivers; their pins are printed at the end. See environment/README.md.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: environment/setup.sh TARGET_PREFIX

Creates a virtual environment at TARGET_PREFIX (must not exist or must be empty),
installs environment/requirements.lock.txt, then installs this repository with pip install -e.

Environment:
  PYTHON_BIN                     CPython 3.12 used to create the venv (default: python3.12)
  SPATIAL_TOOLKIT_REQUIRE_SLURM  1 = refuse to run outside a SLURM allocation (SLURM_JOB_ID); default 0
EOF
}

die() {
  printf 'setup.sh: %s\n' "$*" >&2
  exit 1
}

case "${1:-}" in
  -h | --help)
    usage
    exit 0
    ;;
esac
if [[ $# -ne 1 || -z "$1" ]]; then
  usage >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
lock_file="$script_dir/requirements.lock.txt"
runtime_lock="$script_dir/runtime.lock.sh"
containers_tsv="$script_dir/mcmicro-containers.tsv"
python_bin="${PYTHON_BIN:-python3.12}"
require_slurm="${SPATIAL_TOOLKIT_REQUIRE_SLURM:-0}"

target="$1"
case "$target" in
  /*) ;;
  *) target="$PWD/$target" ;;
esac

case "$require_slurm" in
  0) ;;
  1)
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "SPATIAL_TOOLKIT_REQUIRE_SLURM=1 but SLURM_JOB_ID is unset; run inside sbatch or srun"
    ;;
  *) die "SPATIAL_TOOLKIT_REQUIRE_SLURM must be 0 or 1, got '$require_slurm'" ;;
esac

command -v "$python_bin" >/dev/null 2>&1 || die "PYTHON_BIN '$python_bin' is not on PATH"
py_version="$("$python_bin" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "$py_version" == "3.12" ]] || die "the lock targets CPython 3.12; '$python_bin' is $py_version (set PYTHON_BIN)"

if [[ -e "$target" ]]; then
  [[ -d "$target" ]] || die "'$target' exists and is not a directory"
  [[ -z "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "'$target' is not empty; choose a new TARGET_PREFIX"
fi

echo "Creating virtual environment: $target"
"$python_bin" -m venv "$target"
venv_python="$target/bin/python"

echo "Installing pinned dependencies: $lock_file"
"$venv_python" -m pip install --requirement "$lock_file"

echo "Installing spatial-omics-toolkit in editable mode: $repo_root"
"$venv_python" -m pip install --editable "$repo_root"
"$venv_python" -m pip check

# Read one pin from runtime.lock.sh without leaking its variables into this shell.
pin() {
  (
    SPATIAL_TOOLKIT_SOFTWARE_ROOT="<SPATIAL_TOOLKIT_SOFTWARE_ROOT>"
    # shellcheck source=/dev/null
    . "$runtime_lock"
    printf '%s' "${!1:-unset}"
  )
}

cat <<EOF

Python environment ready: $target

Not installed by this script. Pinned versions (see environment/runtime.lock.sh):
  Java           OpenJDK 21.0.12+1 (Temurin): $(pin JAVA_ARCHIVE)
                 sha256 $(pin JAVA_ARCHIVE_SHA256)
  Nextflow       $(pin NXF_VER)
  MCMICRO        revision $(pin MCMICRO_REVISION)
  Apptainer      not pinned here; install it per your site policy
  Containers     $containers_tsv (not pulled by this script)
  GPU            not covered
EOF
for tool in java nextflow apptainer; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "  on PATH now: $tool"
  else
    echo "  not on PATH now: $tool"
  fi
done

cat <<EOF

Fill in these exports before running workflows. Variables you leave unset fall back to
the built-in Emili lab defaults in the toolkit, so set every one explicitly:

export SPATIAL_TOOLKIT_WORKSPACE_ROOTS=""              # required: absolute workspace roots, joined with ':'
export SPATIAL_TOOLKIT_PYTHON="$venv_python"           # interpreter for workflow jobs (created above)
export SPATIAL_TOOLKIT_ENV_SCRIPT=""                   # optional: script to source before jobs; empty disables
export SPATIAL_TOOLKIT_EXTRA_PYTHONPATH=""             # optional: extra import paths, joined with ':'
export SPATIAL_TOOLKIT_SLURM_ACCOUNT=""                # optional: SLURM account for submitted jobs
export SPATIAL_TOOLKIT_SLURM_PARTITION=""              # optional: SLURM partition for submitted jobs
export SPATIAL_TOOLKIT_SOFTWARE_ROOT=""                # install root with envs/, bin/, libexec/ for runtime.lock.sh
EOF
