# Portable environment for spatial-omics-toolkit

This directory holds everything needed to recreate the Python environment of the
toolkit outside the Emili lab installation:

| File | Purpose |
| --- | --- |
| `requirements.lock.txt` | Exact pins (174 distributions) for CPython 3.12 on Linux x86_64. |
| `generate_lock.py` | Writes the lock from an installed environment. Typed; uses `importlib.metadata`. |
| `setup.sh` | Creates a venv at a target prefix, installs the lock, then installs this repo editable. Prints the environment variables to fill in. |
| `runtime.lock.sh` | Pins for OpenJDK 21, Nextflow 24.10.6 and the MCMICRO revision. Sets nothing else. |
| `mcmicro-containers.tsv` | Container image per MCMICRO module. |

## Install

```bash
PYTHON_BIN=python3.12 environment/setup.sh /path/to/new/venv
```

`TARGET_PREFIX` must not exist or must be empty. `PYTHON_BIN` must be CPython 3.12
(default `python3.12`). Set `SPATIAL_TOOLKIT_REQUIRE_SLURM=1` to refuse to run
outside a SLURM allocation; the default is `0`.

`setup.sh` runs these steps:

1. `python -m venv TARGET_PREFIX`
2. `pip install --requirement environment/requirements.lock.txt`
3. `pip install --editable .`
4. `pip check`

It installs Python packages only. It prints the pinned Java, Nextflow and MCMICRO
versions and whether each tool is already on `PATH`. It then prints the export
lines to fill in. Those are `SPATIAL_TOOLKIT_WORKSPACE_ROOTS`,
`SPATIAL_TOOLKIT_PYTHON`, `SPATIAL_TOOLKIT_ENV_SCRIPT`,
`SPATIAL_TOOLKIT_EXTRA_PYTHONPATH`, the optional `SPATIAL_TOOLKIT_SLURM_ACCOUNT` and
`SPATIAL_TOOLKIT_SLURM_PARTITION`, and `SPATIAL_TOOLKIT_SOFTWARE_ROOT` for
`runtime.lock.sh`.

Unset `SPATIAL_TOOLKIT_*` variables fall back to the built-in Emili lab defaults in
`src/spatial_omics_toolkit/application/config.py`. Set every variable explicitly
on any other installation. An empty `SPATIAL_TOOLKIT_ENV_SCRIPT` disables the
environment script.

## What the lock contains

- **Roots:** the base dependencies and every extra in `pyproject.toml` (`app`,
  `dev`, `flim`, `imaging`, `pathology`, `previews`, `scientific`, `valis`,
  `visium`), plus every third-party module imported under
  `src/spatial_omics_toolkit`, including `external_scripts/`.
- **Closure:** the transitive `Requires-Dist` closure, with markers evaluated for
  CPython 3.12 on Linux x86_64 and the selected extras. `tomli`, which is needed
  only below Python 3.11, is excluded.
- **Excluded:** the toolkit itself.
- **Overlay-located:** 19 pins are installed from directories outside the
  interpreter prefix (the `spatial-toolkit-20261030` overlay and
  `sdtfile-2024.5.24`). The header lists them. Each is pinned at the version the
  overlay provides. The fresh install from the index at those exact versions succeeded (see Verification).

### Declared or imported but not pinned

These are absent from the source environment. The generator cannot take versions
from an environment that lacks them, so they are not pinned. The lock header lists them too.

| Package | Why it is listed |
| --- | --- |
| `black` | `dev` extra; not installed in the source environment |
| `mypy` | `dev` extra; not installed in the source environment |
| `valis-wsi` | `valis` extra; imported by `external_scripts/clinical_roi_valis.py` and `valis_supplied_transform.py` |
| `simpleitk` | `valis` extra; imported lazily by `clinical_roi_valis.py` |

Install them separately if you need them, for example
`pip install valis-wsi simpleitk black mypy`. Those versions are unpinned. Until
`valis-wsi` is installed, `external_scripts/valis_supplied_transform.py` cannot be
imported. The same is true in the source environment.

`tomli` is an import fallback guarded by `python_version < '3.11'`. It is skipped
for this target.

## Regenerating the lock

Run this under CPython 3.12 in an environment that has the lock's versions
installed. Pass the overlay pin file so the generator fails if the environment
drifts from it:

```bash
python environment/generate_lock.py --pins <overlay-pin-file> [--strict]
```

The generator reads versions only through `importlib.metadata` and never queries an
index. It fails if an import cannot be mapped to a distribution, if an installed
requirement conflicts with another installed version, or if `--pins` disagrees with
the environment. With `--strict` it also fails when anything is unresolved.

**Last generation:** 2026-11-10 (UTC) under CPython 3.12.5 on Linux x86_64. The run
used `--pins` pointing at the Emili overlay pin file. The result was 174 pins, 19 of
them overlay-located, with 4 declared or imported packages not installed (listed
above).

**Resolved from the Emili lab environment.** The versions come from the shared
`spatial-proteomics` environment with the `spatial-toolkit-20261030` overlay and
the `sdtfile` directory. The lock was not produced by a fresh resolver run from
`pyproject.toml`. Its correctness depends on that source environment. The fresh-install
check below tests whether the pins install together and import.

## Not covered

- **Java 21.** `runtime.lock.sh` pins OpenJDK 21.0.12+1 (Temurin), the archive, and its SHA-256. It does not install Java.
- **Nextflow 24.10.6.** Pinned in `runtime.lock.sh`. Not installed.
- **Apptainer.** Not pinned, not installed. `runtime.lock.sh` appends `/opt/apptainer/current/bin` to `PATH`, which is a site path. Adjust it on other systems.
- **MCMICRO.** Revision `92c7e027dab8972d26861286a6d1b1eb8de3bbaa` is pinned. Nothing is cloned.
- **MCMICRO containers.** `mcmicro-containers.tsv` lists the images. Nothing is pulled. Images are not pinned by digest.
- **GPU and CUDA.** No CUDA, cuDNN or driver version is pinned. The lock is CPU-only.
- **Other platforms.** Only Linux x86_64 and CPython 3.12 are targets.
- **Non-Python dependencies** such as `graphviz` (`packages.txt`) and system libraries for OpenSlide or the GDAL stack.
- **Optional toolkit overrides** such as `SPATIAL_TOOLKIT_FLIM_SOURCE_DIR` and `SPATIAL_TOOLKIT_HS_SOURCE_DIR`.

## Verification

Status: **verified online on 2026-11-10; offline install not possible from the existing caches.**

- **Offline attempt (failed).** `uv pip install --offline --requirement environment/requirements.lock.txt` into a fresh CPython 3.12.11 venv, using the shared uv cache, stopped at `aicspylibczi==3.3.1` ("needs to be downloaded from a registry"). The cache does not hold every pin, so an offline install is not possible from it.
- **Fresh install (passed).** `PYTHON_BIN=<CPython 3.12.11> environment/setup.sh <new empty dir>` created a new venv, installed the lock with pip from the index, then ran `pip install --editable .` and `pip check`. Result: exit 0, "No broken requirements found."
- **Lock versus installed set (passed).** 174 lock pins, 0 version mismatches, 0 pins missing, and nothing installed beyond pip and the editable toolkit. Every overlay-located pin installed from the index at the pinned version.
- **Imports (partial).** `import spatial_omics_toolkit` succeeded, loading it from the repository's `src`. Of the 15 vendored `external_scripts` modules, 14 imported successfully, with the FLIM and hyperspectral adapters' sibling-directory path insertion applied. `valis_supplied_transform` fails with `ModuleNotFoundError: No module named 'valis'`. This is expected: `valis-wsi` is declared but not pinned. The source lab environment fails the same way.

Not verified: an offline install, and importing `valis_supplied_transform` (it needs `valis-wsi`, which is not pinned). Pins were checked by installation and version comparison only; no workflow was run.
