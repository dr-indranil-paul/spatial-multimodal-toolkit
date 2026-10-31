# Spatial Omics Toolkit

A Python application suite for transparent, modality-specific spatial analysis. It provides a Streamlit interface, validated project metadata, run-scoped provenance, per-stage status and logs, and SLURM-backed execution. It is an active research tool; a workflow being available in the interface does not by itself mean it is fully qualified for every assay or study.

## Supported application modalities

- **CycIF:** raw CZI intake and explicitly configured conversion, registration, marker export, and segmentation; a separate tabular-analysis workflow uses the existing toolkit steps. These are separate workflows with different inputs and requirements.
- **Hyperspectral imaging (HS):** notebook-derived, human-review-gated workflow. The reviewed final manifest and related ledgers/settings are authoritative; the app does not infer sample identity or combine raw intensity across TMAs.
- **FLIM:** notebook-derived `.sdt` workflow with explicit frequency and provenance. Block selection is paused for review when needed. Lifetime values remain uncalibrated apparent values unless a measured reference with matching settings is supplied.
- **H&E:** source-preserving image intake and bounded previews. The app does not infer pairing or perform automatic cross-modal registration.
- **Visium:** Space Ranger output intake through SpatialData-IO for Visium HD and Scanpy for classic Visium. The HD bin-size selector exposes only complete resolutions shared by all detected HD bundles. HD runs select one 2, 8, or 16 µm table and require exactly one feature-slice file, selected-bin `tissue_positions.parquet`, and bin scale factors. Bins are platform-native spatial bins, not cells. The HD reader receives a temporary view with only those required inputs and an optional validated low-resolution image; full-resolution/hires imagery and segmentation payloads are not loaded.
- **Other assays (including BODIPY):** not currently registered as app workflows. Add a modality adapter only after its input contract, methods, QC, outputs, and limitations are explicitly reviewed and tested.

**PLA is intentionally not an application modality.** Existing library code outside the app is not offered as an app workflow.

## Project metadata CSV

Place a `project_samples.csv` in the project folder and select that file in the app. The app provides a header-only download template. Required columns are:

```csv
biological_unit_id,specimen_id,condition,tissue_source,batch_or_block,section_id
```

The first four columns are required by the shared validator. `batch_or_block` and `section_id` are optional. Add one row per physical specimen/section. A `biological_unit_id` may repeat for multiple specimens from the same independent unit. Every `specimen_id` must be unique. Do not infer identity from filenames or folder names. Use the literal `unknown` when a required source value is unavailable; do not guess.

The CSV provides reviewed project-wide metadata. It does not replace modality-specific source manifests, assignment files, QC review, or explicit mappings when a workflow requires them.

## Inputs and workflow behavior

Select the project root, the complete folder for **one modality**, and the project output root. Keep modalities in separate folders and run them independently. Use the typed workflow controls to specify acquisition-specific settings. The app does not guess a modality from folder names, discover biological identity by filename, or silently apply an example marker panel.

The UI follows the Python app at `02_projects_current/02_cfms_emt/05_companion_website`: a persistent sidebar radio router provides **Home**, **Project & Data**, one page per implemented modality, **Review & Submit**, and **Run & Reports**. Each assay page uses a left-controls/right-details layout with nested **Overview**, **Stages & tools**, and **Limitations** tabs. The navy/blue hero, pale background, white cards, compact type scale, and table-first run summaries make inputs, tools, and results easier to scan; exact paths, receipts, and logs remain expandable.

Project root, modality folder, and project CSV paths must be lexically under `/home/groups/EmiliLab/pauli`; outputs must resolve under the same workspace. For group-hosted inputs outside that tree (for example, the ChinData Visium bundles), use an approved symlink beneath the selected project folder rather than entering the external path directly. The input inventory records the path and symlink target.

If a workflow reports acquisition groups, map each group to an exact `specimen_id` or leave it `UNMAPPED` for technical QC only, then explicitly confirm the mapping. Unmapped groups cannot enter biological aggregation or comparison. The H&E image-intake and CyCIF no-group paths require explicit confirmation of technical-QC-only scope. Workflows with their own reviewed source records still enforce those records; an empty mapping interface does not imply identity was inferred.

Some steps intentionally stop as **BLOCKED** or **QC_PENDING** when required tools, settings, calibrations, assignments, or human review are missing. A successful earlier step does not imply later stages ran. Tool names, versions, resolved settings, warnings, QC records, and limitations are recorded with the run.

## Outputs and image handling

Each submission creates a unique run folder beneath the selected output root:

```text
<output-root>/<modality>/runs/<run-id>/
  run.json
  status.json
  job.json
  events.jsonl
  resolved_parameters.json
  mapping_receipt.json       # explicit mapping or QC-only scope receipt
  inputs/project_samples.csv # read-only snapshot used by this run
  inputs/manifest.json       # hashed inputs, inventory scope, and snapshot receipt
  review_request.json        # when a workflow pauses for user review
  review_decision.json      # after explicit QC review
  review_decisions.jsonl    # review history
  artifacts.json
  logs/
  worker-attempt-N.sh       # exact guarded SLURM worker wrapper
  previews/                   # bounded previews only
  ... modality-specific outputs
```

The app keeps original full-resolution data and plots on disk. It copies the reviewed project CSV into the run as a read-only snapshot and uses that snapshot for worker mapping and validation; if the source changes between preflight and submission, submission is blocked. The UI only displays bounded previews and small artifacts; oversized images and files are omitted from inline display. PDF pages are rasterized to bounded previews on the worker. The complete artifact inventory provides file paths, roles, sizes, stage IDs, and source relationships. Review full-resolution files with an appropriate local image viewer, not by loading them into the browser.

The inventory records its scope. Workflows with exact dependency lists, such as Visium, hash the selected count matrices, feature-slice metadata, coordinates, scale metadata, and any validated low-resolution image supplied to the reader; unrelated Space Ranger BAM, Loupe, and segmentation payloads are not hashed. Other adapters keep a complete modality-folder inventory unless they declare a narrower explicit input list.

## Execution and environment

The app runs as a single-user Streamlit interface. Each analysis submission is one guarded Python worker job submitted to SLURM; workflows record their own stage events and call qualified tools as needed. Heavy analysis and package installation must run on allocated compute nodes, not login nodes. The worker uses the institute's guarded `spatial-python` launcher and preserves the repository source path for the run.

The optional dependency groups are declared in `pyproject.toml` (`app`, `scientific`, `visium`, `pathology`, `flim`, `previews`, and `imaging`). The approved `spatial-proteomics` environment supplies the core scientific stack. Add-on UI, Visium, and pathology packages are pinned in `/home/groups/EmiliLab/pauli/00_environments/config/spatial-toolkit-app.lock.txt` and installed in the isolated `00_environments/python-libs/spatial-toolkit-20261030` overlay. This keeps the shared scientific environment unchanged. Install or update packages only on allocated compute nodes; do not modify shared packages without following their maintenance procedure.

The FLIM adapter also uses the existing `sdtfile` distribution under `00_environments/python-packages/sdtfile-2024.5.24`; its directory is included in the worker/app `PYTHONPATH`.

To provision the isolated overlay, submit/run this only inside an allocated compute job:

```bash
source /home/groups/EmiliLab/pauli/00_environments/bin/spatial_env.sh
target="$PAULI_FAST_SOFTWARES/python-libs/spatial-toolkit-20261030"
mkdir -p "$target"
/home/groups/EmiliLab/pauli/00_environments/bin/spatial-python -m pip install \
  --no-deps --target "$target" \
  -r /home/groups/EmiliLab/pauli/00_environments/config/spatial-toolkit-app.lock.txt
```

The `--no-deps` install keeps the pinned add-on packages from replacing the base scientific stack; dependencies not in this lock are supplied by the approved `spatial-proteomics` environment.

After provisioning, launch Streamlit from an allocated compute node with the toolkit source on `PYTHONPATH`:

```bash
export PYTHONPATH="/home/groups/EmiliLab/pauli/00_environments/python-libs/spatial-toolkit-20261030:/home/groups/EmiliLab/pauli/00_environments/python-packages/sdtfile-2024.5.24:/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_toolkit/src${PYTHONPATH:+:$PYTHONPATH}"
/home/groups/EmiliLab/pauli/00_environments/bin/spatial-python -m streamlit run \
  /home/groups/EmiliLab/pauli/02_projects_current/02_spatial_toolkit/apps/streamlit/app.py \
  --global.developmentMode=false --server.address 127.0.0.1 --server.port 8501
```

Use the institute-approved compute allocation and access/tunnel method for the Streamlit port. Do not run this command on a login node. The application is currently a local-filesystem, single-user interface; authentication, multi-user scheduling, a shared database, and cross-user run visibility are not implemented.

## Developer checks

Run tests and application smoke checks only on an allocated compute node with the approved Python environment. The test suite lives in `tests/unit/`. Before reporting a run, inspect its `status.json`, stage events, warnings, logs, and artifacts; do not infer success from job submission alone.

## Existing Python API

The package also contains CyCIF image-processing and tabular-analysis functions. See `src/spatial_omics_toolkit/cycif/` and the relevant tests for those APIs. The Streamlit app exposes only workflows registered by its modality adapters; library functionality that is not part of an adapter should not be presented as an app capability.

## License

Apache-2.0
