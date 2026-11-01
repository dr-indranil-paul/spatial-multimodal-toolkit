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

Select the project root, one complete modality folder, and the project output root for an own-data workflow. Use the server-side Browse dialogs to choose any folder or CSV under `/home/groups/EmiliLab/pauli`; the dialogs support direct path entry and folder navigation. **Input mode** opens with a prepared local example. These defaults are fixed technical-QC examples, not biological analyses: groups remain `UNMAPPED`, and clicking **Run example** explicitly authorizes the displayed technical scope. Example runs use separate prepared metadata, need no Project-page setup, and remain isolated from own-data state and history. The worker guards execution against source-hash changes since preparation. Own-data mode remains separate; supply reviewed metadata before submitting a biological run, and the app does not invent sample identities. Keep modalities in separate folders and run them independently. The app does not guess a modality from folder names, discover biological identity by filename, or silently apply an example marker panel.

For CycIF raw-CZI intake, the default quickstart uses one `r1` file under `fixtures/example_data/quickstart/cycif/data/slide_a1/`. It has no OME sidecar, so intake reports **PARTIAL** QC and does not convert or segment by default. The larger `fixtures/example_data/cycif_raw/` view remains available separately; it groups the full copied raw collection by slide using relative symlinks.

The sidebar provides **Home**, **Project**, each modality by its full name, and optional **Integration**. Open a modality to complete the analysis in one place: **Data & settings**, **Run**, **Quality control**, **Results**, and **Files**. There are no separate global submission or reporting pages. Project stores shared metadata; each modality retains its own input folder, output location, settings, checks, and selected run. The Run section checks the current configuration, requests any required specimen-mapping confirmation, and repeats preflight before enabling submission. Switching modality does not clear another modality's work.

Home has direct project and modality entry points. Advanced execution settings, software versions, source identities, and complete provenance are available on demand rather than occupying the main analysis controls. CyCIF still configures intake, optional conversion, registration, marker export, and MCMICRO segmentation independently; choosing a settings stage does not launch a job.

Open the collapsed **Methods & source** panel to inspect the selected processing step or result. It shows the function's purpose, script, current definition line, and a copyable full path. When a step uses several functions, select one rather than scan a long technical listing. Saved-run software/source identifiers are shown separately from current source definitions; unavailable locations are never guessed. Results initially selects a report or figure before operational files; the Files section retains the complete bundle.

For saved FLIM and native-HS runs, the panel also reads script SHA-256 identifiers from their known, bounded workflow receipts inside that run bundle. These entries are labelled workflow-level, not step-specific call traces, and are not recomputed from current source files. Raw input-file hashes are not presented as code identities. Files retains the complete provenance record.

A worker-generated PDF-preview stage identifies the actual preview renderer in `application/worker.py`, rather than an unrelated analytical function. The original report's processing step contains its analytical-method references.

**Integration is optional and exploratory.** Its prepared example previews a translation of a serial-section H&E moving image against a CycIF r1 DAPI reference. This is not validated biological pairing, an accepted alignment, or generic joint analysis. CyCIF registration aligns acquisition rounds within CyCIF; it does not establish cross-modal alignment. Each modality can be run and reviewed on its own.

If a source checkpoint was **WARNING**, continuation requires explicit confirmation that its QC report was reviewed. The child run records that decision and the warning stages, and the source limitations remain visible in the resumed run.

Each CyCIF raw-start inventories the complete selected raw folder and the current `project_samples.csv`. Continuation is a separate child run, not a mutation of the source bundle. Intake checkpoints can be reused to continue at **conversion** or **registration**: conversion requires `conversion.enabled=true`; registration requires `registration.enabled=true` and intentionally skips the optional, independent conversion deliverable. Both boundaries reuse intake without rerunning step01, then execute the configured stage and later configured stages. Later supported boundaries are **after registration** (continue at export using a durable per-scene handoff containing registered TIFF stacks and transform/round metadata) and **after export** (continue at segmentation from the verified composite and marker map). All continuation runs re-inventory raw files and project metadata, verify checkpoint artifact hashes, stage settings, tool identity, source workflow, and each required upstream stage's successful terminal status. The parent run itself may be BLOCKED or FAILED after a completed checkpoint; that does not invalidate a separately verified successful upstream stage. The child writes `resume_receipt.json` with parent lineage, start stage, reuse event and reused artifact hashes, and the source bundle remains unchanged. `checkpoint_receipts.json` binds successful stage artifacts to the raw-input fingerprint and resolved stage settings; legacy, incomplete, or stale checkpoints cannot be reused. Warning checkpoints remain reusable only after explicit review confirmation. Do not add an outer Nextflow orchestrator for this CyCIF module: the stages are user-configured and review-gated, and explicit checkpoint receipts and child-run lineage remain necessary regardless of runner; Nextflow `-resume` would not replace raw-input and artifact verification. MCMICRO continues to invoke its existing Nextflow pipeline. Evaluate other modules independently, and adopt Nextflow only when nontrivial parallel execution, resource isolation, or external-tool orchestration justifies the added runtime. Each run remains one guarded Python worker/SLURM submission. CyCIF tabular analysis retains its existing interface.

Project root, modality folder, and project CSV paths must be lexically under `/home/groups/EmiliLab/pauli`; outputs must resolve under the same workspace. For group-hosted inputs outside that tree (for example, the ChinData Visium bundles), use an approved symlink beneath the selected project folder rather than entering the external path directly. The input inventory records the path and symlink target.

For own-data workflows that report acquisition groups, map each exact `group_key` to a physical `specimen_id` and its independent `biological_unit_id`, or leave it `UNMAPPED` for technical QC only, then explicitly confirm the mapping. An optional core map CSV has columns `group_key,specimen_id` and may also include `slide_id`, `core_id`, or `section_id`. Keys must match exactly and group keys must be unique; map only reviewed, known sample IDs. Any change invalidates checks and requires checking again. The UI downloads the populated group keys after **Check inputs**. Per-core analysis requires core-indexed inputs; the app does not detect cores or infer a core count. Unmapped groups cannot enter biological aggregation or comparison. The H&E image-intake and CyCIF no-group paths require explicit confirmation of technical-QC-only scope. Workflows with their own reviewed source records still enforce those records; an empty mapping interface does not imply identity was inferred.

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
  checkpoint_receipts.json   # CyCIF raw-stage outputs, hashes, settings, and tool identity
  resume_receipt.json        # continuation child lineage and reused artifacts, when applicable
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

## Local example data

The prepared local defaults are created with `python -m spatial_omics_toolkit.application.examples --prepare`. Sources are local fixtures in `fixtures/example_data/`; the prepared catalog stores relative paths and source provenance, and avoids hashing all raw data during UI rendering. A source-hash guard runs in the worker. Examples use their own explicit technical-QC metadata and require no Project-page setup; they do not change or bypass own-data metadata and review requirements.

| Example | Prepared technical-QC scope and limits |
| --- | --- |
| **CycIF** | One `r1` raw-CZI slide intake. Missing OME sidecar means **PARTIAL** QC. No conversion or segmentation by default. |
| **HS** | Native 20-band CZI intensity and phasor QC. |
| **FLIM** | Real SDT, all-native-block counts and decay QC only—not lifetimes or manual block review. |
| **H&E** | Biopsy SVS thumbnail QC. |
| **Visium** | Real official tiny HD, 16 µm QC only—not biological conclusions. The public source is [10x Genomics Visium HD example data](https://www.10xgenomics.com/support/software/space-ranger/latest/resources/visium-hd-example-data), licensed CC BY 4; ZIP MD5 `7cab710801d3776125a7aa5a792cd7e4`. |
| **Integration** | Exploratory serial-section H&E moving ↔ CycIF r1 DAPI-reference translation preview—not validated biological pairing, accepted alignment, or generic joint analysis. |

Private microscopy originals remain user-restricted and are not redistributed; local fixture copies are available only in the local workspace. Relative symlinks provide quickstart views of those copies. The old collaborator `05` folder is only a placeholder and is not used; `05_spatial_omics/visium_hd_tiny` is the real Visium source. These local data do not make the toolkit a self-contained public example-data package.

For a prepared example, **Single sample** describes one specimen, **Multiple samples** multiple specimens, and **TMA layout** a tissue microarray layout. In all cases the reviewed mapping is `group_key` → physical `specimen_id` → independent `biological_unit_id`. For own data, provide a core map when the workflow needs one; technical example defaults keep groups `UNMAPPED`.

<details>
<summary>Methods &amp; source</summary>

The prepared-example definitions are in `src/spatial_omics_toolkit/application/examples.py`; current workflow definitions are in `src/spatial_omics_toolkit/application/workflows/`, with modality methods in their referenced source modules. A run's recorded implementation version/source hash receipt describes the code identity recorded for that historical run, not necessarily the code currently in the checkout. Consult the recorded run artifacts for its method and provenance.

</details>

## Existing Python API

The package also contains CyCIF image-processing and tabular-analysis functions. See `src/spatial_omics_toolkit/cycif/` and the relevant tests for those APIs. The Streamlit app exposes only workflows registered by its modality adapters; library functionality that is not part of an adapter should not be presented as an app capability.

## License

Apache-2.0
