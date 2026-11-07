# Developer architecture

## Runtime shape

The full application is a Streamlit interface over the HPC workspace filesystem. A submission creates a run bundle and sends one guarded Python worker directly to SLURM; the worker invokes the selected workflow adapter, which records stage events and artifacts. Workflow adapters own their modality-specific processing and may call external scientific tools. Run state and provenance are files in each run directory, not a shared service or database.

```text
Streamlit UI
  → workflow registry + preflight / explicit mapping
  → run bundle + guarded Python worker submitted to SLURM
  → workflow adapter → stage events and run-local artifacts
  → QC, results, and files views read the saved bundle
```

## Deployment modes

One application exposes the full workspace by default, including on Streamlit Community Cloud and without `SLURM_JOB_ID`; `[toolkit] mode = "hpc"` also selects it. An operator can explicitly select `[toolkit] mode = "read_only"` to expose only workflow selection and introductions, without project edits, file browsing, run controls, review actions, or artifact access. Invalid toolkit settings are reported without silently hiding the full workspace. The optional `[toolkit] datasets_url` adds an external Google Drive link; it does not mount Drive.

Interface availability is distinct from execution capability. The app reports when `sbatch` is absent, but scheduler availability and authorization are ultimately checked on submission; failures are surfaced, not replaced with local execution. Submission requires an authorized SLURM backend and server-accessible approved paths. HPC filesystem paths do not become available to Community Cloud merely by deploying the interface; Cloud resource and storage limits also apply.

## Main components

- [`src/spatial_omics_toolkit/application/ui/presentation.py`](src/spatial_omics_toolkit/application/ui/presentation.py) owns the Slate & Indigo workbench visual system, dark-slate sidebar navigation styling, horizontal main-page section tabs, compact modality summary banner, and workflow-stage presentation.
- [`apps/streamlit/app.py`](apps/streamlit/app.py) composes project and modality workbenches, maintains per-modality UI state, renders workflow parameters, confirms sample mappings, and presents preflight and submission controls in side-by-side split panes. Sidebar navigation provides Home, Project, and modality destinations plus an active study status indicator; the current route is stored in `spatial_page`. In full mode, each modality workspace opens action-first on **Data & settings** (`Setup`) with a compact summary banner and a horizontal section bar on the main canvas (`Data & settings`, `Run`, `Quality control`, `Results`, `Files`, and `Introduction`). In `read_only` mode, each modality opens on `Introduction`. The `Introduction` section renders the currently selected workflow descriptor (purpose, stages, tools, inputs, outputs, and limitations) and its conditional stage-and-tool diagram without triggering preflight or execution. Run selection and QC/results/files views are implemented in [`src/spatial_omics_toolkit/application/ui/results.py`](src/spatial_omics_toolkit/application/ui/results.py); source-method details are rendered in [`src/spatial_omics_toolkit/application/ui/implementation.py`](src/spatial_omics_toolkit/application/ui/implementation.py).
- [`src/spatial_omics_toolkit/application/workflows/registry.py`](src/spatial_omics_toolkit/application/workflows/registry.py) discovers workflow modules that implement `describe`, `preflight`, and `execute`. Descriptors define workflow identity, stages, parameters, source references, inputs, outputs, and limitations. Registration marks an implemented adapter as available; it is not a qualification claim.
- [`src/spatial_omics_toolkit/application/service.py`](src/spatial_omics_toolkit/application/service.py) prepares requests, combines shared and modality preflight, and creates/submits runs. [`src/spatial_omics_toolkit/application/intake.py`](src/spatial_omics_toolkit/application/intake.py) validates project metadata and input scope. [`src/spatial_omics_toolkit/application/worker.py`](src/spatial_omics_toolkit/application/worker.py) verifies the frozen sample-sheet snapshot, records/validates the workflow's input inventory, checks mappings, invokes the selected adapter, and records stage outcomes. [`src/spatial_omics_toolkit/application/scheduler.py`](src/spatial_omics_toolkit/application/scheduler.py) submits that worker directly with `sbatch`.
- Shared request, descriptor, event, and result types are in [`src/spatial_omics_toolkit/application/contracts.py`](src/spatial_omics_toolkit/application/contracts.py). File-backed run creation, status, and event updates are in [`src/spatial_omics_toolkit/application/runs.py`](src/spatial_omics_toolkit/application/runs.py).
- Modality adapters live under [`src/spatial_omics_toolkit/application/workflows/`](src/spatial_omics_toolkit/application/workflows/). Current adapters cover CycIF, hyperspectral imaging (HS), FLIM, H&E, Visium, and optional exploratory Integration. Their input requirements, stage boundaries, and limitations are declared per adapter. Reusable CycIF analytical operations are in [`src/spatial_omics_toolkit/cycif/`](src/spatial_omics_toolkit/cycif/); other shared analysis helpers are in [`src/spatial_omics_toolkit/analysis/`](src/spatial_omics_toolkit/analysis/). Library code not exposed through a registered adapter is not an app workflow.

## Request, identity, and execution

An own-data request uses a project root, one modality folder, `project_samples.csv`, an output root, a workflow ID, and resolved parameters. Shared validation requires `biological_unit_id`, `specimen_id`, `condition`, and `tissue_source`; each row represents one physical specimen/section, and `specimen_id` is unique. A biological unit may have multiple specimens. Workflows that report acquisition groups require an explicit mapping from each exact group key to a reviewed physical specimen, or an explicit `UNMAPPED` technical-QC scope. The worker checks those mappings against the sample sheet; filenames and acquisition groups do not establish biological identity.

Preparation runs shared and workflow preflight before enabling submission. Submission snapshots the sample sheet and creates a unique run directory. The worker verifies the snapshot, records the modality input inventory, and runs the adapter's preflight again before execution. For a human-review resume, it also revalidates the recorded inventory against current inputs before continuing. The adapter emits stage events with status, parameters, tool versions, and artifact paths. `BLOCKED`, `FAILED`, `WARNING`, and `QC_PENDING` are distinct outcomes; later stages are not implied by an earlier successful stage.

A `QC_PENDING` workflow may request human review. The decision is written to the run and resumed through another worker attempt after checking the review and input receipts. CycIF additionally supports selected child-run continuations from validated checkpoints; those receipts permit reuse only at the implemented boundaries and do not imply general resume support for other modalities.

## Run bundle and review

Each run is stored beneath `<output-root>/<modality>/runs/<run-id>/`. Its core records include `run.json` (workflow and run metadata), `status.json`, `job.json`, `resolved_parameters.json`, append-only `events.jsonl`, `artifacts.json`, `mapping_receipt.json`, and `inputs/` with the sample-sheet snapshot and input manifest. Logs, the generated worker wrapper, review records, modality-specific receipts, and analysis outputs are present as applicable. The UI reads these records for QC, results, and files views and uses bounded previews rather than loading full-resolution data into the browser.

Some workflows depend on canonical analytical scripts outside this repository. Configure those source directories on the compute workers before enabling those workflows.

## Extending the application

Implement an adapter with `describe`, `preflight`, and `execute` in `application/workflows/`. Describe its actual inputs, parameters, stages, tools, outputs, source references, and limitations. Preflight should surface missing prerequisites as blockers or warnings; execution should emit stage events and return a workflow result using the shared application contracts. Keep modality-specific state and processing in its adapter or reusable domain module, and expose it through the UI via the descriptor rather than duplicating workflow logic in Streamlit. Add or update focused tests alongside a behavior change.
