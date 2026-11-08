# Developer architecture

## Runtime shape

The full application is a Streamlit interface over the HPC workspace filesystem. A submission creates a run bundle and sends one guarded controller worker to SLURM. After validating input identity, mappings, and frozen settings, that worker runs a generated Nextflow DSL2 workflow with a process for each declared stage. Nextflow uses its local executor inside that allocation; controller and task resources share the same allocation. Adapters own modality-specific processing and may call external scientific tools. Run state and provenance remain files, not a database or shared service.

```text
Streamlit UI
  → workflow registry + preflight / explicit mapping
  → run bundle + guarded Python worker submitted to SLURM
  → Nextflow descriptor DAG → stage adapters → events and run-local artifacts
  → QC, results, and files views read the saved bundle
```

## Deployment modes

One application exposes the full workspace by default, including on Streamlit Community Cloud and without `SLURM_JOB_ID`; `[toolkit] mode = "hpc"` also selects it. An operator can explicitly select `[toolkit] mode = "read_only"` to expose only workflow selection and introductions, without project edits, file browsing, run controls, review actions, or artifact access. Invalid toolkit settings are reported without silently hiding the full workspace. The optional `[toolkit] datasets_url` adds an external Google Drive link; it does not mount Drive.

Interface availability is distinct from execution capability. The app reports when `sbatch` is absent, but scheduler availability and authorization are ultimately checked on submission; failures are surfaced, not replaced with local execution. Submission requires an authorized SLURM backend and server-accessible approved paths. HPC filesystem paths do not become available to Community Cloud merely by deploying the interface; Cloud resource and storage limits also apply.

## Main components

- [`src/spatial_omics_toolkit/application/ui/presentation.py`](src/spatial_omics_toolkit/application/ui/presentation.py) owns the Slate & Indigo workbench visual system, dark-slate sidebar navigation styling, bold blue horizontal section-tab labels, white text on active primary buttons, compact modality summary banner, and workflow-stage presentation.
  Shared control typography applies to text fields, select inputs and portalled dropdown options, including controls in file-picker dialogs; retain the compact input scale and separate label hierarchy when adding widgets.
  The home introduction uses the available column width and wraps naturally as the viewport narrows.
  Introduction diagrams retain complete node and dependency annotations. Graphviz `dot` lays out the SVG; `packages.txt` declares the system dependency for deployment. SVG viewBox units map to CSS pixels at 1:1 scale, with the app's exact text sizes applied separately from Graphviz layout metrics. Trusted generated SVG is rendered through Markdown with dynamic labels escaped; `st.html` strips SVG in the supported Streamlit version and must not be used for the connector layer. The accessible text panel is an equivalent representation, not a substitute for diagram information.
  Diagram field labels are bold; descriptions remain normal-weight. Wrap annotations at word boundaries and size each node to its longest line. Long tool identifiers must remain intact rather than being split or clipped.
- [`apps/streamlit/app.py`](apps/streamlit/app.py) composes the existing project and modality split-pane workbenches and maintains per-modality state. Sidebar navigation, summary banners, and the horizontal section bar remain unchanged. New modality workspaces open on **Introduction** in both full and read-only modes; normal navigation and Run actions retain their state. Introduction diagrams and accessible details use the app's typography and the execution descriptor's dependencies, without triggering preflight or execution. Run selection and QC/results/files views are in [`application/ui/results.py`](src/spatial_omics_toolkit/application/ui/results.py); source-method details are in [`application/ui/implementation.py`](src/spatial_omics_toolkit/application/ui/implementation.py).
- [`application/workflows/registry.py`](src/spatial_omics_toolkit/application/workflows/registry.py) discovers modules implementing `describe`, `preflight`, and `execute_stage`. `StageDefinition.depends_on`, conditions, inputs, outputs, and tool references are the common execution and diagram contract. Registration is not a scientific qualification claim.
- [`application/service.py`](src/spatial_omics_toolkit/application/service.py) prepares requests and combines shared and modality preflight. [`application/worker.py`](src/spatial_omics_toolkit/application/worker.py) verifies the frozen sample sheet, input inventory, mappings, and configuration files before execution. [`application/nextflow.py`](src/spatial_omics_toolkit/application/nextflow.py) validates the DAG, writes DSL2 processes, executes stage calls with upstream receipts, and combines outcomes. [`application/scheduler.py`](src/spatial_omics_toolkit/application/scheduler.py) submits the guarded controller with `sbatch`.
- Shared request, descriptor, event, and result types are in [`src/spatial_omics_toolkit/application/contracts.py`](src/spatial_omics_toolkit/application/contracts.py). File-backed run creation, status, and event updates are in [`src/spatial_omics_toolkit/application/runs.py`](src/spatial_omics_toolkit/application/runs.py).
- Modality adapters live under [`src/spatial_omics_toolkit/application/workflows/`](src/spatial_omics_toolkit/application/workflows/). Current adapters cover CycIF, hyperspectral imaging (HS), FLIM, H&E, Visium, and optional exploratory Integration. Their input requirements, stage boundaries, and limitations are declared per adapter. Reusable CycIF analytical operations are in [`src/spatial_omics_toolkit/cycif/`](src/spatial_omics_toolkit/cycif/); other shared analysis helpers are in [`src/spatial_omics_toolkit/analysis/`](src/spatial_omics_toolkit/analysis/). Library code not exposed through a registered adapter is not an app workflow.

## Request, identity, and execution

An own-data request uses a project root, one modality folder, `project_samples.csv`, an output root, a workflow ID, and resolved parameters. Shared validation requires `biological_unit_id`, `specimen_id`, `condition`, and `tissue_source`; each row represents one physical specimen/section, and `specimen_id` is unique. A biological unit may have multiple specimens. Workflows that report acquisition groups require an explicit mapping from each exact group key to a reviewed physical specimen, or an explicit `UNMAPPED` technical-QC scope. The worker checks those mappings against the sample sheet; filenames and acquisition groups do not establish biological identity.

Preparation runs shared and workflow preflight before enabling submission. Submission snapshots the sample sheet, source/effective settings, and referenced configuration files in a unique run directory. The worker verifies those records before preflight, and each Nextflow stage rechecks frozen configuration integrity before computation. Human-review attempts also revalidate the recorded input inventory. `execute_stage(request, stage_id, upstream, emit)` performs only its declared operation and returns a `WorkflowResult`; its `state` contains JSON metadata and file paths rather than large arrays. CycIF table handoffs use typed Parquet so strings, nulls, categories, and indexes are not lost between processes. `SKIPPED`, `BLOCKED`, `FAILED`, `WARNING`, and `QC_PENDING` remain distinct; an earlier success does not imply later execution.

A `QC_PENDING` result stops dependent tasks and preserves the human-review request. After a validated decision, a new Nextflow attempt runs in its own directory. General Nextflow cache reuse is disabled because native caching alone does not validate all external scientific outputs and environments. CycIF child-run continuations retain their explicit input, settings, code, environment, and artifact checks; these do not imply general resume support for other modalities.

## Run bundle and review

Each run is stored beneath `<output-root>/<modality>/runs/<run-id>/`. Its core records include `run.json` (workflow and run metadata), `status.json`, `job.json`, `resolved_parameters.json`, append-only `events.jsonl`, `artifacts.json`, `mapping_receipt.json`, and `inputs/` with the sample-sheet snapshot and input manifest. Logs, the generated worker wrapper, review records, modality-specific receipts, and analysis outputs are present as applicable. The UI reads these records for QC, results, and files views and uses bounded previews rather than loading full-resolution data into the browser.

`inputs/settings_source.toml` and `inputs/settings_effective.toml` record the selected/default profile and its effective values. Referenced configuration snapshots and hashes are indexed by `inputs/settings_configs.json`. Each `nextflow/attempt-<n>/` contains the request, generated `workflow.nf`, resource configuration, trace, report, DAG, task work directories, and stage receipts.

## Settings and plot presentation

[`application/settings.py`](src/spatial_omics_toolkit/application/settings.py) derives commented default TOML from the same `ParameterField` schema used by controls. User profiles are typed, workflow-bound, stored in project-local user-owned files, and cannot supply runtime authorizations or review confirmations. File hashes participate in preflight signatures and are rechecked at submission and execution. [`application/ui/settings.py`](src/spatial_omics_toolkit/application/ui/settings.py) keeps selected files and editor drafts isolated by project and workflow. Applying a profile changes the parameter-widget generation so stale browser widget values cannot replace the imported settings.

[`application/plotting.py`](src/spatial_omics_toolkit/application/plotting.py) renders bounded descriptive views of saved table and single-plane image artifacts. [`application/ui/plotting.py`](src/spatial_omics_toolkit/application/ui/plotting.py) adds opt-in controls within the existing Results pane. Exports preserve source files, use unique PDF/provenance filenames, retain physical canvas and editable text, and bind downloads to source hashes and settings. Original workflow PDFs are not edited or presented as re-plottable source data.

Some workflows depend on canonical analytical scripts outside this repository. Configure those source directories on the compute workers before enabling those workflows.

## Extending the application

Implement an adapter with `describe`, `preflight`, and `execute_stage` in `application/workflows/`. Declare real process boundaries, dependencies, conditions, inputs, outputs, tools, source references, and limitations; do not invent separate processes for inseparable scientific operations. Preflight surfaces missing prerequisites. Each stage emits events and returns its actual outcome, artifacts, and typed handoff state. Keep scientific processing in reusable domain code, preserve review and identity gates, and add focused behavioral tests for changed boundaries or invariants.
