# SpatialOmics Toolkit: Architecture and Skill Design

**Status:** Historical architecture proposal; not the current app implementation contract. See `README.md` and `02_projects_current/02_spatial_proteomics/01_documents/multimodal_spatial_workflow_discussion.md` for current scope and decisions.  
**Owner:** Dr. Indranil Paul  
**Target:** Institute-grade spatial and single-cell multiomics analytics service  
**Working version:** 0.1.0  

**Current app-suite note (2026-10-30):** The implementation now targets CycIF, HS, FLIM, H&E, and Visium, with PLA excluded from the app. The app uses a single-user Streamlit interface and one guarded Python worker job submitted directly to SLURM; the older FastAPI, multi-user service, Nextflow orchestration, tiled web viewer, and generic integration architecture below remain aspirational unless separately re-approved and implemented. Do not treat the historical capability statements in this design as shipped or qualified functionality.

**Work note (2026-08-06):** Approved direction incorporating Red Team audit hardening:
native Python core, Streamlit UI with tiled spatial rendering, FastAPI production job service,
SLURM-wrapped Nextflow DSL2 orchestration, Python-side pseudobulk pre-aggregation prior to containerized
R statistical stages, spatial null permutation controls, channel-agnostic multiomics schemas, and AI/LLM-agent API compatibility.

**Work note (2026-08-08):** Added a notebook-first, package-level analyst API on top of
this architecture (additive, not a replacement for the CLI/contracts/Nextflow production
path above): `spatial_omics_toolkit.cycif` provides 13 numbered `stepNN_*` functions
(`step01_scan_intake` through `step13_pseudobulk`), re-exported at the top-level package
namespace so `import spatial_omics_toolkit as sot; sot.step07_classify_cells(...)` works
directly in Jupyter. Ported and generalized from the PREVAIL Barrett's-EAC CyCIF TMA
research pipeline (`02_projects_current/02_spatial_proteomics`); steps 07-09 validated to
reproduce that project's own published output exactly on real data (100% cell-type
agreement, exact per-core metric match). `step11_spatial_null` fills the spatial-null-
permutation gap this document's non-negotiable #5 requires and which did not exist
anywhere in the source project. Segmentation (`step05`) wraps MCMICRO and PIPEX as
external tools rather than reimplementing segmentation — consistent with this document's
"one qualified canonical implementation" principle in §1. Every step accepts an optional
`output_dir` and writes a human-inspectable artifact (plot/image/CSV) there, per the
lab's requirement that a person can visually confirm each stage before trusting the next
one. Full step table, source mapping, and integrity notes: this project's wiki
`requirements/notebook_first_step_api` page, and the `spatial_omics_toolkit.cycif` module
docstring.

## 1. Decision

SpatialOmics Toolkit will use a Python-first, polyglot-analysis architecture:

- **Python core package and CLI** for contracts, validation, provenance, object
  adapters, pipeline planning, pseudobulk pre-aggregation, and Python scientific engines.
- **FastAPI job service** for authenticated production submission, OpenAPI contracts, and durable
  multi-user state.
- **Nextflow DSL2** (wrapped in single-core SLURM batch jobs) as the only production workflow and SLURM orchestrator.
- **Python Streamlit application** as the scientist-facing interface, equipped with server-side rasterization and OME-Zarr pyramidal tiling for gigapixel spatial imaging.
- **Containerized R stages** executing only on pre-aggregated pseudobulk tables for qualified R statistical methods (e.g. edgeR, limma-voom, muscat).
- **Apptainer containers** and centrally managed environments for production execution under `/home/groups/EmiliLab/pauli/`.

The application platform is native Python, but method selection is not restricted
by language. Each scientific computation has one qualified canonical
implementation and is never independently reimplemented in Python and R merely
for interface convenience.

## 2. Why This Architecture

### Python owns the analytical engine

The spatial ecosystem is centered on AnnData, MuData, SpatialData, Scanpy,
Squidpy, spatialdata-io, image arrays, graph methods, and scalable sparse or
chunked storage. A Python core therefore provides the broadest platform support
and the cleanest plugin boundary.

### Python owns the application platform

The primary users should not need to write Python. Streamlit provides a native
Python browser interface over the same application services and contracts used
by the CLI. This removes the R client, cross-language UI bridge, and duplicate
application packaging.

### Qualified methods may use Python or R via Pseudobulk Serialization

R remains the preferred runtime for statistical differential methods such as `edgeR`, `limma-voom`, `DESeq2`,
`dreamlet`, and `muscat`. To eliminate cross-language I/O bottlenecks and RAM crashes, **Python pre-aggregates full cell-level matrices into sample-level pseudobulk tables ($N_{\text{biological\_units}} \times G$) prior to R execution**. R stages accept lightweight Parquet/Matrix Market tables and design matrices rather than full un-aggregated cell matrices.

### Nextflow owns production execution under SLURM Batch Wrapper

Streamlit and FastAPI submit and monitor work but never perform heavy analysis. Nextflow owns process dependencies, retries, resource requests, container selection, resume behavior, trace files, and SLURM submission. To protect HPC login nodes from cgroup process termination, **FastAPI/CLI launches the `nextflow run` controller inside a dedicated single-core SLURM batch job**. All scratch and intermediate files (`workDir`) live strictly beneath `/home/groups/EmiliLab/pauli/`.

### Gigapixel Spatial Rendering via OME-Zarr & Tiled Rasterization

High-resolution spatial modalities (Xenium, Visium HD) produce gigapixel images and millions of coordinate points. The Streamlit presentation layer decouples raw array rendering by using **OME-Zarr pyramidal tiling** (via Viv/Deck.gl iframe view adapters) and **server-side Datashader rasterization** for high-density spatial scatter overlays, avoiding web-browser DOM memory limits.

## 3. Non-Negotiable Product Requirements

1. Every run has an immutable UUID, owner, schema version, input checksums,
   resolved configuration, event stream, status, artifact inventory, and output
   checksums.
2. Cells, spots, pixels, fields, regions, and sections are never silently treated
   as independent biological replicates.
3. Raw counts and measured values remain separate from normalized, corrected,
   transferred, deconvolved, smoothed, and predicted values.
4. Pseudobulk pre-aggregation is mandatory before cross-sample statistical testing; full single-cell/spatial matrices are never passed un-aggregated to R differential containers.
5. Spatial feature testing enforces explicit spatial null permutation models (distance-decay permutations or label swapping within tissue boundaries) before claiming spatial niche enrichment.
6. High-resolution spatial visualization requires OME-Zarr pyramidal tiling or server-side rasterization; raw gigapixel arrays are never streamed un-tiled to browser DOMs.
7. No universal QC cutoff is embedded as an institute truth. Plugins expose
   assay-aware metrics, cohort distributions, controls, proposed gates, and
   sensitivity results.
8. A failed qualified method may trigger an automatic fallback, but the result
   state is `noncanonical_fallback`, the substitution is prominent, and the
   bundle remains visibly noncanonical.
9. The single Streamlit interface uses progressive disclosure rather than separate
   user roles. Standard choices are shown first; advanced settings remain
   accessible with assumptions and warnings.
10. No heavy computation runs in the Streamlit process or on the login node. Nextflow master controllers run wrapped in single-core SLURM batch jobs.
11. Channel-agnostic feature contracts distinguish `feature_type` (`rna`, `protein`, `antibody_control`, `morphology`) to support spatial proteomics without schema breaks.
12. Every CLI and FastAPI service endpoint emits structured machine-readable schemas and error responses (`ERROR_CODE`, `REASON`, `REMEDIATION_HINT`) for seamless AI/LLM-agent integration.
13. The first fully qualified plugins are scRNA/snRNA, Visium, and Xenium.
14. Every successful or warning-state run emits a complete release bundle.
15. Every failed run retains enough structured evidence to diagnose and resume.

## 4. Initial Scope

### Version 0.1 platform foundation

- Versioned Pydantic and JSON Schema contracts (channel-agnostic multiomics).
- Intake manifest and hierarchy validation.
- File inventory, SHA-256, format probing, and semantic validation.
- Run planning and resource-envelope validation with `cpus <= 32`.
- Plugin discovery through Python entry points.
- Structured errors, warnings, stage states, and JSONL events with machine-readable error codes.
- Atomic immutable run-bundle writer.
- SLURM-wrapped Nextflow DSL2 orchestration skeleton and SLURM profile.
- Streamlit submission, status, report, and OME-Zarr tiled spatial artifact interface.
- HTML/PDF report contracts and machine-readable artifacts.

### First qualified scientific plugins

1. **scRNA/snRNA**
   - 10x matrix and AnnData intake.
   - Sample hierarchy and raw-count validation.
   - Adaptive QC metric generation.
   - Empty-droplet, ambient-RNA, and doublet method interfaces.
   - Normalization and representation plans.
   - Annotation evidence and uncertainty contract.
   - Sample-level pseudobulk design validation.

2. **Visium**
   - Space Ranger bundle and SpatialData intake.
   - Image, spot, tissue-mask, coordinate, and scale-factor validation.
   - Adaptive spot QC.
   - Reference availability and deconvolution planning.
   - Spatial graph and scale-sensitivity contract.
   - Sample-level differential and spatial-null planning.

3. **Xenium**
   - Native output and SpatialData intake.
   - Molecule, control-probe, cell-boundary, image, and coordinate validation.
   - Segmentation and transcript-assignment uncertainty contract.
   - Cell annotation and spatial-neighborhood planning.
   - Sample-level inference and sensitivity requirements.

4. **Spatial Proteomics & Multiplexed Imaging (CyCIF / PREVAIL Exemplar)**
   - OME-TIFF / CZI / MCMICRO bundle and SpatialData intake.
   - Channel, antibody control, morphology, cell boundary, and intensity matrix validation.
   - Tissue-mask-aware cell segmentation and spatial cell-type classification.
   - Spatial neighborhood graphs, Gini/density spatial null models, and cell-cell interaction scoring.
   - Sample-level pseudobulk protein expression and progressor vs. non-progressor risk stratification plans.

Version 0.1 will establish complete qualified paths for intake, QC, planning,
release, and reporting. Technology-specific heavy algorithms will be added and
qualified incrementally behind stable contracts.

## 5. Repository Structure

```text
02_spatial_toolkit/
|-- pyproject.toml
|-- src/spatial_omics_toolkit/
|   |-- application/          # use cases; no UI imports
|   |-- contracts/            # Pydantic models and schema versions
|   |-- domain/               # run, artifact, hierarchy, QC concepts
|   |-- io/                   # manifests, checksums, format probes
|   |-- plugins/              # discovery and interfaces
|   |-- provenance/           # events, manifests, atomic bundles
|   |-- reporting/            # report data contracts
|   |-- cli/                  # stable command adapters
|   |-- ui/                   # Streamlit-facing presentation adapters
|   `-- service/              # authenticated job-service adapter
|-- plugins/
|   |-- scrna/
|   |-- visium/
|   `-- xenium/
|-- workflows/
|   |-- main.nf
|   |-- modules/
|   |-- subworkflows/
|   |-- nextflow_schema.json
|   `-- conf/
|       |-- base.config
|       `-- slurm.config
|-- apps/streamlit/           # thin multipage Streamlit client
|-- methods/r/                # qualified R method entry points
|-- schemas/                  # published generated JSON Schemas
|-- containers/               # OCI/Apptainer build definitions
|-- tests/
|   |-- unit/
|   |-- contract/
|   |-- integration/
|   |-- numerical/
|   |-- e2e/
|   `-- performance/
|-- fixtures/                 # local research fixtures; excluded from publication
|-- docs/
|-- CHANGELOG.md
|-- CITATION.cff
|-- CONTRIBUTING.md
|-- SECURITY.md
|-- LICENSE
`-- README.md
```

## 6. Core Contracts

### Intake contract

Required fields include study ID, biological-unit ID, specimen hierarchy,
condition, batch/block, assay, platform, modality relationship, input files,
checksums, coordinate system, reference build, panel version, access class, and
intended analysis.

Unknown fields are rejected by default. Missing values distinguish `unknown`,
`not_collected`, `not_applicable`, `not_permitted`, and `pending`.

### Job contract

Each job contains:

- schema and toolkit versions;
- owner and study identifiers;
- selected pipeline and plugin versions;
- input artifact IDs and checksums;
- resolved scientific parameters;
- requested runtime resources;
- independent-unit and design declarations;
- fallback policy;
- requested report and release outputs.

### Artifact contract

Each artifact records stable ID, type, semantic role, path/URI, media and schema
version, size, checksum, producing stage, input lineage, measured/derived status,
validation state, intended use, and prohibited use.

### Stage states

```text
CREATED -> VALIDATED -> PLANNED -> QUEUED -> RUNNING -> SUCCEEDED
                                             |          |
                                             |          +-> QC_PENDING
                                             |                  |
                                             |                  +-> CANONICAL
                                             |                  +-> NONCANONICAL_FALLBACK
                                             |                  +-> REJECTED_QC
                                             +-> FAILED_RETRYABLE
                                             +-> FAILED_TERMINAL
                                             +-> CANCELLED
```

Transitions are validated and append-only. A Streamlit session is never the source
of truth for job state.

## 7. Automatic Fallback Policy

Fallbacks are declared in plugin manifests, not improvised by UI code.

1. The primary method fails with a structured code.
2. The engine checks whether that code permits a qualified fallback.
3. The fallback receives a new stage attempt and complete provenance.
4. Its result is marked `noncanonical_fallback`.
5. The Streamlit application displays a persistent warning banner.
6. Reports state the failed primary method, reason, substituted method,
   assumption differences, and claim restrictions.
7. Users may download the complete warning-state release bundle.

Fallback never converts a failed assay into a passed assay and never repairs
non-identifiable design, wrong inferential units, missing raw semantics, or
perfect condition-batch confounding.

## 8. Python CLI

Proposed stable commands:

```text
spatial-omics validate   Validate manifest, files, hierarchy, and semantics
spatial-omics plan       Resolve plugin, stages, resources, and fallback graph
spatial-omics run        Launch a validated Nextflow job
spatial-omics status     Write a machine-readable job status snapshot
spatial-omics artifacts  Write the artifact inventory for a run
spatial-omics report     Build reports from released result contracts
spatial-omics verify     Verify bundle checksums, schemas, and stage states
spatial-omics qualify    Execute plugin qualification fixtures and acceptance tests
spatial-omics schema     Export versioned JSON Schemas
```

Every command supports file output and stable JSON error responses. Human
messages remain short. Commands never print large matrices or reports to stdout.

## 9. Streamlit Application and Method Runtimes

### Streamlit responsibilities

- Guided intake and manifest construction.
- Schema-generated parameter forms.
- Preflight design and QC warnings.
- Job submission, status, cancellation, and logs.
- Report and artifact browsing.
- Prominent canonical/noncanonical state.
- Polling durable job state without blocking a user session.
- No primary computation, workflow orchestration, or shared mutable output paths.

Streamlit imports presentation and application services, not plugin internals.
Callbacks remain thin. Expensive parsing, hashing, conversion, and analysis are
submitted as jobs rather than cached in the web worker.

### Job service boundary

During single-user development, Streamlit may invoke the application submission
adapter in-process. Production deployment uses an authenticated Python job
service backed by PostgreSQL so that web-worker restarts cannot lose jobs and
multiple workers cannot race on state. In both modes, Nextflow remains the sole
SLURM owner.

### R method responsibilities

- Run only as declared Nextflow processes in pinned containers.
- Accept schema-validated file artifacts and resolved parameters.
- Emit versioned tables, objects, diagnostics, and structured status files.
- Never submit jobs, mutate another stage's outputs, or serve application state.
- Preserve native R/Bioconductor objects only as optional secondary artifacts;
  interoperable files remain the workflow boundary.

Durable file-backed job records are acceptable for local development and fixture
tests only. PostgreSQL, authenticated service submission, and institutional SSO
are required before broad concurrent production access.

## 10. Complete Release Bundle

```text
run_<uuid>/
|-- run.json
|-- status.json
|-- events.jsonl
|-- inputs/
|   |-- manifest.json
|   `-- checksums.sha256
|-- parameters/
|   |-- requested.json
|   `-- resolved.json
|-- software/
|   |-- versions.json
|   |-- containers.json
|   `-- sbom.json
|-- qc/
|   |-- metrics.parquet
|   |-- exclusions.parquet
|   `-- qc_report.html
|-- results/
|-- figures/
|-- reports/
|   |-- report.html
|   `-- report.pdf
|-- provenance/
|   |-- nextflow_trace.txt
|   |-- nextflow_report.html
|   |-- nextflow_timeline.html
|   `-- ro-crate-metadata.json
|-- artifacts.json
|-- limitations.json
|-- approvals.json
`-- checksums.sha256
```

## 11. Testing and Qualification

### Python

- Unit tests for pure domain logic.
- JSON Schema and Pydantic round-trip tests.
- Property tests for IDs, paths, transitions, and checksums.
- Installed-wheel CLI smoke tests.
- Golden run-bundle fixtures.
- Numerical tests use disk-backed reference files; no invented analytical data
  are embedded in production or plotting code.

### Streamlit and service

- Component tests for presentation adapters and session-independent state.
- AppTest smoke tests for intake, submission, status, and artifact views.
- Service contract, authentication, authorization, and concurrency tests.
- Python/R cross-language golden fixtures for each R-backed method.

### Workflow and deployment

- `nf-test` process and workflow tests.
- Minimal SLURM integration test.
- Container build and runtime smoke tests.
- Malformed upload, traversal, quota, timeout, cancellation, and isolation tests.
- Performance envelopes are measured from fixtures and production telemetry,
  never hardcoded from memory.

### Plugin qualification

Each plugin publishes intended use, prohibited use, versions, references,
fixtures, truth limitations, metrics, acceptance criteria, failure behavior,
fallbacks, and change-impact rules.

## 12. Canonical Skill

### Name

`spatial-omics-toolkit`

### Install location

Central skill library:

`/home/groups/EmiliLab/pauli/pauli_home/ai-research-companion/skills/agent_skills/life-sciences/spatial-omics-toolkit/SKILL.md`

Use this canonical skill directly; keep project setup and facts in the project README or wiki.

### Purpose

The skill instructs agents to classify incoming data, validate design and
inferential units, invoke the toolkit through stable CLI commands, submit heavy
work through Nextflow/SLURM, inspect outputs, preserve failed gates, and create
or verify complete release bundles.

### Existing skills referenced

- `scientific-analysis-integrity`: inferential units, multiplicity, selection,
  confounds, negative results, and claim labels.
- `verification-before-completion`: fresh evidence before completion claims.
- `publication-figure-defaults`: institute figure formatting where data figures
  are generated.
- `llm-wiki-automation`: source capture and durable project learning once this
  directory is registered as a wiki project.

### Skill error behavior

- Attempt retry/fallback only when the plugin declares it safe.
- Surface every fallback and downgrade.
- Fail loudly for design, identity, authority, schema, checksum, inferential
  unit, or non-identifiability defects.
- Never invent missing metadata or silently coerce malformed inputs.

### Example validation prompt

> Validate this Visium delivery, identify the independent biological units,
> create a production plan, and run only if all preflight gates pass. Return the
> report path and canonical/noncanonical status.

Expected behavior: the skill runs `validate`, writes diagnostics, runs `plan`,
submits only valid work, monitors durable status, verifies artifacts and logs,
and reports limitations and release state.

## 13. Dependencies

### Core Python dependencies

- Pydantic v2 and pydantic-settings
- Typer
- JSON Schema
- platformdirs
- standard library hashing, JSON, logging, pathlib, and UUID

Heavy scientific libraries are plugin dependencies rather than core
dependencies.

### Plugin dependencies

- AnnData and Scanpy for scRNA
- SpatialData and spatialdata-io for Visium/Xenium
- Squidpy and technology-specific packages only where qualified

### Streamlit and service dependencies

- Streamlit
- SQLAlchemy and a PostgreSQL driver for the production registry
- FastAPI for the authenticated production job service
- Institutional authentication integration selected during deployment

### Qualified R method dependencies

- R and Bioconductor versions pinned per container
- Method-specific packages such as edgeR, limma, DESeq2, dreamlet, or muscat
- jsonlite, arrow, zellkonverter, or Matrix only where required by a declared
  artifact boundary

### Workflow dependencies

- Nextflow DSL2
- nf-test
- Apptainer
- SLURM

No third-party API is required for the core workflow, so no third-party API rate
limiter is needed. Any future external network integration must be opt-in,
disclose transmitted fields, and implement its documented rate limits. The
internal FastAPI job service enforces authentication, authorization, quotas, and
submission throttling as deployment controls.

## 14. Security Boundary

- No server-volume browser.
- Per-user/per-run sandbox directories.
- Filename and archive-path validation.
- Upload size and decompression limits.
- No PHI in paths, logs, or identifiers.
- No external data egress by default.
- No user-derived raw HTML.
- Least-privilege process and filesystem access.
- Retention and deletion state recorded in the run registry.
- Institutional SSO and PostgreSQL are deployment requirements before broad
  concurrent production access.

## 15. Delivery Phases

### Phase A: Foundation

Python package, contracts, validation, provenance, CLI, tests, schemas, and
documentation.

### Phase B: Workflow

Nextflow skeleton, SLURM profile, run bundles, fixture-based smoke workflows,
and qualification runner.

### Phase C: User interface and job service

Streamlit pages and an authenticated Python job-service boundary for intake,
planning, submission, monitoring, and artifacts.

### Phase D: Qualified plugins

Complete scRNA, Visium, and Xenium qualification dossiers and end-to-end paths.

### Phase E: Institute service

SSO, PostgreSQL registry, authenticated service boundary, quotas, monitoring,
backup/recovery, release automation, and operational ownership.

## 16. Explicitly Rejected Patterns

- Monolithic Streamlit scripts with domain logic in page callbacks.
- Direct transfer of cell-level expression matrices to R containers without prior pseudobulk pre-aggregation.
- Streaming raw gigapixel image arrays directly to browser DOM elements without pyramidal tiling (OME-Zarr) or server-side rasterization.
- Running `nextflow run` master controllers directly on login nodes without single-core SLURM job wrappers.
- R analytical code imported into the web process through an embedded
  interpreter.
- Heavy computation in web workers.
- Shared mutable output folders.
- Streamlit session state or cache as the run database.
- Project-root `sys.path` mutation or a top-level `utils` package.
- Silent NA-to-zero conversion, sample-name repair, or scale inference.
- Fake biological replicates.
- Spatial cell-niche enrichment reported without spatial null permutation controls.
- Separate per-condition models presented as comparable scores.
- Cell/spot/pixel P values for patient-level claims.
- Automatic fallback presented as canonical.
- Streamlit pages or report templates that rerun a second copy of the analysis.
- Project-local Python virtual environments on the EmiliLab HPC.

## 17. Decisions Approved

Implementation is authorized with the following Red Team hardened specifications:

1. **Python Core + Streamlit UI + Nextflow DSL2 Architecture**: Native Python platform with qualified containerized R statistical stages executing over pseudobulk matrices.
2. **SLURM-Wrapped Master Process Execution**: Nextflow controllers submit as single-core SLURM batch jobs; all scratch paths reside under `/home/groups/EmiliLab/pauli/`.
3. **Gigapixel Visual Infrastructure**: Pyramidal OME-Zarr viewer & server-side datashader rendering in Streamlit.
4. **Canonical Skill Routing**: Use the central `life-sciences/spatial-omics-toolkit` skill.
5. **Version 0.1 Plugin Scope**: First qualified plugins for scRNA, Visium, and Xenium using channel-agnostic multiomics schemas.
6. **Downloadable Warning-State Bundles**: Visibly noncanonical automatic-fallback bundles tagged with explicit claim restrictions.
7. **Agent-Native APIs**: Machine-readable error taxonomies and schema export endpoints (`spatial-omics schema`).
8. **Licensing**: Apache-2.0 cleanroom implementation, subject to OHSU institutional review before public release.
