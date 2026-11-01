from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from typing import Any

from spatial_omics_toolkit.application.contracts import (
    FunctionReference,
    ParameterField,
    PreflightResult,
    RunRequest,
    StageDefinition,
    StageEvent,
    ToolReference,
    WorkflowDescriptor,
    WorkflowEmitter,
    WorkflowResult,
)

_WORKFLOW_ID = "hs.clinical-roi-hyperspectral"
_SCRIPT_RELATIVE = Path("03_analysis/01_exploratory_analysis/02_scripts/clinical_roi_hyperspectral.py")


def describe() -> WorkflowDescriptor:
    source_tool = ToolReference(
        name="clinical_roi_hyperspectral.py",
        version="project source; verified SHA-256 recorded in cache receipts",
        role="Canonical HS metadata audit and exploratory phasor analysis",
    )
    return WorkflowDescriptor(
        workflow_id=_WORKFLOW_ID,
        modality="HS",
        title="Clinical ROI hyperspectral audit and phasors",
        description="Runs the existing clinical HS source workflow against its strict final manifest and evidence ledgers, retaining separate TMA analyses.",
        stages=[
            StageDefinition(stage_id="metadata_audit", title="Metadata audit", purpose="Validate reviewed HS assignment evidence and audit native metadata and exact bands.", tools=[source_tool], implementations=[FunctionReference(source_file="external:clinical_roi_hyperspectral.py", function_name="load_final_analysis_manifest", role="Load reviewed source manifest"), FunctionReference(source_file="external:clinical_roi_hyperspectral.py", function_name="audit_spectral_channel_maps", role="Audit exact spectral bands")]),
            StageDefinition(stage_id="source_phasors", title="Source-bound phasors", purpose="Compute per-TMA first-harmonic summaries with the canonical exact-band map and integrity cache.", tools=[source_tool], implementations=[FunctionReference(source_file="external:clinical_roi_hyperspectral.py", function_name="analyze_tma_spectral_phasors", role="Compute per-TMA source-bound phasors")]),
            StageDefinition(stage_id="exploratory", title="Exploratory diagnostics", purpose="Optional per-TMA filter or GMM sensitivity using the declared source APIs; IDs are not tissue classes.", tools=[source_tool], implementations=[FunctionReference(source_file="external:clinical_roi_hyperspectral.py", function_name="analyze_tma_phasor_filter_sensitivity", role="Optional filter sensitivity"), FunctionReference(source_file="external:clinical_roi_hyperspectral.py", function_name="analyze_tma_phasor_gmm_sensitivity", role="Optional GMM sensitivity")]),
        ],
        input_description="Project tree with canonical source module, strict final HS manifest, assignment manifest, queue, selection ledger, settings manifest, and the exact reviewed raw acquisitions referenced by them.",
        output_description="Run-local metadata and band-map tables, separate per-TMA phasor summaries/figures/receipts, cache and cache-event log.",
        parameter_fields=[
            ParameterField(
                name="max_working_bytes", label="Maximum working memory per channel",
                kind="int", default=536870912, min_value=1, unit="bytes",
                section="Phasor analysis", stage_id="source_phasors", advanced=True,
                help_text="Positive per-channel working-memory limit used by the canonical reader.",
            ),
            ParameterField(
                name="num_threads", label="Threads", kind="int", default=1,
                min_value=1, section="Phasor analysis", stage_id="source_phasors",
                advanced=True, help_text="Thread limit passed to the canonical source analysis.",
            ),
            ParameterField(
                name="exploratory_diagnostic", label="Exploratory diagnostic",
                kind="select", default="none", choices=["none", "filter", "gmm"],
                section="Exploratory diagnostics", stage_id="exploratory",
                help_text="Optional canonical per-TMA filter or GMM sensitivity; does not name tissue classes.",
            ),
            ParameterField(
                name="cluster_counts", label="GMM cluster counts", kind="multiselect",
                default=[2, 3, 4, 6, 8, 12, 16], choices=[2, 3, 4, 6, 8, 12, 16],
                section="Exploratory diagnostics", stage_id="exploratory", advanced=True,
                help_text="Prespecified component counts used only when GMM is selected.",
            ),
            ParameterField(
                name="max_samples", label="Maximum phasor samples per acquisition",
                kind="int", default=50000, min_value=1, max_value=50000,
                section="Exploratory diagnostics", stage_id="exploratory", advanced=True,
                help_text="Systematic raw G/S sample cap for canonical GMM diagnostics.",
            ),
            ParameterField(
                name="random_state", label="GMM random state", kind="int", default=0,
                min_value=0, max_value=4294967295,
                section="Exploratory diagnostics", stage_id="exploratory", advanced=True,
                help_text="Deterministic initialization seed passed to the canonical source API.",
            ),
        ],
        limitations=[
            "Human-reviewed identities and selections are required; no inference or fallback to provisional inputs.",
            "TMA1 and TMA2 remain separate; cross-TMA raw-intensity comparison is prohibited.",
            "Exploratory spectral clusters are diagnostics, not tissue-class labels.",
            "Display galleries are review aids, not quantitative classifications.",
        ],
    )


def _artifact(root: Path, name: str) -> Path | None:
    matches = sorted(path.resolve() for path in root.rglob(name) if path.is_file())
    return matches[0] if len(matches) == 1 else None


def _paths(request: RunRequest) -> tuple[Path, dict[str, Path | None]]:
    script = (request.project_root / _SCRIPT_RELATIVE).resolve()
    if not script.is_file():
        candidate = request.modality_dir / _SCRIPT_RELATIVE.name
        script = candidate.resolve() if candidate.is_file() else script
    names = {
        "final_manifest": "hyperspectral_final_analysis_manifest.csv",
        "ledger": "hyperspectral_tma1_manual_core_selection.csv",
        "assignment_manifest": "hyperspectral_assignment_and_processing_manifest.csv",
        "settings_manifest": "hyperspectral_settings_provisional_working_set.csv",
        "queue": "hyperspectral_assignment_and_processing_queue.csv",
    }
    found: dict[str, Path | None] = {}
    for key, name in names.items():
        selected = _artifact(request.modality_dir, name)
        found[key] = selected if selected is not None else _artifact(request.project_root, name)
    found["script"] = script if script.is_file() else None
    return script, found


def _load_source(script: Path) -> Any:
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("_omp_clinical_roi_hyperspectral", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load canonical HS source module: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def preflight(request: RunRequest) -> PreflightResult:
    blockers: list[str] = []
    script, paths = _paths(request)
    if paths["script"] is None:
        blockers.append(f"Required canonical source module is absent or cannot be located at the project source path: {script}")
    for key in ("final_manifest", "ledger", "assignment_manifest", "queue", "settings_manifest"):
        if paths[key] is None:
            blockers.append(f"Required unique HS prerequisite is absent or ambiguous: {key}; expected its canonical artifact basename in the selected modality folder or project tree.")
    acquisition_groups: list[dict[str, Any]] = []
    if paths["final_manifest"] is not None:
        try:
            with paths["final_manifest"].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required_columns = {"path", "assigned_tma", "confirmed_core_id"}
                if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                    blockers.append(
                        "Strict HS final manifest lacks path, assigned_tma, or confirmed_core_id columns."
                    )
                else:
                    for row in reader:
                        source_path = row["path"]
                        if not source_path:
                            blockers.append("Strict HS final manifest contains an empty acquisition path.")
                            continue
                        acquisition_groups.append({
                            "group_key": f"acquisition:{source_path}",
                            "label": f'{row["assigned_tma"]} / {row["confirmed_core_id"]}',
                            "input_paths": [source_path],
                        })
        except (OSError, UnicodeError, csv.Error) as exc:
            blockers.append(f"Cannot read HS final manifest acquisition rows: {type(exc).__name__}: {exc}")
    resolved: dict[str, Any] = {
        key: str(value) for key, value in paths.items() if value is not None
    }
    resolved["selected_modality_folder"] = str(request.modality_dir.resolve())
    resolved["tma_runs"] = ["TMA1", "TMA2"]
    resolved["acquisition_groups"] = acquisition_groups
    return PreflightResult(ready=not blockers, blockers=blockers, resolved_parameters=resolved)


def execute(request: RunRequest, emit: WorkflowEmitter) -> WorkflowResult:
    check = preflight(request)
    if not check.ready:
        event = StageEvent(stage_id="preflight", status="BLOCKED", message="; ".join(check.blockers), parameters=check.resolved_parameters)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event])
    script, paths = _paths(request)
    try:
        source = _load_source(script)
    except Exception as exc:
        blocker = f"Canonical HS source module cannot be safely imported in the worker: {type(exc).__name__}: {exc}"
        event = StageEvent(stage_id="preflight", status="BLOCKED", message=blocker, parameters=check.resolved_parameters)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event])


    out = request.run_dir / "hs"
    tables, plots = out / "tables", out / "plots"
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    cache = source.HSCache(
        out / "cache", manifest_path=paths["assignment_manifest"],
        queue_path=paths["queue"], force=bool(request.parameters.get("force", False)),
    )
    stages: list[StageEvent] = []
    artifacts: list[str] = []
    warnings = [
        "Exploratory outputs are spectral diagnostics, not tissue classes.",
        "TMA runs are independent; cross-TMA raw-intensity comparison is prohibited.",
    ]

    def publish(stage_id: str, status: str, message: str, params: dict[str, Any], files: list[Path]) -> None:
        event = StageEvent(
            stage_id=stage_id, status=status, message=message,
            tool_versions={
                "clinical_roi_hyperspectral.py": source._full_file_hash(script),
                "phasorpy": _package_version("phasorpy"),
            },
            parameters=params,
            artifacts=[str(path) for path in files],
        )
        stages.append(event)
        emit(event)

    try:
        final = source.load_final_analysis_manifest(
            paths["final_manifest"], ledger_path=paths["ledger"],
            assignment_manifest_path=paths["assignment_manifest"],
            queue_path=paths["queue"], settings_manifest_path=paths["settings_manifest"],
        )
        metadata, channels = source.audit_spectral_channel_maps(final, cache=cache)
        metadata_path, channel_path = tables / "hs_metadata_audit.csv", tables / "hs_exact_channel_map.csv"
        metadata.to_csv(metadata_path, index=False)
        channels.to_csv(channel_path, index=False)
        artifacts.extend((str(metadata_path), str(channel_path)))
        publish("metadata_audit", "SUCCEEDED", "Strict human-reviewed manifest validated; native metadata and exact spectral bands audited.", {"final_manifest": str(paths["final_manifest"]), "manifest_sha256": cache.manifest_sha256, "queue_sha256": cache.queue_sha256}, [metadata_path, channel_path])
        specimen_mapping = request.parameters.get("specimen_mapping", {})
        mapping_confirmed = request.parameters.get("specimen_mapping_confirmed") is True
        required_group_keys = [f"acquisition:{value}" for value in final["path"]]
        complete_mapping = (
            mapping_confirmed
            and isinstance(specimen_mapping, dict)
            and len(required_group_keys) == len(final["path"])
            and all(
                isinstance(specimen_mapping.get(key), str)
                and specimen_mapping[key].strip()
                and specimen_mapping[key].strip().upper() != "UNMAPPED"
                for key in required_group_keys
            )
        )
        if not complete_mapping:
            import json
            event_path = tables / "hs_cache_events.json"
            event_path.write_text(json.dumps(cache.events, indent=2, sort_keys=True), encoding="utf-8")
            artifacts.append(str(event_path))
            warning = "Specimen mapping is absent, unconfirmed, incomplete, or UNMAPPED; only metadata/QC output is retained. Specimen-level phasor aggregation is blocked and no identity is inferred."
            warnings.append(warning)
            publish(
                "specimen_mapping", "WARNING", warning,
                {"required_group_keys": required_group_keys,
                 "mapping_confirmed": mapping_confirmed,
                 "mapped_group_keys": sorted(specimen_mapping) if isinstance(specimen_mapping, dict) else []},
                [],
            )
            return WorkflowResult(status="WARNING", stages=stages, artifacts=artifacts, warnings=warnings)

        max_bytes = int(request.parameters.get("max_working_bytes", 512 * 1024**2))
        num_threads = int(request.parameters.get("num_threads", 1))
        summaries: dict[str, Any] = {}
        for tma in ("TMA1", "TMA2"):
            summary, spectrum_pdf, phasor_pdf = source.analyze_tma_spectral_phasors(
                final, channels, assigned_tma=tma, cache=cache, table_dir=tables,
                plot_dir=plots, max_working_bytes=max_bytes, num_threads=num_threads,
            )
            summary_path = tables / f"{tma.lower()}_spectral_phasor_acquisitions.csv"
            receipt_path = Path(summary.attrs["analysis_receipt_path"])
            outputs = [summary_path, Path(spectrum_pdf), Path(phasor_pdf), receipt_path]
            artifacts.extend(str(path) for path in outputs)
            summaries[tma] = summary
            publish("source_phasors", "SUCCEEDED", f"Completed canonical first-harmonic analysis for {tma} only.", {"assigned_tma": tma, "max_working_bytes": max_bytes, "num_threads": num_threads, "exact_band_map_sha256": source._hs_channel_map_sha256(channels), "raw_source_hashing": "full SHA-256 per selected CZI"}, outputs)

        diagnostic = request.parameters.get("exploratory_diagnostic")
        if diagnostic == "none":
            diagnostic = None
        if diagnostic is not None:
            if diagnostic not in {"filter", "gmm"}:
                raise ValueError("exploratory_diagnostic must be 'filter', 'gmm', or omitted")
            for tma in ("TMA1", "TMA2"):
                if diagnostic == "filter":
                    _, paths_out = source.analyze_tma_phasor_filter_sensitivity(
                        summaries[tma], assigned_tma=tma, cache=cache, table_dir=tables, plot_dir=plots,
                    )
                else:
                    _, paths_out = source.analyze_tma_phasor_gmm_sensitivity(
                        summaries[tma], assigned_tma=tma, cache=cache, table_dir=tables, plot_dir=plots,
                        cluster_counts=tuple(request.parameters.get("cluster_counts", (2, 3, 4, 6, 8, 12, 16))),
                        max_samples=int(request.parameters.get("max_samples", 50000)),
                        random_state=int(request.parameters.get("random_state", 0)),
                    )
                outputs = [Path(value) for value in paths_out]
                artifacts.extend(str(path) for path in outputs)
                publish("exploratory", "WARNING", f"{diagnostic} sensitivity completed separately for {tma}; results are exploratory only.", {"assigned_tma": tma, "diagnostic": diagnostic, "cluster_counts": request.parameters.get("cluster_counts"), "max_samples": request.parameters.get("max_samples", 50000)}, outputs)

        import json
        event_path = tables / "hs_cache_events.json"
        event_path.write_text(json.dumps(cache.events, indent=2, sort_keys=True), encoding="utf-8")
        artifacts.append(str(event_path))
        return WorkflowResult(status="SUCCEEDED", stages=stages, artifacts=artifacts, warnings=warnings)
    except Exception as exc:
        event = StageEvent(stage_id="hs_analysis", status="FAILED", message=f"{type(exc).__name__}: {exc}", tool_versions={"clinical_roi_hyperspectral.py": source._full_file_hash(script)}, parameters=request.parameters, artifacts=artifacts)
        stages.append(event)
        emit(event)
        return WorkflowResult(status="FAILED", stages=stages, artifacts=artifacts, warnings=warnings)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"
