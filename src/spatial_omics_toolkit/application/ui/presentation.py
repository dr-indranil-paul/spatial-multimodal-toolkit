from __future__ import annotations

from collections.abc import Callable
from html import escape
from urllib.parse import urlsplit

import streamlit as st


MODALITY_LABELS: dict[str, str] = {
    "CycIF": "Cyclic immunofluorescence (CycIF)",
    "HS": "Hyperspectral imaging (HS)",
    "FLIM": "Fluorescence lifetime imaging microscopy (FLIM)",
    "H&E": "Hematoxylin and eosin (H&E)",
    "Visium": "Visium spatial transcriptomics",
    "Integration": "Integration",
}

_MODALITY_CARDS: tuple[tuple[str, str, str], ...] = (
    ("CycIF", MODALITY_LABELS["CycIF"], "Explore multiplexed protein staining across imaging rounds."),
    ("HS", MODALITY_LABELS["HS"], "Work with spectral imaging data and its recorded channels."),
    ("FLIM", MODALITY_LABELS["FLIM"], "Examine fluorescence lifetime imaging measurements."),
    ("H&E", MODALITY_LABELS["H&E"], "Work with H&E-stained tissue images."),
    ("Visium", MODALITY_LABELS["Visium"], "Explore spatial gene-expression data from Visium assays."),
    ("Integration", MODALITY_LABELS["Integration"], "Exploratory translation-only comparison across modalities."),
)


def apply_theme() -> None:
    """Apply the toolkit's responsive, light workspace styling."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            --workspace-ink: #243139;
            --workspace-muted: #66747a;
            --workspace-paper: #fbfaf7;
            --workspace-line: #e5e4df;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(ellipse at 88% 4%, #f2eee6 0, transparent 29rem), #fbfaf7;
            color: var(--workspace-ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"], [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        [class*="st-key-workspace_section_"] [role="radiogroup"] { gap: .3rem; flex-wrap: wrap; padding-bottom: .8rem; border-bottom: 1px solid #e5e4df; }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"] { margin: 0; padding: .55rem .85rem; border-radius: 9px; }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"] > div:first-child { display: none; }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"] > div:last-child { margin-left: 0; }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"]:has(input:checked) { background: #e0e9e0; font-weight: 650; }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"]:focus-within { outline: 2px solid #327d68; }
        [data-testid="stAppDeployButton"] { display: none; }
        [data-testid="stMainBlockContainer"] {
            max-width: 1180px;
            padding-top: clamp(2rem, 5vw, 4.5rem);
            padding-bottom: 4rem;
        }
        [data-testid="stMain"] h1 { font-size: clamp(1.65rem, 2.6vw, 2.2rem); line-height: 1.2; letter-spacing: -.025em; }
        [data-testid="stMain"] h2 { font-size: 1.45rem; line-height: 1.3; }
        [data-testid="stMain"] h3 { font-size: 1.1rem; line-height: 1.4; }
        /* Color identifies the content level while body copy stays easy to read. */
        [data-testid="stMain"] h1 { color: #174ea6; }
        [data-testid="stMain"] h2 { color: #174ea6; }
        [data-testid="stMain"] h3 { color: #00695c; }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] > p:not(button *, [data-testid="stAlert"] *) {
            color: #364859;
        }
        [data-testid="stMain"] [data-testid="stCaptionContainer"] { opacity: 1; }
        [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
            color: #52677b;
        }
        [data-testid="stMain"] .workspace-section-title { color: #174ea6; }
        [data-testid="stMain"] .workspace-section-note { color: #52677b; }
        [data-testid="stMain"] .workspace-card-title { color: #164a7e; }
        [data-testid="stMain"] .workspace-card-copy { color: #455c70; }
        [data-testid="stMain"] .workspace-accent { color: #00695c; }
        [data-testid="stMain"] .workspace-hero h1 { color: #174ea6; }
        [data-testid="stMain"] .workspace-hero p { color: #315d61; }
        [data-testid="stMain"] .stage-node h3,
        [data-testid="stMain"] .stage-detail h3 { color: #00695c; }
        [data-testid="stMain"] .stage-kind,
        [data-testid="stMain"] .stage-number { color: #174ea6; }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"]:has(input:checked) p {
            color: #174ea6;
        }
        [class*="st-key-home_hero"], [class*="st-key-home_hero"] [data-testid="stVerticalBlockBorderWrapper"] {
            border: 0;
            border-radius: 24px;
            background: linear-gradient(115deg, #e4f1e9, #edf2fa);
            box-shadow: none;
        }
        [class*="st-key-home_hero"] [data-testid="stVerticalBlock"] { gap: 1rem; }
        .workspace-hero h1 {
            max-width: 760px;
            margin: 0;
            color: #163b33;
            font-size: clamp(2.15rem, 5vw, 3.55rem);
            font-weight: 650;
            letter-spacing: -.045em;
            line-height: 1.08;
        }
        .workspace-hero p {
            max-width: 650px;
            margin: .8rem 0 0;
            color: #36574f;
            font-size: 1.08rem;
            line-height: 1.6;
        }
        [class*="st-key-app_navigation"] {
            max-width: 42rem;
            margin-bottom: 1rem;
            padding-bottom: .8rem;
            border-bottom: 1px solid #dce6e0;
        }
        [class*="st-key-app_navigation"] [data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap;
            align-items: center;
            gap: .5rem;
        }
        [class*="st-key-app_navigation"] [data-testid="stColumn"] {
            min-width: 0 !important;
            flex: 1 1 0 !important;
            width: auto !important;
        }
        [class*="st-key-app_navigation"] [data-testid="stColumn"]:last-child {
            flex: 2 1 0 !important;
        }
        [class*="st-key-project_step_"], [class*="st-key-home_card_"],
        [class*="st-key-project_step_"] [data-testid="stVerticalBlockBorderWrapper"],
        [class*="st-key-home_card_"] [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #dce8df;
            background: #fff;
            box-shadow: none;
        }
        .workspace-section-title {
            margin: 2.2rem 0 .25rem;
            color: #29383c;
            font-size: 1.35rem;
            font-weight: 650;
            letter-spacing: -.02em;
        }
        .workspace-section-note { margin: 0 0 1.1rem; color: var(--workspace-muted); }
        .workspace-accent {
            display: inline-block;
            margin: 0 0 .85rem;
            padding: .3rem .65rem;
            border-radius: 999px;
            background: #e8f2eb;
            color: #315d4e;
            font-size: .75rem;
            font-weight: 700;
            letter-spacing: .04em;
        }
        .workspace-card-title {
            min-height: 4.5rem;
            margin: 0 0 .35rem;
            color: #29383c;
            font-size: 1.05rem;
            font-weight: 650;
            line-height: 1.35;
        }
        .workspace-card-copy {
            min-height: 4.2rem;
            margin: 0 0 .9rem;
            color: #637074;
            font-size: .91rem;
            line-height: 1.5;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #e8e6df;
            border-radius: 17px;
            background: rgba(255, 255, 253, .82);
            box-shadow: 0 5px 18px rgba(38, 51, 49, .035);
        }
        .dataset-link {
            display: inline-flex;
            min-height: 2.65rem;
            align-items: center;
            margin: .45rem 0;
            padding: 0 .9rem;
            border: 1px solid #d8ded8;
            border-radius: 10px;
            background: #fff;
            color: #31423e !important;
            font-weight: 620;
            text-decoration: none !important;
        }
        .stButton > button {
            min-height: 2.65rem;
            border: 1px solid #d8ded8;
            border-radius: 10px;
            background: #fff;
            color: #31423e;
            font-weight: 620;
            transition: border-color .15s ease, background .15s ease, transform .15s ease;
        }
        .stButton > button:hover {
            border-color: #8da89b;
            background: #f4f8f4;
            color: #243b32;
            transform: translateY(-1px);
        }
        .stButton > button:focus-visible,
        button:focus-visible,
        a:focus-visible {
            outline: 3px solid #327d68 !important;
            outline-offset: 3px !important;
            box-shadow: none !important;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            border-color: #365f52;
            background: #365f52;
            color: #fff;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            border-color: #294b40;
            background: #294b40;
            color: #fff;
        }
        .stage-flow { margin: 1rem 0; }
        .stage-graph { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 17rem), 1fr)); gap: .8rem; }
        .stage-node, .stage-detail { min-width: 0; overflow-wrap: anywhere;
            padding: 1rem; border: 1px solid #dbe6de;
            border-radius: 14px; background: linear-gradient(120deg, #fffefa, #f5f9f5);
            box-shadow: 0 5px 18px rgba(38, 51, 49, .04); }
        .stage-node h3, .stage-detail h3 { margin: .3rem 0; color: #293d36; }
        .stage-node p, .stage-detail p { margin: .35rem 0; color: #52635e; }
        .stage-kind, .stage-number { color: #567e6e; font-size: .75rem;
            font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
        .stage-inputs { list-style: none; padding: 0; margin: .5rem 0 0; }
        .stage-inputs li { margin: .25rem 0; font-size: .9rem; }
        .stage-arrow { color: #567e6e; font-weight: 700; padding: 0 .3rem; }
        .stage-details { display: grid; gap: .8rem; margin-top: .8rem; }
        .stage-tools { margin-top: .7rem; padding-top: .6rem;
            border-top: 1px solid #e1ebe3; font-size: .9rem; }
        @media (max-width: 760px) {
            .stage-node, .stage-detail { padding: .85rem; }
            [data-testid="stMainBlockContainer"] { padding: 4rem 1rem 2.5rem; }
            .workspace-card-title, .workspace-card-copy { min-height: 0; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

_WORKFLOW_PATH_NOTES = {
    "cycif-raw-czi": [
        "Conversion is optional; its OME-TIFF outputs are kept separately.",
        "Registration uses the original CZI files, not conversion output, and requires confirmed grouping.",
        "Marker export requires confirmed mapping; optional segmentation uses the registered export.",
    ],
    "cycif.tabular_analysis": [
        "Steps 06–12 run only when selected.",
        "Steps 09–11 independently use the current cell table; step 11 needs the matching step 10 settings, not its output.",
        "Step 12 compares the biological-unit aggregate from step 09.",
    ],
    "hs.clinical-roi-hyperspectral": [
        "Incomplete or unconfirmed specimen mapping stops analysis after the metadata review.",
        "TMA1 and TMA2 phasors are computed separately.",
        "Filtering and Gaussian mixture model (GMM) diagnostics are optional alternatives, not sequential steps.",
    ],
    "hs.native_spectral_qc": [
        "One read produces the inventory, band review, and spectral moments.",
        "Intensity and H1/H2 plots use those moments; this is technical quality control (QC) only.",
    ],
    "flim.raw_sdt": [
        "Unless raw QC for all blocks is explicitly authorized, the inventory pauses at QC_PENDING for block review.",
        "Raw-QC mode stops after counts and decay checks; full mode adds phasors, a count mask, optional filtering, and lifetimes.",
        "A measured reference follows a separate conditional path; its phasors join target phasors only for calibration.",
        "Lifetime output can also come directly from masked or filtered target phasors when calibration is not enabled.",
    ],
    "he-image-intake": [
        "Each image uses one format-specific reader: OpenSlide for SVS, aicspylibczi for CZI, or tifffile for TIFF.",
        "Inventory and available size-limited previews are collected after reading; images are not registered.",
    ],
    "visium-native-qc": [
        "Classic Visium and Visium HD use alternative readers.",
        "HD requires an explicit bin size before count QC.",
        "Previews are conditional; no cell segmentation is performed.",
    ],
    "integration.serial_section_alignment": [
        "The selected H&E and fluorescence/HS CZI images need source and channel confirmation.",
        "Metadata and scale checks and size-limited previews precede translation-only registration.",
        "The resulting metrics require human review and do not validate an alignment.",
    ],
}


# Each card lists its actual inputs rather than implying that adjacent cards are connected.
# Entries are (label, descriptor stage ID, branch kind, [(source label, condition)]).
_WORKFLOW_GRAPHS = {
    "cycif-raw-czi": [
        ("Intake", "intake", "Input", []),
        ("Conversion", "conversion", "Optional", [("Intake", "original CZI")]),
        ("Registration", "registration", "Gate", [("Intake", "original CZI; confirmed groups")]),
        ("Marker export", "export", "Gate", [("Registration", "confirmed panel")]),
        ("Segmentation", "segmentation", "Optional", [("Marker export", "exported composites")]),
    ],
    "cycif.tabular_analysis": [
        ("Selected cell table", None, "Input", []),
        ("Intensity QC", "step06", "Optional", [("Selected cell table", "step 06")]),
        ("Cell classification", "step07", "Optional", [("Intensity QC", "if enabled"), ("Selected cell table", "if step 06 is disabled")]),
        ("Program scoring", "step08", "Optional", [("Cell classification", "if enabled"), ("Intensity QC", "if enabled; otherwise selected table"), ("Selected cell table", "if earlier steps are disabled")]),
        ("Current cell frame", None, "Join", [("Program scoring", "if enabled"), ("Cell classification", "if scoring disabled"), ("Intensity QC", "if later steps disabled"), ("Selected cell table", "if steps 06–08 disabled")]),
        ("Aggregation", "step09", "Optional", [("Current cell frame", "step 09")]),
        ("Spatial features", "step10", "Optional", [("Current cell frame", "step 10")]),
        ("Spatial null", "step11", "Optional", [("Current cell frame", "step 11; same step 10 configuration, not its result")]),
        ("Group comparison", "step12", "Optional", [("Aggregation", "step 09 aggregate")]),
    ],
    "hs.clinical-roi-hyperspectral": [
        ("Metadata audit", "metadata_audit", "Input", []),
        ("Specimen mapping", None, "Gate", [("Metadata audit", "unconfirmed or incomplete: stop")]),
        ("TMA1 phasors", "source_phasors", "Branch", [("Specimen mapping", "complete map")]),
        ("TMA2 phasors", "source_phasors", "Branch", [("Specimen mapping", "complete map")]),
        ("TMA1 filter", "exploratory", "Optional alternative", [("TMA1 phasors", "filter OR GMM")]),
        ("TMA1 GMM", "exploratory", "Optional alternative", [("TMA1 phasors", "GMM OR filter")]),
        ("TMA2 filter", "exploratory", "Optional alternative", [("TMA2 phasors", "filter OR GMM")]),
        ("TMA2 GMM", "exploratory", "Optional alternative", [("TMA2 phasors", "GMM OR filter")]),
    ],
    "hs.native_spectral_qc": [
        ("Native reader call", None, "Input", []),
        ("Inventory and band audit", "hs_native_inventory", "Parallel output", [("Native reader call", "same reader call")]),
        ("Moment maps", "hs_native_moments", "Parallel output", [("Native reader call", "same reader call")]),
        ("Intensity plots", "hs_native_qc_plots", "Branch", [("Moment maps", "in-memory moments")]),
        ("H1/H2 plots", "hs_native_qc_plots", "Branch", [("Moment maps", "in-memory moments")]),
    ],
    "flim.raw_sdt": [
        ("Target inventory", "flim_inventory", "Input", []),
        ("Block review", None, "Gate", [("Target inventory", "review selection; or authorize all-block raw QC")]),
        ("Raw decay QC", "flim_raw_qc", "Branch", [("Block review", "selected blocks or authorized all-block raw QC")]),
        ("Raw QC stop", None, "Stop", [("Raw decay QC", "raw_qc mode")]),
        ("Target phasors", "flim_phasor", "Branch", [("Block review", "full mode; reviewed target inventory")]),
        ("Count mask", "flim_count_mask", "Stage", [("Target phasors", "full mode")]),
        ("Phasor filter", "flim_filter", "Optional", [("Count mask", "if enabled")]),
        ("Measured reference input", None, "Optional input", []),
        ("Reference inventory", "flim_reference_raw_qc", "Optional", [("Measured reference input", "measured SDT")]),
        ("Reference QC", "flim_reference_raw_qc", "Branch", [("Reference inventory", "reference blocks")]),
        ("Reference phasors", "flim_reference_phasor", "Branch", [("Reference inventory", "reference blocks")]),
        ("Reference calibration", "flim_reference_calibration", "Optional join", [("Reference phasors", "measured reference"), ("Count mask", "target, if filter disabled"), ("Phasor filter", "target, if filter enabled")]),
        ("Lifetimes", "flim_lifetimes", "Stage", [("Count mask", "uncalibrated; filter disabled"), ("Phasor filter", "uncalibrated; filter enabled"), ("Reference calibration", "if calibration enabled")]),
    ],
    "he-image-intake": [
        ("File selection", None, "Input", []),
        ("SVS reader", "image-intake", "Alternative", [("File selection", "SVS; OpenSlide if available")]),
        ("CZI reader", "image-intake", "Alternative", [("File selection", "CZI; aicspylibczi if available")]),
        ("TIFF reader", "image-intake", "Alternative", [("File selection", "TIFF; tifffile if available")]),
        ("Inventory and bounded preview", "image-intake", "Join", [("SVS reader", "preview where supported"), ("CZI reader", "preview where supported"), ("TIFF reader", "preview where supported")]),
    ],
    "visium-native-qc": [
        ("Bundle intake", "bundle-intake", "Input", []),
        ("Classic Visium reader", "bundle-intake", "Alternative", [("Bundle intake", "classic Visium; reader availability required")]),
        ("HD native bin", "bundle-intake", "Gate", [("Bundle intake", "Visium HD; explicit native bin")]),
        ("HD reader", "bundle-intake", "Alternative", [("HD native bin", "reader availability required")]),
        ("Native count QC", "count-qc", "Join", [("Classic Visium reader", "classic counts"), ("HD reader", "selected native bin counts")]),
        ("Bounded preview", "count-qc", "Optional", [("Native count QC", "where supported")]),
    ],
    "integration.serial_section_alignment": [
        ("Moving H&E CZI", None, "Input", []),
        ("Reference fluorescence/HS CZI", None, "Input", []),
        ("Source and channel authorization", "serial-section-alignment", "Gate", [("Moving H&E CZI", "explicit input"), ("Reference fluorescence/HS CZI", "explicit input and channel")]),
        ("Metadata and physical scale", "serial-section-alignment", "Stage", [("Source and channel authorization", "preflight passed")]),
        ("Bounded previews", "serial-section-alignment", "Stage", [("Metadata and physical scale", "validated metadata")]),
        ("Translation-only registration", "serial-section-alignment", "Stage", [("Bounded previews", "bounded images")]),
        ("Metrics and human review", "serial-section-alignment", "Gate", [("Translation-only registration", "review required; not validated")]),
    ],
}


def render_stage_flow(descriptor: WorkflowDescriptor) -> None:
    """Render explicit input relationships and the selected descriptor's stage details."""
    st.write(
        "Inputs and settings determine which paths run; optional and alternative paths do not all run. "
        "Arrows show how the steps connect. Tool versions describe the current workflow and may differ from a saved run."
    )
    notes = _WORKFLOW_PATH_NOTES.get(descriptor.workflow_id)
    if notes:
        st.subheader("Which paths run")
        st.markdown("\n".join(f"- {note}" for note in notes))
    if not descriptor.stages:
        st.caption("No analysis stages are declared for this workflow.")
        return
    stages = {stage.stage_id: stage for stage in descriptor.stages}
    nodes = []
    display_labels = {
        "Native reader call": "Read original image",
        "Inventory and bounded preview": "Inventory and size-limited preview",
        "HD native bin": "HD bin size",
        "Native count QC": "Original count QC",
        "Bounded preview": "Size-limited preview",
        "Bounded previews": "Size-limited previews",
        "Current cell frame": "Current cell table",
        "Spatial null": "Spatial null model",
        "Metadata audit": "Metadata review",
        "Inventory and band audit": "Inventory and band review",
        "Moment maps": "Spectral moment maps",
        "Raw QC stop": "Stop after raw QC",
        "Bundle intake": "Input bundle",
        "Moving H&E CZI": "H&E CZI to align",
        "Source and channel authorization": "Confirm sources and channel",
        "Metadata and physical scale": "Check metadata and physical scale",
        "Metrics and human review": "Alignment metrics and human review",
    }
    display_kinds = {
        "Gate": "Review required",
        "Join": "Path convergence",
        "Optional join": "Optional path convergence",
        "Parallel output": "Parallel result",
        "Branch": "Path",
        "Stage": "Step",
        "Stop": "Stopping point",
    }
    display_conditions = {
        "complete map": "mapping complete",
        "filter OR GMM": "filtering or GMM",
        "GMM OR filter": "GMM or filtering",
        "in-memory moments": "moments held in memory",
        "raw_qc mode": "raw-QC mode",
        "preflight passed": "initial checks passed",
        "explicit input": "selected input",
        "explicit input and channel": "selected input and channel",
    }
    for label, stage_id, kind, inputs in _WORKFLOW_GRAPHS.get(descriptor.workflow_id, []):
        stage = stages.get(stage_id)
        display_label = display_labels.get(label, label)
        display_kind = "Selection required" if label == "HD native bin" else display_kinds.get(kind, kind)
        reference = f"<p>Analysis step: {escape(stage.title)}</p>" if stage else ""
        edges = "".join(
            f'<li data-source="{escape(source, quote=True)}" data-target="{escape(label, quote=True)}">'
            f'{escape(display_labels.get(source, source))} <span class="stage-arrow" aria-hidden="true">→</span> '
            f'{escape(display_label)} <small>({escape(display_conditions.get(condition, condition).replace("native bin", "bin size").replace("bounded images", "size-limited images"))})</small></li>'
            for source, condition in inputs
        )
        nodes.append(
            f'<article class="stage-node" data-node="{escape(label, quote=True)}">'
            f'<span class="stage-kind">{escape(display_kind)}</span><h3>{escape(display_label)}</h3>{reference}'
            + (f'<ul class="stage-inputs" aria-label="Inputs to {escape(display_label, quote=True)}">{edges}</ul>' if edges else "")
            + "</article>"
        )
    details = []
    for index, stage in enumerate(descriptor.stages, 1):
        tools = "<br>".join(
            f"<strong>{escape(tool.name)}</strong> · "
            f"{escape('runtime version recorded in the run record' if tool.name == 'MCMICRO' else tool.version)}"
            f" — {escape(tool.role)}"
            for tool in stage.tools
        ) if stage.tools else "No tools declared for this stage."
        details.append(
            f'<article class="stage-detail"><span class="stage-number">Stage {index}</span>'
            f"<h3>{escape(stage.title)}</h3><p>{escape(stage.purpose)}</p>"
            f'<div class="stage-tools"><strong>Tools for this step</strong><br>{tools}</div></article>'
        )
    st.markdown(
        '<section class="stage-flow" aria-label="Workflow paths and declared stages">'
        + ('<div class="stage-graph" aria-label="Workflow path inputs">' + "".join(nodes) + "</div>" if nodes else "")
        + '<div class="stage-details">' + "".join(details) + "</div></section>",
        unsafe_allow_html=True,
    )


def render_home(
    navigate: Callable[[str], None], *, read_only: bool = False, dataset_url: str | None = None,
) -> None:
    """Render the welcoming home screen and its direct workspace actions."""
    intro = (
        "Explore workflow introductions and tools. Analyses run on authorized HPC resources."
        if read_only else
        "Explore an example or set up your study to work with your own spatial data."
    )
    with st.container(border=True, key="home_hero"):
        st.markdown(
            '<section class="workspace-hero"><h1>Spatial Toolkit</h1>'
            f"<p>{escape(intro)}</p></section>",
            unsafe_allow_html=True,
        )
        st.button("Open project", type="primary", on_click=navigate, args=("Project",), key="home_open_project")
        if dataset_url:
            parsed_url = urlsplit(dataset_url)
            if parsed_url.scheme == "https" and parsed_url.hostname in {"drive.google.com", "docs.google.com"}:
                st.markdown(
                    f'<a class="dataset-link" href="{escape(dataset_url, quote=True)}" '
                    'target="_blank" rel="noopener noreferrer">Open shared datasets in Google Drive ↗</a>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("The configured dataset URL is not a secure Google Drive link.")

    st.markdown('<h2 class="workspace-section-title">Explore workflows</h2>', unsafe_allow_html=True)
    st.markdown('<p class="workspace-section-note">Choose a workspace to see its workflows and tools.</p>', unsafe_allow_html=True)
    for row_start in range(0, len(_MODALITY_CARDS), 3):
        columns = st.columns(3, gap="medium")
        for column, (code, title, description) in zip(
            columns, _MODALITY_CARDS[row_start : row_start + 3]
        ):
            with column:
                with st.container(border=True, key=f"home_card_{code}"):
                    badge = "OPTIONAL" if code == "Integration" else code
                    st.markdown(
                        f'<span class="workspace-accent">{escape(badge)}</span>'
                        f'<h3 class="workspace-card-title">{escape(title)}</h3>'
                        f'<p class="workspace-card-copy">{escape(description)}</p>',
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "Explore integration" if code == "Integration" else f"Open {code}",
                        on_click=navigate,
                        args=(code,),
                        key="home_integration" if code == "Integration" else f"home_modality_{code}",
                        help=f"Open {title}",
                        use_container_width=True,
                    )
