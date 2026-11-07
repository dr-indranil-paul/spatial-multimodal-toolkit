from __future__ import annotations

from collections.abc import Callable
from html import escape
from urllib.parse import urlsplit

import streamlit as st

from spatial_omics_toolkit.application.contracts import WorkflowDescriptor


MODALITY_LABELS: dict[str, str] = {
    "CycIF": "Cyclic immunofluorescence (CycIF)",
    "HS": "Hyperspectral imaging (HS)",
    "FLIM": "Fluorescence lifetime imaging microscopy (FLIM)",
    "H&E": "Hematoxylin and eosin (H&E)",
    "Visium": "Visium spatial transcriptomics",
    "Integration": "Integration",
}

_MODALITY_FORMATS: dict[str, str] = {
    "CycIF": ".czi rounds · cell table",
    "HS": ".czi spectral · study manifests",
    "FLIM": ".sdt TCSPC blocks",
    "H&E": ".svs · .czi · .tiff",
    "Visium": "Space Ranger outs",
    "Integration": "H&E + fluorescence .czi",
}

_MODALITY_CARDS: tuple[tuple[str, str, str], ...] = (
    ("CycIF", MODALITY_LABELS["CycIF"], "Multiplexed fluorescence intake, multi-round registration, marker export, MCMICRO segmentation, and cell-table analysis."),
    ("HS", MODALITY_LABELS["HS"], "Spectral band inspection, moment maps, phasor transforms, and clinical TMA spectral workflows."),
    ("FLIM", MODALITY_LABELS["FLIM"], "Native TCSPC block inventory, photon-count and decay QC, phasor masking, and lifetime estimation."),
    ("H&E", MODALITY_LABELS["H&E"], "Whole-slide and core histology metadata verification and size-limited slide previews."),
    ("Visium", MODALITY_LABELS["Visium"], "Classic Visium and Visium HD spatial transcriptomics count QC and spatial coordinate previews."),
    ("Integration", MODALITY_LABELS["Integration"], "Exploratory serial-section translation preview comparing H&E and fluorescence reference channels."),
)


def apply_theme() -> None:
    """Apply the Slate & Indigo scientific workbench theme."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            --wb-canvas: #f8fafc;
            --wb-surface: #ffffff;
            --wb-surface-subtle: #f1f5f9;
            --wb-ink-900: #0f172a;
            --wb-ink-800: #1e293b;
            --wb-ink-600: #475569;
            --wb-ink-500: #64748b;
            --wb-border: #e2e8f0;
            --wb-border-strong: #cbd5e1;
            --wb-accent: #4f46e5;
            --wb-accent-hover: #4338ca;
            --wb-accent-soft: #eef2ff;
            --wb-accent-border: #c7d2fe;
            --wb-font-sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            --wb-font-mono: "JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--wb-canvas);
            color: var(--wb-ink-800);
            font-family: var(--wb-font-sans);
        }
        [data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0.92);
            backdrop-filter: blur(8px);
        }
        [data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid #1e293b;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 0.25rem; }
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0.4rem; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #cbd5e1;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1480px;
            padding-top: 4.5rem;
            padding-bottom: 3.5rem;
        }
        [data-testid="stMain"] [data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
            margin-bottom: 0 !important;
        }
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3 {
            color: var(--wb-ink-900);
            letter-spacing: -0.02em;
            font-family: var(--wb-font-sans);
        }
        [data-testid="stMain"] h1,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] h1 {
            font-size: clamp(1.35rem, 1.8vw, 1.65rem) !important;
            font-weight: 750 !important;
            line-height: 1.25 !important;
            padding: 0 !important;
            margin: 0 0 0.1rem !important;
        }
        [data-testid="stMain"] h2,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] h2 {
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            line-height: 1.3 !important;
            padding: 0 !important;
            margin: 0 0 0.1rem !important;
        }
        [data-testid="stMain"] h3,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] h3 {
            font-size: 0.92rem !important;
            font-weight: 700 !important;
            line-height: 1.35 !important;
            padding: 0 !important;
            margin: 0 0 0.1rem !important;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] p {
            margin: 0 0 0.35rem !important;
            font-size: 0.84rem;
            line-height: 1.45;
            color: var(--wb-ink-800);
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] p:last-child {
            margin-bottom: 0 !important;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] ul,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] ol {
            margin: 0.15rem 0 0.35rem !important;
            padding-left: 1.15rem !important;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] ul:last-child,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] ol:last-child {
            margin-bottom: 0 !important;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] li {
            margin: 0.1rem 0 !important;
            padding: 0 !important;
            font-size: 0.84rem;
            line-height: 1.42;
            color: var(--wb-ink-800);
        }
        [data-testid="stMain"] [data-testid="stWidgetLabel"] p {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            line-height: 1.35 !important;
            color: var(--wb-ink-800) !important;
        }
        [data-testid="stMain"] [data-baseweb="input"] input,
        [data-testid="stMain"] [data-baseweb="select"] {
            font-size: 0.82rem !important;
        }
        [data-testid="stMain"] [data-testid="stCaptionContainer"],
        [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
            opacity: 1;
            color: var(--wb-ink-500) !important;
            font-size: 0.78rem !important;
            line-height: 1.45 !important;
        }
        [data-testid="stMain"] [data-testid="stAlert"] p {
            font-size: 0.82rem !important;
            line-height: 1.45 !important;
        }

        /* Sidebar Brand & Navigation */
        .workspace-brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.25rem 0.35rem 0.65rem;
            margin-bottom: 0.25rem;
            border-bottom: 1px solid #1e293b;
            color: #f8fafc;
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .brand-mark {
            display: grid;
            place-items: center;
            width: 2rem;
            height: 2rem;
            flex: 0 0 auto;
            border-radius: 0.5rem;
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: #ffffff !important;
            font-size: 0.95rem;
            font-weight: 800;
            box-shadow: 0 2px 6px rgba(79, 70, 229, 0.35);
        }
        .workspace-brand small {
            display: block;
            margin-top: 0.18rem;
            color: #94a3b8 !important;
            font-size: 0.6rem;
            font-weight: 600;
            letter-spacing: 0.12em;
        }
        .nav-label {
            margin: 0.65rem 0.35rem 0.2rem;
            padding: 0.15rem 0.25rem;
            color: #64748b !important;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        [data-testid="stMarkdownContainer"]:has(> .nav-label) { margin-bottom: 0; }
        [data-testid="stVerticalBlock"][class*="st-key-app_navigation"],
        [class*="st-key-app_navigation"] [data-testid="stVerticalBlock"] { gap: 0.18rem; }
        [class*="st-key-app_navigation"] .stButton > button {
            min-height: 2.15rem;
            height: auto !important;
            justify-content: flex-start;
            text-align: left !important;
            white-space: normal !important;
            padding: 0.38rem 0.65rem !important;
            border: 1px solid transparent;
            border-radius: 0.45rem;
            background: transparent;
            color: #cbd5e1 !important;
            font-size: 0.8rem !important;
            font-weight: 500;
            line-height: 1.32 !important;
            box-shadow: none;
        }
        [class*="st-key-app_navigation"] .stButton > button > div,
        [class*="st-key-app_navigation"] .stButton > button [data-testid="stMarkdownContainer"],
        [class*="st-key-app_navigation"] .stButton > button p {
            justify-content: flex-start;
            text-align: left !important;
            white-space: normal !important;
            font-size: 0.8rem !important;
            line-height: 1.32 !important;
            margin: 0 !important;
        }
        [class*="st-key-app_navigation"] .stButton > button:hover {
            background: rgba(255, 255, 255, 0.06);
            color: #ffffff !important;
            border-color: rgba(255, 255, 255, 0.08);
        }
        [class*="st-key-app_navigation"] div[data-testid="stButton"] button[kind="primary"] {
            border-color: #6366f1;
            background: #4f46e5;
            color: #ffffff !important;
            font-weight: 600;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        }
        .sidebar-study-card {
            margin-top: 0.9rem;
            padding: 0.65rem 0.75rem;
            border-radius: 0.5rem;
            background: #1e293b;
            border: 1px solid #334155;
            font-size: 0.76rem;
            line-height: 1.45;
            color: #cbd5e1;
        }
        .sidebar-study-card strong {
            display: block;
            color: #f8fafc;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }
        .sidebar-status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.12rem 0;
            color: #cbd5e1;
        }
        .sidebar-pill-ok {
            color: #4ade80 !important;
            font-weight: 600;
            font-family: var(--wb-font-mono);
            font-size: 0.7rem;
        }
        .sidebar-pill-pending {
            color: #fbbf24 !important;
            font-weight: 600;
            font-family: var(--wb-font-mono);
            font-size: 0.7rem;
        }

        /* Main Horizontal Section Bar */
        [class*="st-key-workspace_section_"] {
            width: 100%;
            margin: 0.15rem 0 0.75rem;
            padding: 0.25rem;
            background: #f1f5f9;
            border: 1px solid var(--wb-border);
            border-radius: 0.6rem;
        }
        [class*="st-key-workspace_section_"] > label {
            display: none !important;
        }
        [class*="st-key-workspace_section_"] [role="radiogroup"] {
            width: 100%;
            display: flex;
            flex-wrap: wrap;
            gap: 0.25rem;
        }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"] {
            margin: 0;
            padding: 0.4rem 0.85rem;
            border: 1px solid transparent;
            border-radius: 0.45rem;
            background: transparent;
            color: var(--wb-ink-600);
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.12s ease;
        }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"]:hover {
            color: var(--wb-ink-900);
            background: rgba(255, 255, 255, 0.6);
        }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"] > div:first-child {
            position: absolute;
            opacity: 0;
            width: 0;
            height: 0;
        }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"] > div:last-child {
            margin-left: 0;
        }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"]:has(input:checked) {
            background: #ffffff;
            border-color: var(--wb-border-strong);
            color: var(--wb-accent);
            font-weight: 700;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }
        [class*="st-key-workspace_section_"] label[data-baseweb="radio"]:focus-within {
            outline: 2px solid var(--wb-accent);
            outline-offset: 1px;
        }

        /* Input Mode Segmented Toggle */
        [class*="st-key-data_mode_"] [role="radiogroup"] {
            gap: 0.35rem;
            flex-wrap: wrap;
        }
        [class*="st-key-data_mode_"] label[data-baseweb="radio"] {
            padding: 0.38rem 0.7rem;
            border: 1px solid var(--wb-border);
            border-radius: 0.45rem;
            background: #ffffff;
            font-size: 0.84rem;
            font-weight: 500;
        }
        [class*="st-key-data_mode_"] label[data-baseweb="radio"]:has(input:checked) {
            border-color: var(--wb-accent);
            background: var(--wb-accent-soft);
            color: #312e81;
            font-weight: 600;
        }

        /* Containers & Surfaces */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--wb-border);
            border-radius: 0.75rem;
            background: var(--wb-surface);
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03), 0 1px 2px rgba(15, 23, 42, 0.02);
        }
        [data-testid="stVerticalBlock"][class*="st-key-project_step_"],
        [data-testid="stVerticalBlock"][class*="st-key-own_"],
        [data-testid="stVerticalBlock"][class*="st-key-example_card_"],
        [data-testid="stVerticalBlock"][class*="st-key-intro_"] {
            border: 1px solid var(--wb-border) !important;
            border-radius: 0.75rem !important;
            background: #ffffff;
            padding: 1.1rem 1.25rem !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        }
        [data-testid="stExpander"] {
            border-color: var(--wb-border);
            border-radius: 0.6rem;
            background: var(--wb-surface);
        }

        /* Home Workbench Header & Cards */
        [data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-home_hero"]),
        [data-testid="stVerticalBlock"][class*="st-key-home_hero"] {
            border: 1px solid #1e293b !important;
            border-radius: 1rem !important;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 65%, #1e1b4b 100%) !important;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.14);
            padding: 1.85rem 2.25rem !important;
            margin-top: 0.25rem;
            margin-bottom: 0.5rem;
            overflow: hidden;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-home_hero"]) [data-testid="stVerticalBlock"][class*="st-key-home_hero"] {
            border: 0 !important;
            background: transparent !important;
            box-shadow: none;
            padding: 0 !important;
            margin: 0;
        }
        [class*="st-key-home_hero"] [data-testid="stMarkdownContainer"] {
            margin-bottom: 0 !important;
        }
        .workspace-hero {
            padding: 0.2rem 0;
            text-align: left;
        }
        .workspace-hero-eyebrow {
            display: inline-block;
            padding: 0.22rem 0.65rem;
            margin-bottom: 0.65rem;
            border-radius: 999px;
            background: rgba(99, 102, 241, 0.18);
            border: 1px solid rgba(129, 140, 248, 0.35);
            color: #c7d2fe;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        [data-testid="stMain"] .workspace-hero h1,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] .workspace-hero h1 {
            margin: 0 !important;
            padding: 0 !important;
            color: #f8fafc !important;
            font-size: clamp(1.55rem, 2.2vw, 2rem) !important;
            font-weight: 750 !important;
            letter-spacing: -0.025em;
            line-height: 1.2 !important;
        }
        [data-testid="stMain"] .workspace-hero p,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] .workspace-hero p {
            max-width: 680px;
            margin: 0.55rem 0 0 !important;
            padding: 0 !important;
            color: #cbd5e1 !important;
            font-size: 0.92rem !important;
            line-height: 1.55 !important;
        }
        .section-eyebrow {
            margin: 1rem 0 0.2rem;
            color: var(--wb-accent);
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        [data-testid="stMain"] .workspace-section-title {
            margin: 0 0 0.15rem;
            padding: 0;
            font-weight: 700;
            color: var(--wb-ink-900);
        }
        .workspace-section-note {
            margin: 0 0 0.75rem;
            color: var(--wb-ink-600);
            font-size: 0.88rem;
        }
        [data-testid="stVerticalBlock"][class*="st-key-home_card_"] {
            border: 1px solid var(--wb-border);
            border-top: 3px solid var(--wb-accent);
            border-radius: 0.75rem;
            background: #ffffff;
            padding: 1.15rem 1.25rem;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        [data-testid="stVerticalBlock"][class*="st-key-home_card_"]:hover {
            border-color: var(--wb-border-strong);
            border-top-color: var(--wb-accent-hover);
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
        }
        .workspace-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.45rem;
        }
        .workspace-accent {
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 0.35rem;
            background: var(--wb-accent-soft);
            color: var(--wb-accent);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .workspace-format-tag {
            font-family: var(--wb-font-mono);
            font-size: 0.7rem;
            color: var(--wb-ink-500);
            background: var(--wb-surface-subtle);
            padding: 0.12rem 0.45rem;
            border-radius: 0.3rem;
        }
        [data-testid="stMain"] .workspace-card-title {
            min-height: 2.3rem;
            margin: 0 0 0.3rem;
            padding: 0;
            font-size: 0.96rem;
            font-weight: 700;
            line-height: 1.3;
            color: var(--wb-ink-900);
        }
        .workspace-card-copy {
            min-height: 3.6rem;
            margin: 0 0 0.65rem;
            color: var(--wb-ink-600);
            font-size: 0.84rem;
            line-height: 1.48;
        }
        .dataset-link {
            display: inline-flex;
            min-height: 2.1rem;
            align-items: center;
            margin: 0.15rem 0;
            color: #c7d2fe !important;
            font-size: 0.85rem;
            font-weight: 600;
            text-decoration: underline;
            text-underline-offset: 3px;
        }

        /* Modality Compact Summary Strip */
        .wb-modality-banner {
            padding: 0.65rem 0.9rem;
            margin-bottom: 0.5rem;
            background: #ffffff;
            border: 1px solid var(--wb-border);
            border-left: 3px solid var(--wb-accent);
            border-radius: 0.55rem;
        }
        .wb-modality-banner-top {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-bottom: 0.2rem;
        }
        .wb-modality-pill {
            display: inline-block;
            padding: 0.1rem 0.45rem;
            border-radius: 0.3rem;
            background: var(--wb-accent-soft);
            color: var(--wb-accent);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .wb-modality-workflow {
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--wb-ink-900);
        }
        .wb-modality-desc {
            margin: 0;
            color: var(--wb-ink-600);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        /* Pane Headers & Status Strips */
        .wb-pane-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 0.45rem;
            margin-bottom: 0.65rem;
            border-bottom: 1px solid var(--wb-border);
        }
        .wb-pane-title {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--wb-ink-600);
            margin: 0;
        }
        .wb-readiness-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.35rem 0 0.75rem;
        }
        .wb-check-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            border: 1px solid var(--wb-border);
            background: var(--wb-surface-subtle);
            color: var(--wb-ink-600);
        }
        .wb-check-pill.ready {
            background: #f0fdf4;
            border-color: #bbf7d0;
            color: #166534;
        }
        .wb-check-pill.pending {
            background: #fffbeb;
            border-color: #fde68a;
            color: #92400e;
        }

        /* Buttons */
        .stButton > button, [data-testid="stDownloadButton"] > button {
            min-height: 2.2rem !important;
            height: auto !important;
            padding: 0.38rem 0.75rem !important;
            border: 1px solid var(--wb-border-strong);
            border-radius: 0.5rem;
            background: #ffffff;
            color: var(--wb-ink-800);
            font-size: 0.8125rem !important;
            font-weight: 600 !important;
            line-height: 1.3 !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            transition: all 0.12s ease;
        }
        .stButton > button p,
        [data-testid="stDownloadButton"] > button p {
            font-size: 0.8125rem !important;
            font-weight: 600 !important;
            line-height: 1.3 !important;
            margin: 0 !important;
            white-space: normal !important;
            color: inherit !important;
        }
        .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
            border-color: #94a3b8;
            background: #f8fafc;
            color: var(--wb-ink-900);
        }
        .stButton > button:focus-visible, button:focus-visible, a:focus-visible {
            outline: 2px solid var(--wb-accent) !important;
            outline-offset: 2px !important;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            border-color: var(--wb-accent-hover);
            background: var(--wb-accent);
            color: #ffffff;
            box-shadow: 0 1px 2px rgba(79, 70, 229, 0.25);
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            border-color: #3730a3;
            background: var(--wb-accent-hover);
            color: #ffffff;
        }

        /* Stage Flow Diagram */
        .stage-flow { margin: 0.65rem 0; }
        .stage-graph {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 15.5rem), 1fr));
            gap: 0.6rem;
        }
        .stage-node, .stage-detail {
            min-width: 0;
            overflow-wrap: anywhere;
            padding: 0.75rem 0.85rem;
            border: 1px solid var(--wb-border);
            border-radius: 0.55rem;
            background: #ffffff;
        }
        [data-testid="stMain"] .stage-node h3,
        [data-testid="stMain"] .stage-detail h3 {
            margin: 0.25rem 0;
            padding: 0;
            color: var(--wb-ink-900);
            font-size: 0.92rem;
        }
        .stage-node p, .stage-detail p {
            margin: 0.3rem 0;
            color: var(--wb-ink-600);
            font-size: 0.84rem;
        }
        .stage-kind, .stage-number {
            display: inline-block;
            padding: 0.1rem 0.4rem;
            border-radius: 0.25rem;
            background: var(--wb-accent-soft);
            color: var(--wb-accent);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .stage-inputs { list-style: none; padding: 0; margin: 0.45rem 0 0; }
        .stage-inputs li { margin: 0.2rem 0; font-size: 0.82rem; color: var(--wb-ink-600); }
        .stage-arrow { color: var(--wb-accent); font-weight: 700; padding: 0 0.25rem; }
        .stage-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 20rem), 1fr));
            gap: 0.6rem;
            margin-top: 0.6rem;
        }
        .stage-tools {
            margin-top: 0.55rem;
            padding-top: 0.5rem;
            border-top: 1px solid var(--wb-border);
            font-size: 0.82rem;
            color: var(--wb-ink-600);
        }
        @media (max-width: 760px) {
            [data-testid="stMainBlockContainer"] { padding: 4.5rem 0.85rem 2.25rem; }
            .workspace-card-title, .workspace-card-copy { min-height: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_modality_banner(modality: str, descriptor: WorkflowDescriptor) -> None:
    """Render a compact summary strip at the top of a modality workbench."""
    format_tag = _MODALITY_FORMATS.get(modality, modality)
    st.markdown(
        f'<div class="wb-modality-banner">'
        f'<div class="wb-modality-banner-top">'
        f'<span class="wb-modality-pill">{escape(modality)}</span>'
        f'<span class="wb-modality-workflow">{escape(descriptor.title)}</span>'
        f'<span class="workspace-format-tag">{escape(format_tag)}</span>'
        f"</div>"
        f'<p class="wb-modality-desc">{escape(descriptor.description)}</p>'
        f"</div>",
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
    """Render the Slate & Indigo home workbench and modality cards."""
    intro = (
        "Inspect spatial omics workflow architectures and declared tools. Execution runs on authorized HPC resources."
        if read_only else
        "Launch a file-backed technical example or configure your study sample sheet to run spatial imaging and omics workflows."
    )
    with st.container(border=True, key="home_hero"):
        hero_left, hero_right = st.columns([3, 1], vertical_alignment="center")
        with hero_left:
            st.markdown(
                '<section class="workspace-hero">'
                '<span class="workspace-hero-eyebrow">Spatial Omics Workbench</span>'
                "<h1>Spatial Toolkit</h1>"
                f"<p>{escape(intro)}</p></section>",
                unsafe_allow_html=True,
            )
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
        with hero_right:
            st.button(
                "Open project",
                type="primary",
                on_click=navigate,
                args=("Project",),
                key="home_open_project",
                use_container_width=True,
            )

    st.markdown('<p class="section-eyebrow">Modality Workbenches</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="workspace-section-title">Explore workflows</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="workspace-section-note">Select a modality workbench to configure inputs, run quality control, and inspect outputs.</p>',
        unsafe_allow_html=True,
    )
    for row_start in range(0, len(_MODALITY_CARDS), 3):
        columns = st.columns(3, gap="medium")
        for column, (code, title, description) in zip(
            columns, _MODALITY_CARDS[row_start : row_start + 3]
        ):
            with column:
                with st.container(border=True, key=f"home_card_{code}"):
                    format_tag = _MODALITY_FORMATS.get(code, code)
                    st.markdown(
                        f'<div class="workspace-card-header">'
                        f'<span class="workspace-accent">{escape(code)}</span>'
                        f'<span class="workspace-format-tag">{escape(format_tag)}</span>'
                        f"</div>"
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
