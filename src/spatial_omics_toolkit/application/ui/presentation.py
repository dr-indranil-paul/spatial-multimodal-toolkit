from __future__ import annotations

from collections.abc import Callable
from graphlib import TopologicalSorter
from html import escape
import re
import subprocess
from textwrap import wrap
from urllib.parse import urlsplit

import streamlit as st

from spatial_omics_toolkit.application.contracts import WorkflowDescriptor


MODALITY_LABELS: dict[str, str] = {
    "CycIF": "Cyclic immunofluorescence (CycIF)",
    "HS": "Hyperspectral imaging (HS)",
    "FLIM": "Fluorescence lifetime imaging microscopy (FLIM)",
    "H&E": "Hematoxylin and eosin (H&E)",
    "Visium": "Visium spatial transcriptomics",
    "Integration": "Multimodal integration",
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
            --wb-control-size: 0.82rem;
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
        :is([data-testid="stMain"], [role="dialog"]) [data-testid="stWidgetLabel"] p {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            line-height: 1.35 !important;
            color: var(--wb-ink-800) !important;
        }
        :is([data-testid="stMain"], [role="dialog"]) [data-baseweb="input"] input,
        :is([data-testid="stMain"], [role="dialog"]) [data-baseweb="textarea"] textarea,
        :is([data-testid="stMain"], [role="dialog"]) [data-baseweb="select"],
        :is([data-testid="stMain"], [role="dialog"]) [data-baseweb="select"] input,
        [data-baseweb="popover"] [role="option"],
        [data-testid="stSelectboxVirtualDropdownEmpty"] li {
            font-size: var(--wb-control-size) !important;
        }
        /* Dropdown menus are portalled outside the main app content. */
        [data-baseweb="popover"] [role="option"] {
            line-height: 1.4;
        }
        [role="dialog"] [data-testid="stMarkdownContainer"] p {
            font-size: 0.84rem;
            line-height: 1.45;
        }
        [role="dialog"] [data-testid="stCaptionContainer"] p {
            color: var(--wb-ink-500);
            font-size: 0.78rem;
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
        [data-testid="stMain"] [class*="st-key-workspace_section_"]
        label[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p {
            color: #1d4ed8;
            font-weight: 700;
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
        div[data-testid="stButton"] button[kind="primary"]:not(:disabled) p {
            color: #ffffff !important;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            border-color: #3730a3;
            background: var(--wb-accent-hover);
            color: #ffffff;
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


def _stage_graph(descriptor: WorkflowDescriptor) -> str:
    """Render the execution DAG at a fixed, application-matched text scale."""
    stages = descriptor.stages
    indexes = {stage.stage_id: index for index, stage in enumerate(stages)}
    if len(indexes) != len(stages):
        raise ValueError("Workflow stages must have unique IDs.")
    dependencies = {stage.stage_id: stage.depends_on for stage in stages}
    for stage in stages:
        for dependency in stage.depends_on:
            if dependency not in indexes:
                raise ValueError(f"Unknown workflow stage dependency: {dependency}")
    tuple(TopologicalSorter(dependencies).static_order())

    def label_lines(value: str, width: int, prefix: str = "") -> str:
        lines = [line for paragraph in value.split("\n")
                 for line in (wrap(paragraph, width=width, break_long_words=False,
                                   break_on_hyphens=False) or [""])]
        if prefix:
            first = lines[0]
            lines[0] = f"<B>{escape(prefix, quote=True)}</B>{escape(first[len(prefix):], quote=True)}"
            return '<BR ALIGN="LEFT"/>'.join([lines[0], *(escape(line, quote=True) for line in lines[1:])])
        return '<BR ALIGN="LEFT"/>'.join(escape(line, quote=True) for line in lines)

    def label(fields: list[tuple[bool, str, str]], width: int) -> str:
        rows = []
        for heading, prefix, value in fields:
            face = "Inter" if heading else "Source Sans"
            # Graphviz rounds HTML font sizes; reserve space before applying exact CSS sizes.
            size = "15" if heading else "14"
            content = label_lines(f"{prefix} {value}" if prefix else value, width, prefix)
            if heading:
                content = f"<B>{content}</B>"
            rows.append(
                f'<TR><TD ALIGN="LEFT" BALIGN="LEFT"><FONT FACE="{face}" POINT-SIZE="{size}">'
                f'{content}</FONT></TD></TR>'
            )
        return '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">' + "".join(rows) + "</TABLE>>"

    dot = [
        'digraph workflow {',
        'graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.45", ranksep="0.8"];',
        'node [shape=box, style="rounded,filled", fillcolor="white", color="#a5b4fc", penwidth=1.5];',
        'edge [color="#64748b", fontcolor="#1e293b", arrowsize=0.8];',
    ]
    for index, stage in enumerate(stages):
        fields = [
            (True, "", stage.title),
            (False, "Purpose:", stage.purpose),
            *[(False, "Tool:", f"{tool.name} ({tool.version}) — {tool.role}") for tool in stage.tools],
            (False, "Inputs:", "; ".join(stage.inputs) or "None declared"),
            (False, "Outputs:", "; ".join(stage.outputs) or "None declared"),
            (False, "Condition / review:", stage.condition or "None declared"),
        ]
        dot.append(f'n{index} [id="stage-{index}", label={label(fields, 38)}];')
    for target, stage in enumerate(stages):
        for dependency in stage.depends_on:
            source = indexes[dependency]
            upstream = stages[source]
            fields = [
                (False, "Upstream outputs (when produced):", "; ".join(upstream.outputs) or "None declared"),
                (False, "Downstream condition / review:", stage.condition or "None declared"),
            ]
            dot.append(f'n{source} -> n{target} [id="edge-{source}-{target}", label={label(fields, 38)}];')
    dot.append("}")
    result = subprocess.run(
        ["dot", "-Tsvg"], input="\n".join(dot), text=True, capture_output=True, check=True, timeout=60,
    )
    # Graphviz emits dimensions in points. Match CSS pixels to viewBox units so
    # its SVG font sizes remain the same as the surrounding application.
    svg = result.stdout[result.stdout.index("<svg "):]
    root_end = svg.index(">")
    root = svg[:root_end]
    viewbox = re.search(r'viewBox="([^"]+)"', root)
    if viewbox is None:
        raise ValueError("Graphviz SVG is missing its viewBox")
    _, _, width, height = viewbox.group(1).split()
    root = re.sub(r'\bwidth="[^"]+"', f'width="{width}px"', root, count=1)
    root = re.sub(r'\bheight="[^"]+"', f'height="{height}px"', root, count=1)
    svg = root + ' role="img" aria-label="Workflow stages and directed dependencies"' + svg[root_end:]
    return f"""<div class="workflow-diagram"><style>
.workflow-diagram .viewport {{ width:100%; height:500px; overflow:auto; border:1px solid #e2e8f0;
                               border-radius:0.5rem; background:#f8fafc; }}
.workflow-diagram svg {{ display:block; max-width:none; }}
.workflow-diagram text[font-family="Source Sans"] {{ font-family:"Source Sans",sans-serif; font-size:13.44px; }}
.workflow-diagram text[font-family="Inter"] {{ font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; font-size:14.72px; font-weight:700; }}
</style>
<div class="viewport" tabindex="0" role="region" aria-label="Workflow diagram; scroll to see all stages">
{svg}</div></div>"""


def render_stage_flow(descriptor: WorkflowDescriptor) -> None:
    """Show the execution stages and their declared dependency relationships."""
    st.write(
        "Inputs and settings determine which paths run; optional and alternative paths do not all run. "
        "Arrows show declared stage dependencies. Tool versions describe the current workflow and may differ from a saved run."
    )
    if not descriptor.stages:
        st.caption("No analysis stages are declared for this workflow.")
        return
    st.markdown(_stage_graph(descriptor), unsafe_allow_html=True)
    st.caption("Scroll the diagram to see all stages and dependency annotations.")
    titles = {stage.stage_id: stage.title for stage in descriptor.stages}
    with st.expander("Workflow details (accessible text)"):
        for stage in descriptor.stages:
            dependencies = ", ".join(titles[item] for item in stage.depends_on) or "None"
            tools = "; ".join(
                f"{tool.name} ({tool.version}): {tool.role}"
                for tool in stage.tools
            ) or "None declared"
            fields = (
                ("Purpose", stage.purpose), ("Depends on", dependencies), ("Tools", tools),
                ("Inputs", "; ".join(stage.inputs) or "None declared"),
                ("Outputs", "; ".join(stage.outputs) or "None declared"),
                ("Condition / review", stage.condition or "None declared"),
            )
            st.markdown(
                f"<h3>{escape(stage.title)}</h3>"
                + "".join(f"<p><strong>{label}:</strong> {escape(value)}</p>" for label, value in fields),
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
        hero_left, hero_right = st.columns([4, 1], vertical_alignment="center")
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
