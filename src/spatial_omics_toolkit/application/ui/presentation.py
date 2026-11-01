from __future__ import annotations

from collections.abc import Callable
from html import escape

import streamlit as st


MODALITY_LABELS: dict[str, str] = {
    "CycIF": "Cyclic immunofluorescence (CycIF)",
    "HS": "Hyperspectral imaging (HS)",
    "FLIM": "Fluorescence lifetime imaging microscopy (FLIM)",
    "H&E": "Hematoxylin and eosin (H&E)",
    "Visium": "Visium spatial transcriptomics",
    "Integration": "Integration",
}

_MODALITY_CARDS: tuple[tuple[str, str, str, str], ...] = (
    ("CycIF", MODALITY_LABELS["CycIF"], "Explore multiplexed protein staining across imaging rounds.", "#c05e56"),
    ("HS", MODALITY_LABELS["HS"], "Work with spectral imaging data and its recorded channels.", "#438c87"),
    ("FLIM", MODALITY_LABELS["FLIM"], "Examine fluorescence lifetime imaging measurements.", "#7580bb"),
    ("H&E", MODALITY_LABELS["H&E"], "Work with H&E-stained tissue images.", "#c18b48"),
    ("Visium", MODALITY_LABELS["Visium"], "Explore spatial gene-expression data from Visium assays.", "#618d62"),
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
        [data-testid="stSidebar"] {
            min-width: 290px;
            max-width: 290px;
            background: #f5f3ee;
            border-right: 1px solid var(--workspace-line);
        }
        [data-testid="stSidebar"] * { color: #303b40; }
        [data-testid="stSidebar"] [role="radiogroup"] { gap: .25rem; }
        [data-testid="stSidebar"] label[data-baseweb="radio"] {
            width: 100%; box-sizing: border-box; margin: 0; padding: .6rem .75rem;
            border-radius: 10px; transition: background .15s ease;
        }
        [data-testid="stSidebar"] label[data-baseweb="radio"]:hover { background: #eaece5; }
        [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) { background: #e0e9e0; font-weight: 650; }
        [data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child { display: none; }
        [data-testid="stSidebar"] label[data-baseweb="radio"] > div:last-child { margin-left: 0; }
        [data-testid="stSidebar"] label[data-baseweb="radio"]:focus-within { outline: 2px solid #327d68; outline-offset: 2px; }
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
        .workspace-hero {
            position: relative;
            overflow: hidden;
            margin: 0 0 2.2rem;
            padding: clamp(1.7rem, 5vw, 3.4rem);
            border: 1px solid #e6e1d7;
            border-radius: 24px;
            background: linear-gradient(118deg, #fffefa 12%, #f7f2e9 64%, #f0f4ef 100%);
            box-shadow: 0 12px 36px rgba(48, 54, 47, .055);
        }
        .workspace-kicker {
            margin: 0 0 .65rem;
            color: #637a72;
            font-size: .77rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .workspace-hero h1 {
            max-width: 760px;
            margin: 0;
            color: #26353a;
            font-size: clamp(2.15rem, 5vw, 3.55rem);
            font-weight: 650;
            letter-spacing: -.045em;
            line-height: 1.08;
        }
        .workspace-hero p {
            max-width: 600px;
            margin: .95rem 0 0;
            color: #5e6c6d;
            font-size: 1.08rem;
            line-height: 1.6;
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
            width: 2.25rem;
            height: .25rem;
            margin: .1rem 0 .9rem;
            border-radius: 2rem;
        }
        .workspace-card-title {
            min-height: 2.9rem;
            margin: 0 0 .35rem;
            color: #29383c;
            font-size: 1.05rem;
            font-weight: 650;
            line-height: 1.35;
        }
        .workspace-card-copy {
            min-height: 3.2rem;
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
        .workspace-steps {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .8rem;
            margin-top: .8rem;
        }
        .workspace-step {
            display: flex;
            align-items: center;
            gap: .7rem;
            min-width: 0;
            padding: .85rem .95rem;
            border: 1px solid #e8e6df;
            border-radius: 13px;
            background: rgba(255,255,253,.72);
            color: #47575a;
            font-size: .92rem;
        }
        .workspace-step-number {
            display: grid;
            flex: 0 0 1.7rem;
            width: 1.7rem;
            height: 1.7rem;
            place-items: center;
            border-radius: 50%;
            background: #e7eee8;
            color: #365f52;
            font-size: .8rem;
            font-weight: 700;
        }
        @media (max-width: 760px) {
            [data-testid="stSidebar"] { min-width: min(260px, 82vw); max-width: min(260px, 82vw); }
            [data-testid="stMainBlockContainer"] { padding: 1.4rem 1rem 2.5rem; }
            .workspace-steps { grid-template-columns: 1fr; }
            .workspace-card-title, .workspace-card-copy { min-height: 0; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home(navigate: Callable[[str], None]) -> None:
    """Render the welcoming home screen and its direct workspace actions."""
    st.markdown(
        """
        <section class="workspace-hero">
          <div class="workspace-kicker">Spatial Toolkit</div>
          <h1>Your data. A clearer picture.</h1>
          <p>Choose a modality, then run its ready-to-use example or switch to your own data. Examples need no project setup.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.button("Open project", type="primary", on_click=navigate, args=("Project",), key="home_open_project")

    st.markdown('<h2 class="workspace-section-title">Choose an analysis</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="workspace-section-note">Choose a modality workspace. Start with Run example, or select your own data and follow the guided setup.</p>',
        unsafe_allow_html=True,
    )
    for row_start in range(0, len(_MODALITY_CARDS), 2):
        columns = st.columns(2, gap="medium")
        for column, (code, title, description, accent) in zip(
            columns, _MODALITY_CARDS[row_start : row_start + 2]
        ):
            with column:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="workspace-accent" style="background:{escape(accent, quote=True)}"></div>'
                        f'<h3 class="workspace-card-title">{escape(title)}</h3>'
                        f'<p class="workspace-card-copy">{escape(description)}</p>',
                        unsafe_allow_html=True,
                    )
                    st.button(
                        f"Open {code}",
                        on_click=navigate,
                        args=(code,),
                        key=f"home_modality_{code}",
                        help=f"Open {title}",
                        use_container_width=True,
                    )

    st.markdown('<h2 class="workspace-section-title">A simple path through your work</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="workspace-steps" aria-label="Three steps">
          <div class="workspace-step"><span class="workspace-step-number">1</span><span>Choose data</span></div>
          <div class="workspace-step"><span class="workspace-step-number">2</span><span>Run your analysis</span></div>
          <div class="workspace-step"><span class="workspace-step-number">3</span><span>Explore results</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<h2 class="workspace-section-title">Working across modalities?</h2>', unsafe_allow_html=True)
    st.caption("Integration is optional; each modality can be used independently.")
    st.button(
        "Explore integration",
        on_click=navigate,
        args=("Integration",),
        key="home_integration",
    )
