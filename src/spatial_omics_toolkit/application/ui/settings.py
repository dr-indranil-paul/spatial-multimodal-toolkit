from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import streamlit as st

from spatial_omics_toolkit.application.contracts import WorkflowDescriptor
from spatial_omics_toolkit.application.settings import (
    PROFILE_KEY, apply_values, default_settings, parse_settings, profile_metadata,
    read_user_settings, reset_values, save_user_settings, settings_directory,
)


def _clear_confirmations(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lower = key.lower()
            if isinstance(item, bool) and any(word in lower for word in ("confirm", "authoriz", "review", "verified")):
                value[key] = False
            else:
                _clear_confirmations(item)
    elif isinstance(value, list):
        for item in value:
            _clear_confirmations(item)


def _persist_editor(editor_key: str, draft_key: str) -> None:
    st.session_state[draft_key] = st.session_state[editor_key]


def render_settings(descriptor: WorkflowDescriptor, project_root: Path, parameters: dict[str, Any], key_prefix: str) -> dict[str, Any] | None:
    """Return a full replacement parameter dictionary only when Apply or Reset is pressed."""
    resolved_project = str(project_root.expanduser().resolve())
    project_key = hashlib.sha256(resolved_project.encode("utf-8")).hexdigest()
    prefix = f"{key_prefix}_settings_{descriptor.workflow_id}_{project_key}"
    defaults = default_settings(descriptor)
    with st.expander("Settings profiles", expanded=False):
        st.caption("Defaults are generated from the workflow controls and cannot be edited. Mappings and review confirmations remain explicit.")
        st.code(defaults, language="toml")
        st.download_button("Download read-only defaults", defaults, file_name=f"{descriptor.workflow_id}-defaults.toml", mime="application/toml", key=f"{prefix}_download_defaults")
        if st.button("Reset to defaults", key=f"{prefix}_reset"):
            result = reset_values(descriptor, parameters)
            _clear_confirmations(result)
            result[PROFILE_KEY] = profile_metadata(defaults)
            return result
        try:
            directory = settings_directory(project_root)
            names = sorted(path.name for path in directory.iterdir() if path.name.endswith(".toml") and not path.is_symlink() and path.is_file() and path.stat().st_uid == __import__("os").getuid() and path.name.lower() not in {"default.toml", "defaults.toml"})
        except (ValueError, OSError) as exc:
            st.warning(f"Select a writable project directory to use your own settings: {exc}")
            return None
        options = ["(new profile)", *names]
        select_key, select_state = f"{prefix}_select", f"{prefix}_selected_state"
        if select_key not in st.session_state or st.session_state[select_key] not in options:
            saved = st.session_state.get(select_state, options[0])
            st.session_state[select_key] = saved if saved in options else options[0]
        selected = st.selectbox(
            "User settings file", options, key=select_key,
            on_change=_persist_editor, args=(select_key, select_state),
        )
        name_key, name_state = f"{prefix}_name", f"{prefix}_name_state"
        if selected == "(new profile)":
            if name_key not in st.session_state:
                st.session_state[name_key] = st.session_state.get(name_state, "my-settings.toml")
            name = st.text_input(
                "New profile filename", key=name_key,
                on_change=_persist_editor, args=(name_key, name_state),
            )
        else:
            name = selected
        editor_key = f"{prefix}_editor"
        loaded_key = f"{prefix}_loaded"
        loaded_identity = (resolved_project, selected, name)
        file_key = hashlib.sha256(f"{selected}\0{name}".encode("utf-8")).hexdigest()
        draft_key = f"{prefix}_draft_{file_key}"
        try:
            if draft_key not in st.session_state:
                st.session_state[draft_key] = defaults if selected == "(new profile)" else read_user_settings(project_root, selected)
            if st.session_state.get(loaded_key) != loaded_identity or editor_key not in st.session_state:
                st.session_state[editor_key] = st.session_state[draft_key]
                st.session_state[loaded_key] = loaded_identity
        except (OSError, ValueError, UnicodeError) as exc:
            st.error(f"Cannot open settings: {exc}")
            return None
        uploaded = st.file_uploader("Import TOML into editor", type=["toml"], key=f"{prefix}_upload")
        if uploaded is not None and st.button("Use imported text", key=f"{prefix}_import"):
            try:
                raw = uploaded.getvalue()
                if len(raw) > 1024 * 1024:
                    raise ValueError("Settings file exceeds 1 MiB")
                st.session_state[editor_key] = raw.decode("utf-8")
                st.session_state[draft_key] = st.session_state[editor_key]
            except (ValueError, UnicodeError) as exc:
                st.error(f"Cannot import settings: {exc}")
        text = st.text_area("Editable user TOML", key=editor_key, height=280, on_change=_persist_editor, args=(editor_key, draft_key))
        st.download_button("Export editor TOML", text, file_name=name if name.endswith(".toml") else "settings.toml", mime="application/toml", key=f"{prefix}_export")
        if st.button("Create user file" if selected == "(new profile)" else "Save user file", key=f"{prefix}_save"):
            try:
                parse_settings(descriptor, text)
                save_user_settings(project_root, name, text, create=selected == "(new profile)")
                st.success(f"Saved {name}. Select it and apply it to use these settings.")
            except (ValueError, OSError, UnicodeError) as exc:
                st.error(f"Cannot save settings: {exc}")
        if selected != "(new profile)" and st.button("Apply saved settings", key=f"{prefix}_apply"):
            try:
                source = read_user_settings(project_root, selected)
                values = parse_settings(descriptor, source)
                result = apply_values(reset_values(descriptor, parameters), values)
                _clear_confirmations(result)
                result[PROFILE_KEY] = profile_metadata(source, project_root=project_root, name=selected)
                return result
            except (ValueError, OSError, UnicodeError) as exc:
                st.error(f"Cannot apply settings: {exc}")
    return None
