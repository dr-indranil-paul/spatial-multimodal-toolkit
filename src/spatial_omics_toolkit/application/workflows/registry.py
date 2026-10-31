from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from types import ModuleType
from typing import Any

from spatial_omics_toolkit.application.contracts import WorkflowDescriptor


SUPPORTED_MODALITIES = frozenset({"CycIF", "HS", "FLIM", "H&E", "Visium"})


def register_workflow(module: ModuleType) -> WorkflowDescriptor:
    """Validate and register one fully implemented workflow module."""
    required = ("describe", "preflight", "execute")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise ValueError(f"Workflow module {module.__name__} lacks required API: {', '.join(missing)}")
    descriptor = WorkflowDescriptor.model_validate(module.describe())
    if descriptor.modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"Unsupported modality {descriptor.modality!r} in {module.__name__}")
    return descriptor


_REGISTRY_LOAD_WARNINGS: list[str] = []


@lru_cache(maxsize=1)
def _registry() -> dict[str, tuple[ModuleType, WorkflowDescriptor]]:
    registry: dict[str, tuple[ModuleType, WorkflowDescriptor]] = {}
    _REGISTRY_LOAD_WARNINGS.clear()
    package = importlib.import_module("spatial_omics_toolkit.application.workflows")
    for info in pkgutil.iter_modules(package.__path__):
        if info.name in {"registry", "__init__"} or info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{package.__name__}.{info.name}")
            if not all(callable(getattr(module, name, None)) for name in ("describe", "preflight", "execute")):
                continue
            descriptor = register_workflow(module)
            if descriptor.workflow_id in registry:
                raise ValueError(f"Duplicate workflow ID: {descriptor.workflow_id}")
            registry[descriptor.workflow_id] = (module, descriptor)
        except Exception as exc:
            _REGISTRY_LOAD_WARNINGS.append(f"{info.name}: {exc}")
    return registry


def registry_warnings() -> list[str]:
    _registry()
    return list(_REGISTRY_LOAD_WARNINGS)


def refresh_registry() -> None:
    """Clear cached registry after workflow modules are installed or changed."""
    _registry.cache_clear()


def list_workflows() -> list[WorkflowDescriptor]:
    return [descriptor for _, descriptor in sorted(_registry().values(), key=lambda item: item[1].workflow_id)]


def get_workflow(workflow_id: str) -> tuple[ModuleType, WorkflowDescriptor]:
    try:
        return _registry()[workflow_id]
    except KeyError as exc:
        raise KeyError(f"Workflow {workflow_id!r} is not registered as implemented") from exc


def workflow_catalog() -> dict[str, dict[str, Any]]:
    return {descriptor.workflow_id: descriptor.model_dump(mode="json") for descriptor in list_workflows()}
