"""CLI interface module for spatial-omics command suite."""

from importlib import import_module

__all__ = ["app"]


def __getattr__(name: str):
    """Resolve the Typer application without preloading its module."""
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    app = import_module("spatial_omics_toolkit.cli.main").app
    globals()["app"] = app
    return app


def __dir__() -> list[str]:
    """Expose the public application name without importing it."""
    return sorted(set(globals()) | set(__all__))
