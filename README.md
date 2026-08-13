# SpatialOmics Toolkit

**Institute-grade spatial and single-cell multiomics analytics platform.**

## Overview

SpatialOmics Toolkit provides native Python contracts, validation engine, pseudobulk pre-aggregators, CLI tools, Streamlit application interfaces, and Nextflow SLURM pipelines for high-throughput single-cell and spatial omics analytics (scRNA/snRNA, Visium, Xenium, Spatial Proteomics).

## Architecture

- **Python Core Package (`spatial_omics_toolkit`)**: Primary business logic, Pydantic contracts, manifest validation, pseudobulk pre-aggregation, and CLI.
- **Nextflow DSL2 Orchestrator**: SLURM batch-wrapped workflow engine for high-performance process execution.
- **Streamlit Application**: Scientist-facing GUI with OME-Zarr pyramidal tiling and server-side spatial rasterization.
- **Containerized R Stages**: Polyglot statistical engines (`edgeR`, `limma-voom`, `muscat`) executing strictly on pre-aggregated sample pseudobulk matrices.

## Installation

```bash
pip install -e .
# or, for the CyCIF imaging steps (registration/segmentation/conversion):
pip install -e ".[imaging]"
```

## Notebook quickstart (CyCIF / spatial proteomics)

The CyCIF pipeline is driven one numbered step at a time from a single
top-level import — tab-complete `sot.step<TAB>` in Jupyter to see every step
in pipeline order:

```python
import pandas as pd
import spatial_omics_toolkit as sot

# Load a cells x markers feature table (e.g. from Cellpose/MCMICRO/PIPEX segmentation output)
cells = pd.read_csv("features.csv")

# step07: per-core z-score marker-threshold cell typing
cells = sot.step07_classify_cells(cells, sot.PREVAIL_BARRETTS_MARKER_CONFIG, core_col="slide_scene")

# step08: composite per-cell program scores (EL/IH/ME EMT programs, or your own ProgramDefinitions)
cells = sot.step08_score_programs(cells, sot.PREVAIL_BARRETTS_EMT_PROGRAMS)

# step09: mean-based per-core summary table (cell-type fractions, marker means, program means)
cores = sot.step09_aggregate_per_core(cells, core_col="slide_scene")

# step10: per-core density / neighbor-enrichment spatial features
spatial = sot.step10_spatial_features(cells, core_col="slide_scene")

# step11: spatial null permutation control -- required before any niche/enrichment claim
null_check = sot.step11_spatial_null(
    cells, source_type="Epithelial", target_type="Myeloid_immune",
    core_col="slide_scene", k=5, n_permutations=1000,
)

# step12: BH-corrected two-group comparison (with an explicit biological-unit warning)
metadata = pd.read_csv("progression_map.csv")
cores = cores.merge(metadata[["slide_scene", "progression_group"]], on="slide_scene")
comparison = sot.step12_compare_groups(
    cores, group_col="progression_group",
    metrics=["Epithelial", "Hybrid_epithelial", "immune_fraction"],
)
```

`sot.CellTypeMarkerConfig` and `sot.ProgramDefinition` are how you point
`step07`/`step08` at your own marker panel instead of the PREVAIL Barrett's
example config — see `spatial_omics_toolkit.cycif` module docstring for the
full step table (steps 01-05 cover raw-image intake, conversion,
registration, export, and segmentation ahead of this tabular stage).

## CLI Usage (production job orchestration)

```bash
# Validate intake manifest
spatial-omics validate --manifest manifest.json

# Export JSON schemas
spatial-omics schema --output-dir ./schemas

# Plan execution DAG
spatial-omics plan --job job.json --output plan.json

# Check status
spatial-omics status --job-id <UUID>
```

## License

Apache-2.0
