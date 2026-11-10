"""Image-only C2 provenance and coarse orientation, not biological registration.

Stage 0 inspects TIFF headers and reads producer code as text only. Stage 1
sums the original HS stack, extracts candidate voids and a bright peri-void
OMNI proxy, and compares these with CyCIF voids and Ecad. All angles compete
on the same two equally weighted Dice scores. No labels, cell tables, zones,
or reconstructed arrays are inputs. A high score is a review hypothesis only.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import cv2
import matplotlib
import matplotlib.figure
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import skimage
import tifffile
from numpy.typing import NDArray
from scipy import ndimage, optimize
from skimage import filters, morphology, transform
from guided_serial_registration import RegistrationConfig, estimate_affine_orb

FloatImage = NDArray[np.float64]
BoolImage = NDArray[np.bool_]


class Stage1PrerequisiteError(ValueError):
    """Expected missing image evidence; never a reason to use a label fallback."""

    def __init__(
        self, message: str, details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details or {}


SOURCE = Path("02_data_sources/04_collaborators/from_dylan/20260903_clustering_files")
ANALYSIS = Path("03_analysis/01_exploratory_analysis")
ROUTE = Path("03_for_dylan_paper/01_omni_cycif_one_core_exploratory")
MANIFEST = ANALYSIS / "03_tables" / ROUTE / "input_data/cycif_tma1_c2_source_manifest.json"
REGISTERED = Path(
    "/home/groups/EmiliLab/ChinData/Cyclic_Workflow/"
    "cmIF_2026-05-15_BE-TMA/RegisteredImages/hBE-TMA1-1_sceneC2"
)
AF_DAPI = REGISTERED / (
    "Registered-R5Q_R5Qc2.R5Qc3.R5Qc4.R5Qc5_hBE-TMA1-1_sceneC2_c1_ORG.tif"
)
REFERENCES = {
    "DAPI": "Registered-R1_MUC1.MUC5AC.MUC2.MUC6_hBE-TMA1-1_sceneC2_c1_ORG.tif",
    "Ecad": "Registered-R3_Vim.CD11c.CD45.Ecad_hBE-TMA1-1_sceneC2_c5_ORG.tif",
}
CORES = ("A2", "B1", "B2", "C2", "A1", "B3", "C1", "C3")
WORK_SIZE = 512
DISPLAY_SIZE = 1536
CONTEXT_SOURCE_PACKETS: dict[str, str] = {
    "prior_original_raster_search": "02_spatial_proteomics:SRC-2026-10-22-084",
    "dylan_artifact_search": "04_dylan_bodipy_proteomics:SRC-2026-10-22-017",
    "current_plan": "04_dylan_bodipy_proteomics:SRC-2026-10-22-023",
}
MASK_RECIPE: dict[str, object] = {
    "normalization": "finite nonnegative intensities; positive p1/p99, clipped 0..1",
    "resampling": "isotropic longest-side 512, bilinear with anti-aliasing",
    "foreground": "Gaussian sigma 1 then Otsu, remove objects smaller than 8 px",
    "holes": (
        "disk-2 closing; convex hull of closed foreground as tissue support; "
        "retain nonforeground components inside support with area >=8 work px; "
        "includes open-to-edge candidate voids, not confirmed anatomical lumens"
    ),
    "cycif_structure": "pixelwise maximum of normalized DAPI, Ecad and usable R5Q",
    "R5Q_alignment": (
        "R5Q c1 DAPI to R1 DAPI reviewed ORB/RANSAC residual affine; "
        "apply forward affine to eligible AF at work resolution, bilinear, zero exterior"
    ),
    "omni_gland_proxy": "bright OMNI foreground within disk-5 dilation of voids",
    "cycif_epithelium": "bright Ecad foreground (same smoothing/Otsu recipe)",
    "minimum_internal_evidence": "at least 8 candidate-void and epithelial pixels",
    "outer_outline": "not scored directly; convex hull only bounds candidate voids",
}
SEARCH: dict[str, object] = {
    "coarse_states": "rotation sectors 0,90,180,270 times reflected False/True",
    "residual_degrees": list(range(0, 90, 2)),
    "angle_fit": "fixed per candidate; no continuous angle refinement",
    "optimizer": "scipy.optimize.minimize Powell; Gaussian-softened mask Dice",
    "optimizer_options": {"maxiter": 35, "maxfev": 220, "xtol": 0.05, "ftol": 0.0001},
    "cost_bound": (
        "360 candidates; each at most 220 Powell evaluations plus one "
        "initialization comparison; two cached nonzero-crop soft-mask warps "
        "onto expanded transformed bounds per evaluation, then two "
        "nearest-neighbor final warps; no native-resolution optimization"
    ),
    "scale_bounds": "0.5..2.0 times internal-mask RMS-radius ratio initialization",
    "translation_bounds": "+/- 0.25 fixed-canvas diagonal around centroid match",
    "initialization": "centroid and RMS radius of union(hole, epithelial) pixels",
    "soft_mask_sigma": 1.0,
    "final_scores": (
        "exact rasterized-union lumen and epithelial Dice, separately; "
        "entire transformed raster mass remains in the denominator; "
        "clipping is raster mass geometrically outside the fixed canvas"
    ),
    "rank": "descending equal-weight arithmetic mean; stable candidate-ID ties",
}
LIMITATIONS = [
    "Exploratory coarse orientation hypotheses, not validated biological registration.",
    "OMNI bright peri-void signal is a gland-like proxy, not an epithelial identity.",
    (
        "Candidate voids include open-to-edge regions within convex-hull support, "
        "concavities, tears and artifacts; they have no established anatomical "
        "identity and require manual full-core review."
    ),
    "Unknown OMNI sampling implies relative image-pixel scale, not calibrated micrometers.",
    "Equal registered grid dimensions do not independently validate R5Q registration.",
    (
        "The validated R5Q c1 DAPI affine is shared by its same-round c2–c5 "
        "exports; per-channel optical shifts are not independently established."
    ),
    "Bounded Powell fits are local optima; this screen does not establish "
    "a global registration optimum.",
    "Original historical label raster absent: Stage 0 item 2 and Stages 3–6 blocked.",
    "Stage 0 item 3 and image-only Stage 1 do not require reconstructed labels.",
    "No Stage 2–6 analysis and no reconstruction arrays are consumed.",
]


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": _sha256(path),
            "bytes": path.stat().st_size}


def _save_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _ome_pixels(xml: str | None) -> dict[str, str]:
    if not xml:
        return {}
    root = ElementTree.fromstring(xml)
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "Pixels":
            return dict(element.attrib)
    return {}


def _physical_size(pixels: dict[str, str]) -> dict[str, float | None]:
    """Use explicit OME units only; missing or ambiguous units remain unknown."""
    conversion = {"µm": 1.0, "um": 1.0, "nm": 0.001, "mm": 1000.0, "m": 1e6}
    result: dict[str, float | None] = {}
    for axis in ("X", "Y"):
        value = pixels.get(f"PhysicalSize{axis}")
        factor = conversion.get(pixels.get(f"PhysicalSize{axis}Unit", ""))
        result[f"pixel_size_{axis.lower()}_um"] = (
            float(value) * factor if value is not None and factor is not None else None
        )
    return result


def inspect_tiff(path: Path, role: str) -> dict[str, object]:
    """Read headers, not pixel values; resolution tags are retained verbatim.

    TIFF print-resolution tags are not silently promoted to microscope sampling.
    OME physical sizes with explicit units are reported separately.
    """
    with tifffile.TiffFile(path) as source:
        series = source.series[0]
        page = source.pages[0]
        tags: dict[str, object] = {}
        for name in ("XResolution", "YResolution", "ResolutionUnit", "Software",
                     "ImageDescription", "SamplesPerPixel", "PhotometricInterpretation"):
            if name in page.tags:
                value = page.tags[name].value
                tags[name] = int(value) if isinstance(value, int) else value
        pixels = _ome_pixels(source.ome_metadata)
        return {
            **_binding(path), "role": role, "axes": series.axes,
            "shape": list(series.shape), "dtype": str(series.dtype),
            "page_count": len(source.pages), "series_count": len(source.series),
            "tiff_tags": tags, "ome_pixels": pixels, **_physical_size(pixels),
            "physical_size_source": "embedded OME with explicit units" if pixels else None,
        }


def _sum_axis(metadata: dict[str, Any]) -> int:
    """Require a metadata-resolved leading stack axis and spatial YX planes.

    QYX is a generic TIFF page stack, not authenticated spectral channel names.
    The historical producer's explicit axis-0 reduction justifies summing that
    stack; no biological or spectral channel identities are inferred from Q.
    """
    axes = str(metadata["axes"])
    shape = metadata["shape"]
    if axes not in {"CYX", "SYX", "QYX", "IYX"} or len(shape) != 3:
        raise Stage1PrerequisiteError(
            f"Cannot justify leading stack/YX sum: axes={axes}, shape={shape}"
        )
    return 0


def _sum_hs(
    stack: NDArray[Any], metadata: dict[str, Any],
) -> NDArray[Any]:
    """Match Dylan's default NumPy reduction; preserve its dtype promotion."""
    axis = _sum_axis(metadata)
    if list(stack.shape) != metadata["shape"]:
        raise Stage1PrerequisiteError("Loaded HS shape disagrees with TIFF metadata")
    return np.sum(stack, axis=axis)


def _paths(project_root: Path) -> tuple[Path, dict[str, Path], list[Path]]:
    manifest = project_root / MANIFEST
    references = {name: manifest.parent / "registered_references" / filename
                  for name, filename in REFERENCES.items()}
    af = [REGISTERED / (
        "Registered-R5Q_R5Qc2.R5Qc3.R5Qc4.R5Qc5_"
        f"hBE-TMA1-1_sceneC2_c{channel}_ORG.tif"
    ) for channel in range(2, 6)]
    return manifest, references, af


def _artifact_inventory(project_root: Path) -> pd.DataFrame:
    """Hash known source artifacts only; never read table values or raster arrays."""
    sibling = project_root.parent / "04_dylan_bodipy_proteomics"
    candidates = [
        (project_root / "01_documents/figures_from_dylan/image33.pdf", "static_map",
         "C2 K9 figure candidate; no native pixels, authenticated IDs or orientation",
         'Identifies C2, \"Unsupervised clustering (k=9),\" and a cluster legend'),
        (project_root / "01_documents/figures_from_dylan/images2.pdf", "mean_spectrum",
         "whole-tissue mean spectrum, not per-cluster mean spectra",
         'Panel labeled \"Mean Tissue Spectrum\"; whole-tissue spectral summary'),
        (sibling / "04_manuscripts/02_source_presentations/"
         "2026-08-06_new_pr_sequencing_nature_figure_dh.pdf", "static_map",
         "source presentation; not an authenticated native C2 K9 raster",
         "Figure 4 / BE Tissue Microarray cluster material; no native label raster"),
        (sibling / "04_manuscripts/02_source_presentations/"
         "2026-08-14_yi_additional_figure_material_dh.pptx", "static_map",
         "PCA cluster visual candidate; slide-9 core and historical IDs unconfirmed",
         "Slide 9 describes PCA clusters, cluster IDs and a composite image; "
         "its core is unconfirmed. Slide 10 contains A2 cluster statistics"),
        (sibling / "04_manuscripts/01_figures_and_tables/"
         "figures_04_05_tma_progression_clusters_dh/Panel C/TMA1/"
         "TMAFigureCollage.pptx", "static_map",
         "TMA1 artwork; no authenticated C2 native raster",
         "TMA1 composition collage; no authenticated C2 K9 pixel raster"),
    ]
    for folder in (
        sibling / "03_analysis/03_tables/dylan_imaging/eight_core_clustering_zscore",
        sibling / "03_analysis/04_plots/00_workspace/guided_notebook/omni_state_clustering",
    ):
        for stem, role, limitation, content_summary in (
            ("omni_core_state_abundances", "state_counts",
             "different direct 10-feature K-means model; not historical PCA5/K9",
             "Per-core/state pixel counts and fractions from a separate "
             "direct 10-feature K-means export, not the historical shared PCA5/K9"),
            ("omni_state_centers", "mean_spectrum_candidate",
             "10-feature centroids, not channel-wise per-cluster mean spectra",
             "Centroids of ten named FLIM/HS features, not per-cluster mean spectra"),
        ):
            for suffix in (".tsv", ".parquet"):
                candidates.append(
                    (folder / (stem + suffix), role, limitation, content_summary)
                )
    rows: list[dict[str, object]] = []
    for path, role, limitation, content_summary in candidates:
        exists = path.is_file()
        rows.append({"path": str(path.resolve()), "exists": exists,
                     "sha256": _sha256(path) if exists else None,
                     "role": role, "limitation": limitation,
                     "content_summary": content_summary,
                     "content_summary_source_packet": (
                         CONTEXT_SOURCE_PACKETS["dylan_artifact_search"]
                     ),
                     "content_summary_basis": "prior source-artifact search; no values loaded",
                     "stage1_input": False, "values_read": False,
                     "authorship": (
                         "Dylan source folder / project provenance; "
                         "not independently authenticated"
                     )})
    return pd.DataFrame(rows)


def _code_provenance(project_root: Path) -> dict[str, object]:
    original = project_root / SOURCE / "OpenCV_multiimage_8input.py"
    reconstruction = project_root / ANALYSIS / "02_scripts/dylan_k9_reconstruction.py"
    # Read as text only: neither code path is imported or executed.
    original_text = original.read_text(encoding="utf-8-sig")
    reconstruction_text = reconstruction.read_text(encoding="utf-8")
    evidence_terms = ("PCA", "kmeans", "sumsigzD", "final_Kdim")
    evidence = {term: [index for index, line in enumerate(original_text.splitlines(), 1)
                       if term in line] for term in evidence_terms}
    inputs = [_binding(project_root / SOURCE / "Original" / filename)
              for core in CORES
              for filename in (f"{core}_all2048.tif", f"{core}Image2048.tif")]
    return {
        "original_code": _binding(original),
        "reconstruction_code": _binding(reconstruction),
        "original_evidence_line_numbers": evidence,
        "reconstruction_local_mapping_present_in_code": "label + 1" in reconstruction_text,
        "code_executed_or_imported": False, "input_bindings": inputs,
        "core_order": list(CORES),
        "code_stated_method": [
            "10-feature stacks and separate axis-0 summed mask sources; center crop 2000x2000",
            "Lifetime features 3/4 outside 0..10 set to zero; nonfinite features nan_to_num",
            "Per-core per-feature z-score, then sum-source threshold 40000 zeros features",
            "Joint PCA5 across eight cores including zero-masked pixels, then OpenCV K9",
            "OpenCV EPS+MAX_ITER, 5 iterations, epsilon 0.1, 5 attempts, KMEANS_PP_CENTERS",
            "Original lacks explicit seed; deterministic reconstruction adds seed 20261022",
            "Original plots C2 fourth-core slice; no historical K9 raster save identified",
        ],
        "historical_ids": "Dylan's displayed Cluster 1–9",
        "reconstruction_crosswalk": "numeric 0–8 -> R1–R9 by label+1; reconstruction-local only",
        "crosswalk_authenticated": False,
        "reconstruction_arrays_read": False,
    }


def _report_data(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path, references, af_paths = _paths(project_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    geometry = manifest["image_geometry"]
    rows = [inspect_tiff(project_root / SOURCE / "Original/C2Image2048.tif", "OMNI_sum_source")]
    try:
        rows[0]["sum_axis"] = _sum_axis(rows[0])
        rows[0]["derived_axes"] = "YX"
        rows[0]["derived_shape_yx"] = rows[0]["shape"][1:]
        rows[0]["derived_image_storage"] = "in-memory only; not a located saved sum"
        rows[0]["stage1_sum_dtype"] = (
            "NumPy default np.sum(stack, axis=0), matching original producer"
        )
        rows[0]["sum_status"] = "in-memory axis-0 sum permitted by TIFF stack/YX and producer code"
    except ValueError as error:
        rows[0]["sum_status"] = str(error)
        rows[0]["derived_axes"] = None
    rows[0]["channel_identity_limitation"] = (
        "Generic Q/I axes identify pages, not spectral channel identities"
    )
    for role, path in references.items():
        row = inspect_tiff(path, role)
        row["manifest_geometry_matches"] = (
            row["axes"] == "YX" and row["shape"] == [geometry["height"], geometry["width"]]
        )
        row["registered_parent_ome_pixels"] = geometry["metadata"].get("ome_pixels", {})
        row["registered_parent_physical_size"] = _physical_size(row["registered_parent_ome_pixels"])
        rows.append(row)
    reference_grid_ok = all(row["manifest_geometry_matches"] for row in rows[1:])
    for path in af_paths:
        if not path.is_file():
            rows.append({"role": "R5Q_AF", "path": str(path), "available": False,
                         "usable": False, "grid_matches": False})
            continue
        row = inspect_tiff(path, "R5Q_AF")
        row["grid_matches"] = reference_grid_ok and (
            row["axes"] == "YX" and row["shape"] == rows[1]["shape"]
        )
        expected_scale = rows[1]["registered_parent_physical_size"]
        scale_conflicts = [
            key for key in ("pixel_size_x_um", "pixel_size_y_um")
            if row[key] is not None and expected_scale[key] is not None
            and not math.isclose(row[key], expected_scale[key], rel_tol=1e-6)
        ]
        row["explicit_physical_scale_conflicts"] = scale_conflicts
        row["grid_eligible"] = row["grid_matches"] and not scale_conflicts
        row["usable"] = False
        row["origin_verified"] = False
        row["registered_origin_yx"] = None
        row["transform_to_dapi"] = None
        row["registration_evidence"] = (
            "matching headers only; requires R5Q DAPI residual-affine verification"
        )
        rows.append(row)
    if AF_DAPI.is_file():
        af_dapi = inspect_tiff(AF_DAPI, "R5Q_DAPI_validation")
        af_dapi["grid_matches"] = reference_grid_ok and (
            af_dapi["axes"] == "YX" and af_dapi["shape"] == rows[1]["shape"]
        )
        rows.append(af_dapi)
    else:
        rows.append({
            "role": "R5Q_DAPI_validation", "path": str(AF_DAPI),
            "available": False, "grid_matches": False,
        })
    return rows, {
        "manifest": _binding(manifest_path),
        "historical_sum_generator": _binding(
            project_root / SOURCE / "OpenCV_multiimage_8input.py"
        ),
        "registered_parent_ome": {
            "path": geometry["source"], "manifest_recorded_sha256": geometry["source_sha256"],
            "ome_pixels": geometry["metadata"].get("ome_pixels", {}),
            "provenance": (
                "source OME fields recorded in bound local manifest; "
                "not remeasured raw CZI"
            ),
        },
        "omni_to_cycif_transform": None,
        "registered_grid_matches_manifest": reference_grid_ok,
    }


def collect_stage0_report(
    project_root: Path, tables_dir: Path, objects_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Save metadata/provenance without loading image pixels or label arrays.

    Supplied directories are the notebook's canonical per-analysis routes and
    are created if absent. Existing unrelated historical outputs are untouched.
    """
    tables = tables_dir
    objects = objects_dir
    tables.mkdir(parents=True, exist_ok=True)
    objects.mkdir(parents=True, exist_ok=True)
    rows, source_info = _report_data(project_root)
    metadata = pd.DataFrame(rows)
    artifacts = _artifact_inventory(project_root)
    metadata_path = tables / "stage0_image_metadata.csv"
    artifact_path = tables / "dylan_cluster_artifact_candidates.csv"
    metadata.to_csv(metadata_path, index=False)
    artifacts.to_csv(artifact_path, index=False)
    summary: dict[str, object] = {
        "schema_version": 1, "stage": "Stage 0 item 3 image metadata/provenance",
        "claim_label": "Exploratory", "generator": _binding(Path(__file__)),
        "context_source_packets": CONTEXT_SOURCE_PACKETS,
        "original_label_raster_found": False,
        "missing_artifact": (
            "Dylan's original per-pixel TMA1/C2 OMNI K9 label raster "
            "with authenticated historical cluster IDs and axis convention"
        ),
        "stage0_item2_status": "blocked_missing_original_label_raster",
        "stage0_item3_status": "reported_image_metadata_and_source_provenance",
        "stage1_image_only": True,
        "stage1_status": (
            "authorized_image_only; execution status is recorded separately "
            "in stage1_orientation_summary.json"
        ),
        "stage2_status": "not_requested_not_run",
        "stages3_to6_status": "blocked_missing_original_label_raster",
        "blocked_stages": [3, 4, 5, 6],
        "reconstructed_labels_used": False,
        "images": rows, **source_info,
        "reconstruction_provenance": _code_provenance(project_root),
        "dylan_artifacts": artifacts.to_dict(orient="records"),
        "artifact_search_scope": (
            "known scout-discovered Dylan figure/presentation and "
            "state-export paths; not exhaustive"
        ),
        "per_cluster_mean_spectra": (
            "not identified; whole-tissue spectrum and feature centers "
            "are not substitutes"
        ),
        "outputs": {"metadata": _binding(metadata_path), "artifacts": _binding(artifact_path)},
        "limitations": LIMITATIONS,
    }
    _save_json(objects / "stage0_provenance_summary.json", summary)
    return metadata, artifacts, summary


def angular_grid() -> list[tuple[bool, int, int, int]]:
    """Eight coarse states with unique 2-degree angles within each parity."""
    return [(reflection, sector, residual, sector + residual)
            for reflection in (False, True)
            for sector in (0, 90, 180, 270)
            for residual in range(0, 90, 2)]


def dice_score(left: NDArray[Any], right: NDArray[Any]) -> float:
    """Empty masks are missing evidence, never a perfect match."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    denominator = float(a.sum() + b.sum())
    return float(2.0 * np.minimum(a, b).sum() / denominator) if denominator else 0.0


def _normalize(image: NDArray[Any]) -> FloatImage:
    image = np.asarray(image, dtype=float)
    if not np.isfinite(image).all() or np.any(image < 0):
        raise Stage1PrerequisiteError(
            "Image contains negative/nonfinite values; no silent imputation"
        )
    positive = image[image > 0]
    if positive.size == 0:
        raise Stage1PrerequisiteError("Image contains no positive signal")
    low, high = np.percentile(positive, [1, 99])
    if high <= low:
        raise Stage1PrerequisiteError("Image has no usable positive intensity range")
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def _working_image(image: NDArray[Any]) -> tuple[FloatImage, float]:
    factor = WORK_SIZE / max(image.shape)
    shape = tuple(max(1, round(size * factor)) for size in image.shape)
    resized = transform.resize(
        _normalize(image), shape, order=1, preserve_range=True, anti_aliasing=True,
    )
    # Pixel-center transforms use exact per-axis factors below, not this nominal one.
    return np.asarray(resized, dtype=float), factor


def _foreground(image: FloatImage) -> BoolImage:
    smooth = ndimage.gaussian_filter(image, 1.0)
    return morphology.remove_small_objects(smooth > filters.threshold_otsu(smooth), min_size=8)


def _fully_enclosed_holes(foreground: BoolImage) -> BoolImage:
    """Preserve the former enclosed-only recipe for before/after diagnostics."""
    closed = morphology.closing(foreground, footprint=morphology.disk(2))
    holes = ndimage.binary_fill_holes(closed) & ~closed
    labels, count = ndimage.label(holes)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= 8
    keep[0] = False
    return keep[labels]


def _holes(foreground: BoolImage) -> BoolImage:
    """Retain candidate voids, including edge openings, not anatomical lumens."""
    closed = morphology.closing(foreground, footprint=morphology.disk(2))
    if not np.any(closed):
        return np.zeros_like(closed)
    support = morphology.convex_hull_image(closed)
    candidates = support & ~closed
    labels, count = ndimage.label(candidates)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= 8
    keep[0] = False
    return keep[labels]


def _masks(
    omni: FloatImage, channels: dict[str, FloatImage],
) -> tuple[tuple[BoolImage, BoolImage], tuple[BoolImage, BoolImage]]:
    omni_bright = _foreground(omni)
    omni_holes = _holes(omni_bright)
    omni_proxy = omni_bright & morphology.binary_dilation(
        omni_holes, footprint=morphology.disk(5),
    )
    composite = np.maximum.reduce(list(channels.values()))
    cycif_holes = _holes(_foreground(composite))
    ecad = _foreground(channels["Ecad"])
    masks = ((omni_holes, omni_proxy), (cycif_holes, ecad))
    if any(np.count_nonzero(mask) < 8 for pair in masks for mask in pair):
        raise Stage1PrerequisiteError(
            "Insufficient internal holes/epithelium: orientation cannot use outline fallback"
        )
    return masks


def _moments(mask: BoolImage) -> tuple[FloatImage, float]:
    points = np.argwhere(mask).astype(float)
    if points.size == 0:
        raise Stage1PrerequisiteError("Internal evidence is empty")
    centroid = points.mean(axis=0)
    radius = float(np.sqrt(np.square(points - centroid).sum(axis=1).mean()))
    if radius <= 0:
        raise Stage1PrerequisiteError("Internal evidence has no spatial spread")
    return centroid, radius


def similarity_matrix(
    angle_degrees: float, reflected: bool, scale: float,
    source_center_yx: FloatImage, target_center_yx: FloatImage,
) -> FloatImage:
    """Forward homogeneous YX matrix; positive angle is clockwise on the display.

    Reflection reverses source columns before rotation about its internal-mask
    centroid. Scale maps moving work pixels to fixed work pixels.
    """
    if scale <= 0:
        raise ValueError("Scale must be positive")
    radians = math.radians(angle_degrees)
    rotation = np.asarray(((math.cos(radians), math.sin(radians)),
                           (-math.sin(radians), math.cos(radians))))
    reflection = np.diag((1.0, -1.0 if reflected else 1.0))
    linear = scale * rotation @ reflection
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = target_center_yx - linear @ source_center_yx
    return matrix


def _warp(
    image: NDArray[Any], matrix: FloatImage, shape: tuple[int, int], order: int,
) -> FloatImage:
    inverse = np.linalg.inv(matrix)
    return ndimage.affine_transform(
        np.asarray(image, dtype=float), inverse[:2, :2], inverse[:2, 2],
        output_shape=shape, order=order, mode="constant", cval=0.0, prefilter=False,
    )


def _native_to_work(native_shape: tuple[int, int], work_shape: tuple[int, int]) -> FloatImage:
    factors = np.asarray(work_shape, dtype=float) / np.asarray(native_shape, dtype=float)
    matrix = np.diag((*factors, 1.0))
    matrix[:2, 2] = 0.5 * factors - 0.5
    return matrix


def _xy_affine_to_yx(matrix_xy: NDArray[Any]) -> FloatImage:
    """Convert a forward moving-to-fixed OpenCV XY affine to our forward YX."""
    homogeneous = np.eye(3)
    homogeneous[:2] = matrix_xy
    swap = np.asarray(((0.0, 1.0, 0.0),
                       (1.0, 0.0, 0.0),
                       (0.0, 0.0, 1.0)))
    return swap @ homogeneous @ swap


def _validate_af_registration(
    metadata: list[dict[str, Any]], project_root: Path,
) -> tuple[FloatImage, dict[str, object]]:
    """Align R5Q DAPI to R1 DAPI with reviewed ORB/RANSAC, never tissue outlines.

    This residual within-CyCIF affine is applied to every R5Q AF channel before
    the structural composite. Explicit near-identity and distributed-inlier
    gates reject ambiguous nuclear matches rather than assume a shared origin.
    """
    fixed_info = next(row for row in metadata if row["role"] == "DAPI")
    moving_info = next(
        row for row in metadata if row["role"] == "R5Q_DAPI_validation"
    )
    details: dict[str, object] = {
        "validation_input": moving_info, "fixed_input": fixed_info,
        "registered_af_channel_statuses": [
            row for row in metadata if row["role"] == "R5Q_AF"
        ],
        "purpose": "within-CyCIF input-grid verification, not OMNI Stage 2",
        "helper": _binding(
            project_root / ANALYSIS / "02_scripts/guided_serial_registration.py"
        ),
        "thresholds": {
            "minimum_inliers": 30, "minimum_inlier_span_each_axis": 0.2,
            "maximum_inlier_tre_p95_px": 4.0,
            "singular_value_bounds": [0.95, 1.05],
            "maximum_rotation_degrees": 3.0,
            "maximum_translation_diagonal_fraction": 0.05,
        },
    }
    if not moving_info.get("grid_matches", False):
        raise Stage1PrerequisiteError(
            "R5Q validation DAPI absent or incompatible with the R1 DAPI grid",
            details,
        )
    native_shape = tuple(fixed_info["shape"])
    factor = DISPLAY_SIZE / max(native_shape)
    shape = tuple(max(1, round(size * factor)) for size in native_shape)
    images = []
    for info in (fixed_info, moving_info):
        images.append(transform.resize(
            _normalize(tifffile.imread(Path(info["path"]))),
            shape, order=1, preserve_range=True, anti_aliasing=True,
        ))
    config = RegistrationConfig()
    details["parameters"] = {
        "longest_side": DISPLAY_SIZE, "orb_features": config.orb_features,
        "ratio_test": config.ratio_test,
        "ransac_residual_px": config.ransac_residual_px,
        "ransac_trials": config.ransac_trials, "confidence": 0.995,
        "refine_iterations": 25, "opencv_seed": 20261022,
    }
    cv2.setRNGSeed(20261022)
    try:
        candidate = estimate_affine_orb(images[0], images[1], config)
    except RuntimeError as error:
        details["fit_failure"] = str(error)
        raise Stage1PrerequisiteError(
            "R5Q DAPI residual affine fit failed", details,
        ) from error
    fixed_xy = candidate.match_fixed_xy
    moving_xy = candidate.match_moving_xy
    if fixed_xy is None or moving_xy is None or len(fixed_xy) == 0:
        raise Stage1PrerequisiteError(
            "R5Q DAPI fit returned no inlier pairs", details,
        )
    matrix_xy = np.asarray(candidate.matrix, dtype=float)
    predicted = (
        np.column_stack((moving_xy, np.ones(len(moving_xy)))) @ matrix_xy.T
    )
    residuals = np.linalg.norm(predicted - fixed_xy, axis=1)
    singular = np.linalg.svd(matrix_xy[:, :2], compute_uv=False)
    determinant = float(np.linalg.det(matrix_xy[:, :2]))
    rotation = math.degrees(math.atan2(matrix_xy[1, 0], matrix_xy[0, 0]))
    translation = float(np.linalg.norm(matrix_xy[:, 2]))
    span = np.minimum(
        np.ptp(fixed_xy, axis=0), np.ptp(moving_xy, axis=0),
    ) / np.asarray((shape[1], shape[0]))
    checks = {
        "inlier_count": len(residuals) >= 30,
        "distributed_inliers": bool(np.all(span >= 0.2)),
        "inlier_tre": float(np.percentile(residuals, 95)) <= 4.0,
        "near_unit_scale": bool(
            np.all((singular >= 0.95) & (singular <= 1.05))
        ),
        "positive_determinant": determinant > 0,
        "small_rotation": abs(rotation) <= 3.0,
        "small_translation": translation <= 0.05 * math.hypot(*shape),
    }
    sampling = _native_to_work(native_shape, shape)
    native_matrix = (
        np.linalg.inv(sampling) @ _xy_affine_to_yx(matrix_xy) @ sampling
    )
    details.update({
        "validation_shape_yx": list(shape), "inlier_count": len(residuals),
        "match_fraction_limitation": "reviewed helper returns inlier pairs only",
        "inlier_tre_median_px": float(np.median(residuals)),
        "inlier_tre_p95_px": float(np.percentile(residuals, 95)),
        "inlier_span_fraction_xy": span.tolist(),
        "determinant": determinant, "singular_values": singular.tolist(),
        "rotation_degrees": rotation,
        "translation_validation_px": translation,
        "matrix_validation_xy": matrix_xy.tolist(),
        "matrix_native_yx": native_matrix.tolist(), "checks": checks,
        "direction": "R5Q native YX -> R1 DAPI native YX",
        "status": (
            "accepted_for_AF_input_alignment" if all(checks.values()) else "rejected"
        ),
    })
    if not all(checks.values()):
        raise Stage1PrerequisiteError(
            "R5Q DAPI residual affine failed QC gates", details,
        )
    return native_matrix, details


def _prepare_mask(moving: NDArray[Any]) -> tuple[FloatImage, FloatImage]:
    """Crop to nonzero support once and pad for complete interpolation support."""
    coordinates = np.argwhere(moving > 0)
    if not len(coordinates):
        return np.zeros((1, 1), dtype=float), np.zeros(2)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0) + 1
    cropped = moving[low[0]:high[0], low[1]:high[1]]
    return np.pad(np.asarray(cropped, dtype=float), 1), low.astype(float) - 1


def _complete_mask_raster(
    prepared: tuple[FloatImage, FloatImage], matrix: FloatImage, order: int,
) -> tuple[FloatImage, NDArray[np.int64]]:
    """Rasterize every transformed support pixel, independent of fixed bounds."""
    cropped, source_origin = prepared
    source_shift = np.eye(3)
    source_shift[:2, 2] = source_origin
    forward = matrix @ source_shift
    height, width = cropped.shape
    corners = np.asarray((
        (-0.5, -0.5, 1), (-0.5, width - 0.5, 1),
        (height - 0.5, -0.5, 1), (height - 0.5, width - 0.5, 1),
    ))
    transformed = (corners @ forward.T)[:, :2]
    low = np.floor(transformed.min(axis=0)).astype(np.int64) - 1
    high = np.ceil(transformed.max(axis=0)).astype(np.int64) + 2
    canvas_shift = np.eye(3)
    canvas_shift[:2, 2] = -low
    shape = (int(high[0] - low[0]), int(high[1] - low[1]))
    return _warp(cropped, canvas_shift @ forward, shape, order), low


def _transformed_mask_score(
    moving: NDArray[Any], fixed: NDArray[Any], matrix: FloatImage, order: int,
    *, prepared: tuple[FloatImage, FloatImage] | None = None,
    fixed_mass: float | None = None,
) -> dict[str, float]:
    """Exact discrete Dice on the union, without allocating its zero-only margins.

    Rasterize the entire transformed nonzero crop on an expanded pixel-aligned
    canvas. Its complete sampled mass and the full fixed mass form the Dice
    denominator. Only their geometric intersection contributes to the numerator.
    Clipping is measured from raster pixels outside fixed image bounds, never
    from a continuous-area versus sampled-area discrepancy. Soft fitting and
    nearest-neighbor final scoring use this identical support-accounting rule.
    """
    if prepared is None:
        prepared = _prepare_mask(moving)
    warped, origin = _complete_mask_raster(prepared, matrix, order)
    low = np.maximum(origin, 0)
    high = np.minimum(origin + np.asarray(warped.shape), fixed.shape)
    intersection = 0.0
    visible_area = 0.0
    if np.all(high > low):
        y0, x0 = low - origin
        y1, x1 = high - origin
        visible = warped[y0:y1, x0:x1]
        visible_area = float(visible.sum())
        fixed_region = fixed[low[0]:high[0], low[1]:high[1]]
        intersection = float(np.minimum(visible, fixed_region).sum())
        # Disjoint border strips count actual off-canvas mass directly.
        outside_area = float(
            warped[:y0].sum() + warped[y1:].sum()
            + warped[y0:y1, :x0].sum() + warped[y0:y1, x1:].sum()
        )
    else:
        outside_area = float(warped.sum())
    full_area = visible_area + outside_area
    fixed_area = float(np.asarray(fixed).sum()) if fixed_mass is None else fixed_mass
    denominator = full_area + fixed_area
    clipped_fraction = outside_area / full_area if full_area else 0.0
    return {
        "dice": 2 * intersection / denominator if denominator else 0.0,
        "visible_fraction": 1 - clipped_fraction if full_area else 0.0,
        "clipped_fraction": clipped_fraction,
        "full_transformed_area": full_area, "visible_area": visible_area,
        "outside_fixed_area": outside_area,
    }


def _fit_candidates(
    moving: tuple[BoolImage, BoolImage], fixed: tuple[BoolImage, BoolImage],
) -> pd.DataFrame:
    source_center, source_radius = _moments(moving[0] | moving[1])
    target_center, target_radius = _moments(fixed[0] | fixed[1])
    initial_scale = target_radius / source_radius
    shift_bound = 0.25 * math.hypot(*fixed[0].shape)
    soft_moving = tuple(ndimage.gaussian_filter(mask.astype(float), 1.0) for mask in moving)
    soft_fixed = tuple(ndimage.gaussian_filter(mask.astype(float), 1.0) for mask in fixed)
    prepared_soft = tuple(_prepare_mask(mask) for mask in soft_moving)
    prepared_binary = tuple(_prepare_mask(mask) for mask in moving)
    soft_fixed_masses = tuple(float(mask.sum()) for mask in soft_fixed)
    fixed_masses = tuple(float(mask.sum()) for mask in fixed)
    records: list[dict[str, object]] = []
    for candidate_id, (reflected, sector, residual, angle) in enumerate(angular_grid()):
        def matrix_for(parameters: NDArray[Any]) -> FloatImage:
            return similarity_matrix(
                angle, reflected, float(parameters[0]), source_center,
                target_center + parameters[1:3],
            )

        def objective(parameters: NDArray[Any]) -> float:
            matrix = matrix_for(parameters)
            scores = [
                _transformed_mask_score(
                    a, b, matrix, 1, prepared=crop, fixed_mass=mass,
                )["dice"]
                for a, b, crop, mass in zip(
                    soft_moving, soft_fixed, prepared_soft, soft_fixed_masses,
                )
            ]
            return -float(np.mean(scores))

        initial = np.asarray((initial_scale, 0.0, 0.0))
        bounds = [(0.5 * initial_scale, 2.0 * initial_scale),
                  (-shift_bound, shift_bound), (-shift_bound, shift_bound)]
        fit = optimize.minimize(
            objective, initial, method="Powell", bounds=bounds,
            options=SEARCH["optimizer_options"],
        )
        # A bounded optimizer can finish worse than the evidence-derived start.
        parameters = fit.x if fit.fun <= objective(initial) else initial
        matrix = matrix_for(parameters)
        metric_scores = [
            _transformed_mask_score(
                a, b, matrix, 0, prepared=crop, fixed_mass=mass,
            )
            for a, b, crop, mass in zip(
                moving, fixed, prepared_binary, fixed_masses,
            )
        ]
        lumen, epithelial = (metric["dice"] for metric in metric_scores)
        records.append({
            "candidate_id": candidate_id, "reflected": reflected,
            "score_definition": "exact_rasterized_union_dice",
            "sector_degrees": sector, "residual_degrees": residual,
            "angle_degrees": angle, "scale_work": float(parameters[0]),
            "delta_y_work": float(parameters[1]), "delta_x_work": float(parameters[2]),
            "initial_scale_work": initial_scale, "shift_bound_work": shift_bound,
            "source_centroid_y": float(source_center[0]),
            "source_centroid_x": float(source_center[1]),
            "target_centroid_y": float(target_center[0]),
            "target_centroid_x": float(target_center[1]),
            "lumen_dice": lumen, "epithelial_dice": epithelial,
            **{
                f"{name}_{key}": value
                for name, metric in zip(("lumen", "epithelial"), metric_scores)
                for key, value in metric.items() if key != "dice"
            },
            "mean_score": 0.5 * (lumen + epithelial),
            "optimizer_success": bool(fit.success), "optimizer_message": str(fit.message),
            "optimizer_evaluations": int(fit.nfev),
            "used_initialization": bool(np.array_equal(parameters, initial)),
            "matrix_work_yx": json.dumps(matrix.tolist()),
        })
    result = pd.DataFrame(records).sort_values(
        ["mean_score", "candidate_id"], ascending=[False, True], kind="stable",
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def _require_registered_af(
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Require at least one grid-eligible AF channel before DAPI verification."""
    inspected = [row for row in metadata if row["role"] == "R5Q_AF"]
    usable = [row for row in inspected if row.get("grid_eligible", False)]
    if not usable:
        raise Stage1PrerequisiteError(
            "No usable grid-verified registered R5Q autofluorescence channel",
            {"registered_af_channel_statuses": inspected},
        )
    return usable


def _run_stage1(
    project_root: Path, tables_dir: Path, plots_dir: Path, objects_dir: Path,
) -> tuple[pd.DataFrame, matplotlib.figure.Figure, dict[str, object]]:
    """Run only the image-derived full-angle coarse screen and save review outputs.

    The 360 candidates (180 angles per reflection parity) fit relative scale and
    translation from internal structures, never from core-outline agreement.
    Native-coordinate matrices include downsampling pixel-center conventions.
    No calibrated OMNI scale or accepted biological match is asserted.
    """
    tables, plots, objects = tables_dir, plots_dir, objects_dir
    for directory in (tables, plots, objects):
        directory.mkdir(parents=True, exist_ok=True)
    metadata, source_info = _report_data(project_root)
    if not source_info["registered_grid_matches_manifest"]:
        raise Stage1PrerequisiteError(
            "DAPI/Ecad actual TIFF grids disagree with source OME manifest"
        )
    usable_af = _require_registered_af(metadata)
    af_matrix_native, af_validation = _validate_af_registration(metadata, project_root)
    for info in usable_af:
        info["usable"] = True
        info["origin_verified"] = True
        info["transform_to_dapi"] = af_matrix_native.tolist()
        info["registration_evidence"] = (
            "R5Q c1 DAPI to R1 DAPI ORB/RANSAC residual-affine QC passed; "
            "same per-round affine applied to AF before structural composite"
        )
    omni_info = metadata[0]
    _sum_axis(omni_info)
    omni_stack = tifffile.imread(Path(omni_info["path"]))
    omni_native = _sum_hs(omni_stack, omni_info)
    del omni_stack
    omni, _ = _working_image(omni_native)
    native_omni_shape = omni_native.shape
    native_channels: dict[str, NDArray[Any]] = {}
    channels: dict[str, FloatImage] = {}
    fixed_native_shape: tuple[int, int] | None = None
    used_inputs = [
        omni_info,
        next(row for row in metadata if row["role"] == "R5Q_DAPI_validation"),
    ]
    for info in metadata[1:]:
        role = info["role"]
        if role == "R5Q_DAPI_validation":
            continue
        if role == "R5Q_AF" and not info.get("usable", False):
            continue
        image = tifffile.imread(Path(info["path"]))
        fixed_native_shape = image.shape
        name = role if role != "R5Q_AF" else Path(info["path"]).stem
        channels[name], _ = _working_image(image)
        if role == "R5Q_AF":
            sampling = _native_to_work(image.shape, channels[name].shape)
            matrix_work = sampling @ af_matrix_native @ np.linalg.inv(sampling)
            channels[name] = _warp(
                channels[name], matrix_work, channels["DAPI"].shape, 1,
            )
        if role in {"DAPI", "Ecad"}:
            native_channels[role] = image
        used_inputs.append(info)
    if fixed_native_shape is None:
        raise Stage1PrerequisiteError("Missing registered image references")
    moving_masks, fixed_masks = _masks(omni, channels)
    scores = _fit_candidates(moving_masks, fixed_masks)
    if scores.empty or float(scores["mean_score"].max()) <= 0:
        raise Stage1PrerequisiteError("No candidate has usable internal image overlap")
    source_sampling = _native_to_work(native_omni_shape, omni.shape)
    fixed_shape = channels["DAPI"].shape
    fixed_sampling = _native_to_work(fixed_native_shape, fixed_shape)
    native_matrices = []
    native_translations_yx: list[tuple[float, float]] = []
    for matrix_json in scores["matrix_work_yx"]:
        matrix = (
            np.linalg.inv(fixed_sampling)
            @ np.asarray(json.loads(matrix_json))
            @ source_sampling
        )
        native_matrices.append(json.dumps(matrix.tolist()))
        native_translations_yx.append((float(matrix[0, 2]), float(matrix[1, 2])))
    scores["matrix_native_yx"] = native_matrices
    scores["scale_native_area_equivalent"] = [
        float(np.sqrt(abs(np.linalg.det(np.asarray(json.loads(value))[:2, :2]))))
        for value in native_matrices
    ]
    scores["translation_y_native_px"] = [
        translation[0] for translation in native_translations_yx
    ]
    scores["translation_x_native_px"] = [
        translation[1] for translation in native_translations_yx
    ]
    scores["scale_units"] = "relative CyCIF native pixels per OMNI native pixel"
    score_path = tables / "stage1_orientation_candidate_scores.csv"
    scores.to_csv(score_path, index=False)
    display_factor = DISPLAY_SIZE / max(fixed_native_shape)
    display_shape = tuple(
        max(1, round(size * display_factor)) for size in fixed_native_shape
    )
    display_sampling = _native_to_work(fixed_native_shape, display_shape)
    native_display = {
        role: transform.resize(
            _normalize(image), display_shape, order=1,
            preserve_range=True, anti_aliasing=True,
        )
        for role, image in native_channels.items()
    }
    normalized_native_omni = _normalize(omni_native)
    sum_dtype = str(omni_native.dtype)
    del omni_native, native_channels
    font_path = project_root.parents[1] / "00_environments/share/fonts/arial/Arial.TTF"
    if font_path.is_file():
        matplotlib.font_manager.fontManager.addfont(str(font_path))
    resolved_font = matplotlib.font_manager.findfont(
        matplotlib.font_manager.FontProperties(family="Arial", weight="normal"),
        fallback_to_default=False,
    )
    with plt.rc_context({
        "font.family": "Arial", "font.size": 5, "font.weight": "normal",
        "axes.labelsize": 5, "axes.labelweight": "normal",
        "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
        "axes.linewidth": 0.5, "lines.linewidth": 0.5,
        "grid.linewidth": 0.5, "grid.color": "#E5E7EB",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        figure, axes = plt.subplots(
            3, 3, figsize=(180 / 25.4, 190 / 25.4), squeeze=False,
        )
        for column, row in scores.head(3).iterrows():
            matrix = np.asarray(json.loads(row["matrix_work_yx"]))
            native_matrix = np.asarray(json.loads(row["matrix_native_yx"]))
            warped = _warp(
                normalized_native_omni, display_sampling @ native_matrix,
                display_shape, 1,
            )
            rgb = np.zeros((*display_shape, 3), dtype=np.float32)
            rgb[..., 0] = warped
            rgb[..., 1] = native_display["Ecad"]
            rgb[..., 2] = 0.45 * native_display["DAPI"]
            extent = (
                -0.5, fixed_shape[1] - 0.5,
                fixed_shape[0] - 0.5, -0.5,
            )
            axes[0, column].imshow(
                rgb, interpolation="nearest", extent=extent,
            )
            for mask_row, (moving, fixed) in enumerate(
                zip(moving_masks, fixed_masks), start=1,
            ):
                mask_rgb = np.zeros((*fixed_shape, 3), dtype=np.float32)
                complete, origin = _complete_mask_raster(
                    _prepare_mask(moving), matrix, 0,
                )
                low = np.maximum(origin, 0)
                high = np.minimum(origin + complete.shape, fixed_shape)
                if np.all(high > low):
                    y0, x0 = low - origin
                    y1, x1 = high - origin
                    mask_rgb[low[0]:high[0], low[1]:high[1], 0] = (
                        complete[y0:y1, x0:x1]
                    )
                mask_rgb[..., 1] = fixed
                axes[mask_row, column].imshow(
                    mask_rgb, interpolation="nearest",
                )
            axes[0, column].set_xlabel(
                f"Rank {row['rank']}: {row['angle_degrees']}°, flip={row['reflected']}\n"
                f"lumen={row['lumen_dice']:.3f}; epithelium={row['epithelial_dice']:.3f}"
            )
            axes[1, column].set_xlabel(
                "Internal holes: OMNI red; CyCIF green\n"
                f"Raster outside canvas: {row['lumen_clipped_fraction']:.1%}"
            )
            axes[2, column].set_xlabel(
                "Epithelium: OMNI proxy red; CyCIF Ecad green\n"
                f"Raster outside canvas: {row['epithelial_clipped_fraction']:.1%}"
            )
        for axis in axes.ravel():
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlim(-0.5, fixed_shape[1] - 0.5)
            axis.set_ylim(fixed_shape[0] - 0.5, -0.5)
            axis.set_axisbelow(True)
            axis.grid(True, which="major", color="#E5E7EB", linewidth=0.5)
        generator = _binding(Path(__file__))
        figure.text(0.02, 0.065,
                    "Coarse hypotheses only. Top: native-source OMNI red, "
                    "Ecad green, DAPI blue.",
                    fontsize=5)
        figure.text(0.02, 0.047,
                    f"Generator: {ANALYSIS / '02_scripts' / Path(__file__).name}",
                    fontsize=5)
        figure.text(0.02, 0.029,
                    f"Script SHA256: {generator['sha256']}", fontsize=5)
        figure.text(0.02, 0.011,
                    f"OMNI input SHA256: {omni_info['sha256']}", fontsize=5)
        figure.subplots_adjust(left=0.03, right=0.98, top=0.98,
                               bottom=0.15, hspace=0.22)
        pdf_path = plots / "stage1_top3_full_core_overlays.pdf"
        figure.savefig(pdf_path, dpi=600)
    receipt: dict[str, object] = {
        "schema_version": 1, "stage": "image-only Stage 1", "claim_label": "Exploratory",
        "context_source_packets": CONTEXT_SOURCE_PACKETS,
        "status": "complete_coarse_screen_review_required",
        "font": _binding(Path(resolved_font)),
        "generator": _binding(Path(__file__)), "input_bindings": used_inputs,
        "inspected_images_including_exclusions": metadata, **source_info,
        "mask_recipe": MASK_RECIPE, "search": SEARCH,
        "registered_af_channels_used": [row["path"] for row in usable_af],
        "registered_af_channel_statuses": [
            row for row in metadata if row["role"] == "R5Q_AF"
        ],
        "omni_sum_accumulation": {
            "source_reduction_matches_original": True,
            "operation": "np.sum(stack, axis=0); NumPy default accumulation",
            "input_dtype": omni_info["dtype"], "output_dtype": sum_dtype,
        },
        "overlay_rendering": {
            "intensity_source": "native OMNI sum and native registered DAPI/Ecad",
            "native_transform": "display_sampling @ saved matrix_native_yx",
            "intensity_canvas_yx": list(display_shape),
            "intensity_resampling": "native-source bilinear; reference anti-alias resize",
            "mask_canvas_yx": list(fixed_shape),
            "mask_rows": ["internal holes", "OMNI gland proxy vs CyCIF Ecad"],
            "pdf_canvas_mm": [180, 190], "export_dpi": 600,
        },
        "registered_af_dapi_validation": af_validation,
        "score_support_accounting": (
            "exact rasterized-union Dice: complete transformed moving raster mass "
            "plus full fixed mass in denominator; clipping is the transformed "
            "raster mass geometrically outside fixed bounds, not sampling loss"
        ),
        "candidate_count": len(scores), "coarse_state_count": 8,
        "mask_pixel_counts": {
            "omni_holes": int(moving_masks[0].sum()), "omni_proxy": int(moving_masks[1].sum()),
            "cycif_holes": int(fixed_masks[0].sum()), "cycif_ecad": int(fixed_masks[1].sum()),
        },
        "working_shapes_yx": {"omni": list(omni.shape), "cycif": list(fixed_shape)},
        "native_shapes_yx": {"omni": list(native_omni_shape), "cycif": list(fixed_native_shape)},
        "transform_convention": {
            "vectors": "homogeneous [row=y, column=x, 1]; integer pixel centers",
            "forward": "CyCIF_native_yx = matrix_native_yx @ OMNI_native_yx",
            "operation": (
                "column reflection about internal centroid, clockwise "
                "rotation, scale, translation"
            ),
            "resampling": "work=(native+0.5)*actual_axis_factor-0.5; zero exterior",
            "scoring": "fixed complete CyCIF work canvas; nearest neighbor for binary final masks",
            "scale": (
                "relative pixels only; native area-equivalent scale from "
                "sqrt(abs(det(linear)))"
            ),
            "native_translation": (
                "translation_y_native_px and translation_x_native_px are "
                "matrix_native_yx[0,2] and [1,2], in target CyCIF native pixels; "
                "these are affine offsets at the source origin, not centroid deltas"
            ),
            "calibrated_omni_scale_claimed": False,
        },
        "top3": scores.head(3).to_dict(orient="records"),
        "outputs": {"scores": _binding(score_path), "overlays": _binding(pdf_path)},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "tifffile": tifffile.__version__,
                    "matplotlib": matplotlib.__version__, "skimage": skimage.__version__,
                    "opencv": cv2.__version__,
                    "pandas": pd.__version__},
        "limitations": LIMITATIONS, "reconstructed_labels_used": False,
        "biological_registration_validated": False,
    }
    _save_json(objects / "stage1_orientation_summary.json", receipt)
    return scores, figure, receipt


def run_stage1(
    project_root: Path, tables_dir: Path, plots_dir: Path, objects_dir: Path,
) -> tuple[pd.DataFrame, matplotlib.figure.Figure | None, dict[str, object]]:
    """Run the screen, or save an honest prerequisite stop with no fake scores.

    Only expected missing files, invalid TIFF containers and explicit scientific
    prerequisite errors become a stopped receipt. Programming errors propagate.
    A stopped run returns an empty table and no figure, never historical outputs.
    """
    try:
        return _run_stage1(project_root, tables_dir, plots_dir, objects_dir)
    except (Stage1PrerequisiteError, FileNotFoundError, tifffile.TiffFileError) as error:
        for directory in (tables_dir, plots_dir, objects_dir):
            directory.mkdir(parents=True, exist_ok=True)
        manifest, references, af_paths = _paths(project_root)
        inputs = [project_root / SOURCE / "Original/C2Image2048.tif",
                  manifest, *references.values()]
        # Do not touch shared AF sources when required local inputs are absent.
        if all(path.is_file() for path in inputs):
            inputs.extend([*af_paths, AF_DAPI])
        receipt: dict[str, object] = {
            "schema_version": 1, "stage": "image-only Stage 1",
            "context_source_packets": CONTEXT_SOURCE_PACKETS,
            "status": "stopped_missing_image_prerequisite",
            "reason": str(error), "error_type": type(error).__name__,
            "prerequisite_details": (
                error.details if isinstance(error, Stage1PrerequisiteError) else {}
            ),
            "generator": _binding(Path(__file__)),
            "existing_input_bindings": [_binding(path) for path in inputs if path.is_file()],
            "missing_inputs": [str(path) for path in inputs if not path.is_file()],
            "mask_recipe": MASK_RECIPE, "search": SEARCH,
            "candidate_count": 0, "outputs": {},
            "limitations": LIMITATIONS, "reconstructed_labels_used": False,
            "biological_registration_validated": False,
        }
        _save_json(objects_dir / "stage1_orientation_summary.json", receipt)
        return pd.DataFrame(), None, receipt
