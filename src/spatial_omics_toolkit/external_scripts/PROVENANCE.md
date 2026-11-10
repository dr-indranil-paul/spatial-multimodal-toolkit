# Vendored external scripts: provenance

These 15 files are byte-identical copies of the FLIM and hyperspectral source
scripts used by the `flim.raw_sdt` and `hs.native_spectral_qc` workflows. They
are the closure of modules loaded by path, or imported from the source
directory, by the two adapters (`clinical_roi_flim.py`, `clinical_roi_flim_plots.py`,
`clinical_roi_hs_flim.py`, `clinical_roi_hyperspectral.py`), including lazy
imports inside functions. `plotnine_style_helper.py` lives in the original
`01_utils/` sibling directory; it is vendored here flat, next to the scripts
that import it, so the adapters' `sys.path` insertion of this directory makes it
importable.

The toolkit copy is the one other labs run. Changes to the original scripts
are NOT propagated automatically. Any change must be synced deliberately: edit
the original (or this copy), copy the file, re-run `cmp`, and update the hashes
and date in the table below. `tests/unit/test_external_scripts.py` fails when a
copy no longer matches the SHA-256 recorded here.

Override: set `SPATIAL_TOOLKIT_FLIM_SOURCE_DIR` or `SPATIAL_TOOLKIT_HS_SOURCE_DIR`
to use a different directory. Those overrides take precedence over this copy.

Known limitations of the byte-identical copies (not fixed here by design):

- `cycif_zeb1_segment_quantify.py` resolves the toolkit source as
  `PROJECT_ROOT.parent / "02_spatial_toolkit" / "src"` at import time and raises
  `RuntimeError` if that is absent. It is imported lazily (Backsub path of
  `clinical_roi_cycif.py`), so the FLIM and native HS stages do not reach it.
- `clinical_roi_cycif.py::_toolkit_registration` and `clinical_roi_valis.py`
  reference fixed workspace paths (`00_environments`, `02_spatial_toolkit`);
  `guided_serial_registration.py` and `dylan_c2_image_orientation.py` contain
  workspace-specific data or binary paths. These are used only in functions
  outside the FLIM and native HS stages.
- `dylan_k9_reconstruction.py` and `dylan_c2_image_orientation.py` read project
  data files relative to a project root; they are not vendored data.

| File | Original absolute path | Size (bytes) | Original SHA-256 | Copy SHA-256 | Copy date |
| `clinical_roi_flim.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/clinical_roi_flim.py` | 41763 | `10c128fe694aaae99490e9f2a9e079f69d34395d3dd01aa382aa297326a9e1db` | `10c128fe694aaae99490e9f2a9e079f69d34395d3dd01aa382aa297326a9e1db` | 2026-11-10 |
| `clinical_roi_flim_plots.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/clinical_roi_flim_plots.py` | 28257 | `f25c37a6f14d874859b0b8644f6be0713cff42d086e87869a2d905d8c9975bc8` | `f25c37a6f14d874859b0b8644f6be0713cff42d086e87869a2d905d8c9975bc8` | 2026-11-10 |
| `clinical_roi_hs_flim.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/clinical_roi_hs_flim.py` | 70569 | `f310e04a9a9ed86a642c9223bf7c612abe6cc5a350f4bc18e8437ccee0db74d8` | `f310e04a9a9ed86a642c9223bf7c612abe6cc5a350f4bc18e8437ccee0db74d8` | 2026-11-10 |
| `clinical_roi_hyperspectral.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/clinical_roi_hyperspectral.py` | 323952 | `30198b145c9fc52ec77c583c096e6c939ca2e217a0a9e2d7fef4008e376843f6` | `30198b145c9fc52ec77c583c096e6c939ca2e217a0a9e2d7fef4008e376843f6` | 2026-11-10 |
| `dylan_k9_reconstruction.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/dylan_k9_reconstruction.py` | 43954 | `c6124d959bce612b992881228b9c05e9f3f1c0966ed9771568c585f821709bd6` | `c6124d959bce612b992881228b9c05e9f3f1c0966ed9771568c585f821709bd6` | 2026-11-10 |
| `clinical_roi_valis.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/clinical_roi_valis.py` | 94384 | `fc8837d4102159d102f71ee1007d7de7fe963b62ec6bcb6bdcec2b9762fcad46` | `fc8837d4102159d102f71ee1007d7de7fe963b62ec6bcb6bdcec2b9762fcad46` | 2026-11-10 |
| `clinical_roi_cycif.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/clinical_roi_cycif.py` | 59836 | `16f411cea433fcc69d65adfcec327383371ea0ff99580c9d5c85500fc8db026d` | `16f411cea433fcc69d65adfcec327383371ea0ff99580c9d5c85500fc8db026d` | 2026-11-10 |
| `clinical_roi_tissue.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/clinical_roi_tissue.py` | 15648 | `7c4d440b2d6e24558c4ed9485ae2de6c9bc51528cc2f13d97354af0415084743` | `7c4d440b2d6e24558c4ed9485ae2de6c9bc51528cc2f13d97354af0415084743` | 2026-11-10 |
| `clinical_roi_pattern_recovery.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/clinical_roi_pattern_recovery.py` | 118905 | `9f5d2ca81d8b19fc73828ef76aac92c692346862fdd3ad529ef0b0311021a029` | `9f5d2ca81d8b19fc73828ef76aac92c692346862fdd3ad529ef0b0311021a029` | 2026-11-10 |
| `image_previews.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/image_previews.py` | 26794 | `1f0caebdf43f9fd3885f57cd18f3ac1d97afcb920d8177e5f3f6cfaa17b2dd9d` | `1f0caebdf43f9fd3885f57cd18f3ac1d97afcb920d8177e5f3f6cfaa17b2dd9d` | 2026-11-10 |
| `guided_serial_registration.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/guided_serial_registration.py` | 21498 | `6d9674ef6af29bf2d5b34df3a487563ec663a5801fc873b972ac093622e8059e` | `6d9674ef6af29bf2d5b34df3a487563ec663a5801fc873b972ac093622e8059e` | 2026-11-10 |
| `dylan_c2_image_orientation.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/dylan_c2_image_orientation.py` | 57497 | `051536e9e23825b0a95d687c3df2ead296346b39ef0bee2cf88c11face453e9a` | `051536e9e23825b0a95d687c3df2ead296346b39ef0bee2cf88c11face453e9a` | 2026-11-10 |
| `valis_supplied_transform.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/valis_supplied_transform.py` | 3735 | `c81c8c6f1c725368857327b6f7e1e169e0719faaa70ba62a04fad111c44c0436` | `c81c8c6f1c725368857327b6f7e1e169e0719faaa70ba62a04fad111c44c0436` | 2026-11-10 |
| `cycif_zeb1_segment_quantify.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/02_scripts/cycif_zeb1_segment_quantify.py` | 27750 | `414359bd62d2a6579b294e15fcc855282345fa8d79bee63c00f8968175df7432` | `414359bd62d2a6579b294e15fcc855282345fa8d79bee63c00f8968175df7432` | 2026-11-10 |
| `plotnine_style_helper.py` | `/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/01_exploratory_analysis/01_utils/plotnine_style_helper.py` | 8175 | `2cca53ac7a452d363e5db6f9fd916f73ef3ce1741e022d489f84d490bbeb6ebe` | `2cca53ac7a452d363e5db6f9fd916f73ef3ce1741e022d489f84d490bbeb6ebe` | 2026-11-10 |

Hash method: `sha256sum` over the complete file. The copy hash equals the
original hash for every row because each copy was verified with `cmp` against
its original before this table was written.
