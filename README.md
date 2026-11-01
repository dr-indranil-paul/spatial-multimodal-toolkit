# Spatial Omics Toolkit

A guided app for exploring spatial imaging and omics data. Work independently in each modality workspace, inspect quality control, and review that modality’s outputs. Integration is optional.

The app provides workflows for Cyclic immunofluorescence (CycIF), hyperspectral imaging (HS), fluorescence lifetime imaging microscopy (FLIM), hematoxylin and eosin (H&E), and Visium. Integration currently offers an **exploratory serial-section alignment preview**; it does not establish biological pairing or validate an alignment.

## Start here

Choose a modality from Home. If an **Included example** is available, run it to see the workflow and its outputs; no project setup is needed. Examples demonstrate technical QC and are not analyses of your study cohort. The app describes the expected inputs and outputs before you start.

To use your own data, select **Use my own data** and follow **Data & settings**:

1. Choose the study folder, that modality’s input files, and an output folder.
2. If the analysis needs project sample information, choose the reviewed `project_samples.csv` or download the template and fill it in from study records.
3. Choose the data layout: one sample, multiple samples, or a tissue microarray (TMA).
4. Select **Run**, choose **Check inputs**, and resolve any requests for mapping or review before starting the run.
5. Follow progress in **Run**; open **Quality control** to review warnings, **Results** to explore plots and reports, and **Files** to access outputs. To continue a verified CycIF run, choose **Continue from a verified run** in its settings. The app creates a new run and reuses completed stages only when their inputs and settings still match.

Source files are read-only. Results are written to a new run folder and do not overwrite earlier runs. Each modality keeps its own settings and run history.

## Sample information

When needed, the project sample sheet has one row per physical specimen or section. It requires these columns:

```csv
biological_unit_id,specimen_id,condition,tissue_source
```

`specimen_id` identifies the physical specimen or section. `biological_unit_id` identifies the independent experimental unit, such as a participant or animal, and may repeat across that unit’s specimens or sections. Use `unknown` when information is unavailable; do not infer identities from filenames.

For TMA workflows, the optional core-map CSV links each detected `group_key` to a reviewed `specimen_id`. You can download a map populated with the detected group keys after **Check inputs**. Several cores may belong to the same biological unit. A whole-slide image is not automatically divided into cores; per-core analysis requires core-indexed inputs.

An acquisition group left `UNMAPPED` can be used for technical QC only. The app requests confirmation before using mapped groups for analysis.

## Modality workflows

| Workspace | Typical starting input | What to expect |
| --- | --- | --- |
| **CycIF** | Raw CZI images, or a cell-feature table for tabular analysis | Run intake and configure conversion, round registration, marker export, and MCMICRO segmentation as needed. Some raw-image inputs lack OME metadata; intake reports unavailable checks rather than treating them as passes. |
| **Hyperspectral imaging (HS)** | Native spectral CZI or reviewed study files and processing records | Inspect spectral bands, metadata, and QC outputs. The study workflow requires its reviewed manifest, core-selection ledger, assignment and processing records, and settings. |
| **FLIM** | Native `.sdt` measurement files | Review native blocks and acquisition information. Raw-count and decay QC is distinct from calibrated lifetime analysis. |
| **H&E** | SVS, CZI, TIFF, or OME-TIFF images | Inspect image metadata and view bounded previews. Image intake does not perform cross-modality registration. |
| **Visium** | A supported Space Ranger output bundle | Inspect the gene-expression and spatial data at a supported resolution. Spatial bins are not biological replicates. |
| **Integration** | An explicit moving H&E image and fluorescence reference | Inspect an exploratory serial-section translation preview. Confirming candidate files does not confirm biological pairing or accept the alignment. |
