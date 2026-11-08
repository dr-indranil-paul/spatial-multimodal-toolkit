# THIS TOOL IS CURRENTLY IN ACTIVE DEVELOPMENT AND IS NOT INTENDED FOR PRODUCTION USE.


# Spatial Omics Toolkit

A guided app for exploring spatial imaging and omics data. Work independently in each modality workspace, inspect quality control, and review that modality’s outputs. Integration is optional.

The app provides workflows for Cyclic immunofluorescence (CycIF), hyperspectral imaging (HS), fluorescence lifetime imaging microscopy (FLIM), hematoxylin and eosin (H&E), and Visium. Integration currently offers an **exploratory serial-section alignment preview**; it does not establish biological pairing or validate an alignment.

## Deployment modes

The same full interface is available on Streamlit Community Cloud and on HPC, regardless of whether `SLURM_JOB_ID` is set. Home, Project, workflow setup, Run, Quality control, Results, and Files are all available by default; `[toolkit] mode = "hpc"` also selects the full interface. To deliberately restrict an instance to workflow introductions, set `[toolkit] mode = "read_only"` in its secrets. Invalid mode settings are reported and do not silently hide the workspace.

The full interface does not make the Cloud server an HPC cluster. Workflow submission requires `sbatch` on the app server and an authorized, functioning SLURM backend; the app reports when `sbatch` is missing and submission errors are surfaced. Inputs and outputs must be accessible to that server and satisfy the approved-workspace restrictions. Lab HPC paths are not automatically accessible from Community Cloud, and Cloud resource and storage limits still apply. For analysis, use an authorized HPC deployment with the required data, tools, and storage.

Deploy `apps/streamlit/app.py` from the private GitHub repository and keep the Community Cloud app private. Invite the collaborator as a Cloud viewer; give separate GitHub repository access only if they need to use or inspect the code. An optional `datasets_url` under `[toolkit]` in Cloud **App settings → Secrets** adds an external Google Drive link on Home; the app does not mount Drive or automatically copy its files. Drive permissions remain separate: check that the folder is accessible only to intended users. See [app sharing and privacy](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app). Do not place identifiable or otherwise restricted study data in a personal Google account unless institutional policy approves it.

## Start here

Use the left sidebar to open **Home**, **Project**, or a modality workbench; **Multimodal integration** opens the integration workspace. Home also provides direct modality cards. Use **Project** to set up a study folder, sample sheet, and results location with live sample-sheet validation beside the controls.

A modality workbench opens on **Introduction** by default, with the existing summary banner and horizontal section bar. Your selected section is retained when you return to a workspace. Use the section bar to switch views:

- **Introduction** to inspect the annotated workflow diagram, input and output bullets, shared own-data prerequisites, limitations, and **Methods & source** implementation locations. Diagram connections come from the same stage dependencies used for Nextflow execution. Scroll the diagram and expand its accessible details to inspect tools, conditions, inputs, outputs, and review gates. These are declared paths, not a record of a particular run: optional stages do not always run, and successful input checks do not guarantee outputs.
- **Data & settings** to choose example or own-data inputs and configure the selected workflow in the existing side-by-side split pane, including reusable **Settings profiles**.
- **Run** to check inputs, review requested decisions, and start or monitor work with a live execution monitor beside the preflight gate.
- **Quality control**, **Results**, and **Files** to inspect the selected run's recorded review information and outputs.

If a prepared **Included example** is available for the modality, it is selected initially and can be run from **Data & settings** or **Run**; no project setup is needed. When the file-backed example catalog is unavailable (as it may be on Community Cloud), the workspace instead opens with **Use my own data**, shows a notice, and retains all workflow sections. An explicitly selected input mode is preserved. Ask the toolkit administrator to prepare the example catalog if needed; the app does not substitute placeholder data. Examples demonstrate technical QC and are not analyses of your study cohort.

To use your own data, select **Use my own data** and follow **Data & settings**:

1. Choose existing project and modality folders under the approved workspace, and an output folder under that workspace, outside the modality folder. The modality folder may be outside the project folder.
2. Select a reviewed `project_samples.csv` under the project folder. It is required for every submitted own-data run, including technical QC.
3. Choose the data layout and the workflow-specific inputs and settings; specify SLURM resources (1–32 CPUs, 1–512 GB memory, and an 8h, 24h, 48h, 72h, or 168h time limit).
4. Select **Run**, choose **Check inputs**, and review the discovered groups. Where groups are returned, confirm an exact entry for every displayed key: a sample-sheet specimen ID or explicit `UNMAPPED` for technical QC. H&E intake returns no groups and needs no image mapping. The cell-feature table requires actual specimen IDs, either by confirmed core mapping or a confirmed existing specimen-ID column. Integration requires confirmed entries for both image keys; its report remains unmapped and does not establish biological pairing. Clinical HS stops after metadata QC if any mapping is `UNMAPPED`.
5. Start the run and follow progress in **Run**; open **Quality control** to review warnings, **Results** to explore plots and reports, and **Files** to access outputs. To continue a verified CycIF run, choose **Continue from a verified run** in its settings. The app creates a new run and reuses completed stages only when their inputs and settings still match.

Source files are read-only. Results are written to a new run folder and do not overwrite earlier runs. Each modality keeps its own settings and run history.

## Reusable settings

Open **Settings profiles** in the parameter pane of **Data & settings**. The app displays a read-only default TOML file generated from the workflow's parameter definitions. TOML is plain text with comments, named settings, and typed values.

1. Create a named user file from the defaults, or import a TOML file into the editor.
2. Edit the settings and save the user file. Files are stored in `.spatial_toolkit/settings/` beneath the selected project.
3. Select the saved file and choose **Apply saved settings**. Unspecified settings use the shipped defaults; they do not inherit another profile's overrides. The workflow controls show the applied values.
4. Review the controls, then run **Check inputs** again. Applying or resetting a profile invalidates earlier checks and clears review confirmations. Saving a file alone does not apply it.

Defaults cannot be overwritten through profile storage. A profile does not confirm specimen mappings, authorize an example, or approve a scientific review. Included examples retain their fixed technical-QC recipes. Each submitted run freezes its source and effective settings, hashes, and referenced configuration snapshots; changed configuration files block execution.

## Plot customization and PDF export

In **Results**, select a saved table or single-plane image and open **Customize and export a descriptive PDF**. Load the source, choose the appropriate view, and adjust axes, labels, scales, limits, colors, categorical legends, markers, lines, opacity, grid, font, and physical canvas size. Scalar images also support colormap, color limits, and a colorbar.

Preview the settings before generating the PDF. Download the PDF and its provenance JSON; export settings can be downloaded and reused. Exports have unique names under the selected run's `plot_exports/` folder and never replace source files or earlier exports. Text remains editable in the PDF. The default publication style is Arial, 5-point text, and 0.5-point lines; these are plot defaults, separate from the app's interface typography.

These are descriptive views of recorded values, not new statistical analyses or reconstructions of existing PDF figures. The app reports unsupported, oversized, missing, or nonfinite data rather than inventing values or silently substituting a downsampled export. An image artifact may itself be a recorded preview.

## Workflow execution

Every registered workflow runs its declared stages through Nextflow inside its allocated SLURM job. Runs retain the generated workflow, stage receipts, execution trace, and report. Dependencies and review gates prevent downstream work from being reported as completed when a prerequisite fails or needs review. Some scientific operations remain a single process where their canonical implementation requires shared in-memory state.

Nextflow manages execution; it does not guarantee success, scientific validity, or accepted alignment. General automatic cache reuse is disabled. Existing reviewed CycIF child-run continuation and FLIM review paths retain their input, configuration, and provenance checks.


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
| **Multimodal integration** | An explicit moving H&E image and fluorescence reference | Inspect an exploratory serial-section translation preview. Confirming candidate files does not confirm biological pairing or accept the alignment. |
