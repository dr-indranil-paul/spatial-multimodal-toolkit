# Spatial Omics Toolkit

A guided app for exploring spatial imaging and omics data. Work independently in each modality workspace, inspect quality control, and review that modality’s outputs. Integration is optional.

The app provides workflows for Cyclic immunofluorescence (CycIF), hyperspectral imaging (HS), fluorescence lifetime imaging microscopy (FLIM), hematoxylin and eosin (H&E), and Visium. Integration currently offers an **exploratory serial-section alignment preview**; it does not establish biological pairing or validate an alignment.

## Deployment modes

The same app supports full HPC use and a read-only Streamlit Community Cloud view. Set this in the Community Cloud **App settings → Secrets**:

```toml
[toolkit]
mode = "read_only"
```

`datasets_url` is optional; when supplied, the viewer shows it as an external Google Drive link. The URL is intentionally supplied through deployment secrets, not committed to GitHub. Drive permissions remain separate: give collaborators access to the folder as needed. The app only links to these files; it does not copy or process them in Community Cloud. For analysis, download permitted datasets to the authorized HPC workspace and run the full app there.

An app launched outside an SLURM allocation is view-only by default. The full workspace requires an allocated HPC job and, when toolkit settings are configured, `mode = "hpc"`. Community Cloud's free tier is reported here as limited to 1 GB of RAM; this viewer mode avoids dataset processing and job submission. Streamlit Community Cloud supports private apps and invited viewers, so the app and GitHub repository can remain private. See [app sharing and privacy](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app). Do not place identifiable or otherwise restricted study data in a personal Google account unless institutional policy approves it.

Deploy `apps/streamlit/app.py` from the private GitHub repository and keep the Community Cloud app private. Invite the collaborator as a Cloud viewer; give separate GitHub repository access only if they need to use or inspect the code. Setting `datasets_url` under `[toolkit]` in Cloud secrets adds a Google Drive link on Home when the sharing URL is available. The app does not mount Drive or persist downloaded files. Check that the Drive folder's access permissions allow only intended users.

## Start here

Choose a workspace from the cards in the center of Home, including optional Integration. Use **Open project** to set up a study folder, sample sheet, and results location. Inside a workspace, use the top navigation to return **Home** or open **Project**.

A modality workspace opens on **Introduction**, which describes the selected workflow's purpose, detailed input and output bullets, shared own-data prerequisites, and limitations. Its stage-and-tool diagram is an overview of the declared workflow, not a record of a particular run: notes identify conditional pathways, so not every stage or tool necessarily applies to every input or configuration. Outputs depend on the selected stages and successful execution; passing input checks does not guarantee them. Expand **Methods & source** for implementation-level detail; it shows current source definitions, which may differ from those used by an earlier saved run. This page is informational: it does not check inputs, start a run, or change results. Use the remaining workspace tabs to continue:

- **Data & settings** to choose example or own-data inputs and configure the selected workflow.
- **Run** to check inputs, review requested decisions, and start or monitor work.
- **Quality control**, **Results**, and **Files** to inspect the selected run's recorded review information and outputs.

If an **Included example** is available, choose it in the input-mode selector and run it from **Data & settings** or **Run**; no project setup is needed. Examples demonstrate technical QC and are not analyses of your study cohort.

To use your own data, select **Use my own data** and follow **Data & settings**:

1. Choose existing project and modality folders under the approved workspace, and an output folder under that workspace, outside the modality folder. The modality folder may be outside the project folder.
2. Select a reviewed `project_samples.csv` under the project folder. It is required for every submitted own-data run, including technical QC.
3. Choose the data layout and the workflow-specific inputs and settings; specify SLURM resources (1–32 CPUs, 1–512 GB memory, and an 8h, 24h, 48h, 72h, or 168h time limit).
4. Select **Run**, choose **Check inputs**, and review the discovered groups. Where groups are returned, confirm an exact entry for every displayed key: a sample-sheet specimen ID or explicit `UNMAPPED` for technical QC. H&E intake returns no groups and needs no image mapping. The cell-feature table requires actual specimen IDs, either by confirmed core mapping or a confirmed existing specimen-ID column. Integration requires confirmed entries for both image keys; its report remains unmapped and does not establish biological pairing. Clinical HS stops after metadata QC if any mapping is `UNMAPPED`.
5. Start the run and follow progress in **Run**; open **Quality control** to review warnings, **Results** to explore plots and reports, and **Files** to access outputs. To continue a verified CycIF run, choose **Continue from a verified run** in its settings. The app creates a new run and reuses completed stages only when their inputs and settings still match.

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
