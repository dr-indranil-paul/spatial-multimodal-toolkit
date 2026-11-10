# Collaborator setup (another lab on the same HPC)

This guide is for a researcher in a lab other than the repository owner's, working on the university HPC. It covers access, a private clone in your own workspace, building your environment, configuring the site variables, and running the app on loopback.

The toolkit is in active development and is not intended for production use. Read the limitations in section 8 before relying on any output.

Placeholders in angle brackets (for example `<your-lab-workspace>`, `<owner>`, `<port>`) must be replaced with your own values. Paths shown under `/home/groups/EmiliLab/...` are the Emili lab defaults, given as examples only.

## 1. Request access

The repository is private. The owner invites collaborators by GitHub username. Send the owner your GitHub username and ask for read access, or write access to push topic branches. Accept the invitation from GitHub before cloning.

## 2. Clone into your own lab workspace

Clone into a directory that your lab controls. Do not clone into another lab's directory, and do not rely on another lab's data or environment.

```bash
cd <your-lab-workspace>
git clone git@github.com:<owner>/<repo>.git spatial-toolkit
cd spatial-toolkit
```

Each lab keeps its own clone, environment, and data. The GitHub repository is the only channel shared between labs.

## 3. Branches and pull requests

- Never push to `main`. All changes reach `main` through pull requests.
- Use one short-lived branch per task, named `<name>-<topic>`. The worked example is `yhc-cycif-registration`:

  ```bash
  git fetch origin
  git switch -c <name>-<topic> origin/main
  # ...edit, commit...
  git push -u origin <name>-<topic>
  ```

- Open a pull request from `<name>-<topic>` into `main` on GitHub.
- Delete the branch after its pull request is merged.
- A branch created before the portability changes are merged must run `git fetch origin && git merge origin/main` after that pull request lands, so it picks up the site-configuration and environment changes.
- Before starting new work, update your local `main` with `git switch main && git pull`.
- Branch protection on `main` and code-owner review depend on the owner's GitHub plan and settings. See [CONTRIBUTING.md](../CONTRIBUTING.md#repository-rules-owner).

## 4. Build your environment

Environment definitions live in `environment/`. Read `environment/README.md` first, then run:

```bash
bash environment/setup.sh
```

Run this only on a SLURM compute allocation (see section 6), never on a login node. Keep installs and scratch files under your own workspace.

Prerequisites, as pinned in `environment/runtime.lock.sh`:

- Java 21
- Nextflow 24.10.6
- Apptainer (used to run MCMICRO containers listed in `environment/mcmicro-containers.tsv`)

These are required for CycIF segmentation. Other modalities (HS, FLIM, H&E, Visium) do not need Nextflow-based MCMICRO segmentation; check the workflow's preflight messages for any missing tools.

## 5. Set the site variables

The app reads its site configuration from environment variables on every call. Set them in your shell, or in the site environment script they point to. Defaults reproduce the Emili lab setup and are not portable.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SPATIAL_TOOLKIT_WORKSPACE_ROOTS` | `/home/groups/EmiliLab/pauli` | `os.pathsep`-separated list of approved workspace roots. A path is approved if it resolves under any root. Set this to your lab workspace. |
| `SPATIAL_TOOLKIT_PYTHON` | `/home/groups/EmiliLab/pauli/00_environments/bin/spatial-python` | Interpreter used to run the worker, controller, and Nextflow in job scripts. Point it at the interpreter that `setup.sh` produces. |
| `SPATIAL_TOOLKIT_ENV_SCRIPT` | `/home/groups/EmiliLab/pauli/00_environments/bin/spatial_env.sh` | Shell script sourced at the top of every job script (modules, Java, Nextflow). Set to an empty string to source nothing. |
| `SPATIAL_TOOLKIT_EXTRA_PYTHONPATH` | Two Emili overlay directories under `00_environments/` | `os.pathsep`-separated extra `PYTHONPATH` entries for jobs. The toolkit `src` directory is always derived from the installed package, not from this variable. |
| `SPATIAL_TOOLKIT_SLURM_ACCOUNT` | unset | When set, `sbatch` receives `--account=<value>`. When unset, no flag is passed. |
| `SPATIAL_TOOLKIT_SLURM_PARTITION` | unset | When set, `sbatch` receives `--partition=<value>`. When unset, no flag is passed. |
| `SPATIAL_TOOLKIT_EXAMPLES` | `<repo>/fixtures/example_data` | Root of your prepared example data (see section 7). |
| `SPATIAL_TOOLKIT_FLIM_SOURCE_DIR` | vendored copy in `src/spatial_omics_toolkit/external_scripts/` | Directory of FLIM analysis scripts. Overrides the vendored copy when set. |
| `SPATIAL_TOOLKIT_HS_SOURCE_DIR` | vendored copy in `src/spatial_omics_toolkit/external_scripts/` | Directory of hyperspectral analysis scripts. Overrides the vendored copy when set. |

Example:

```bash
export SPATIAL_TOOLKIT_WORKSPACE_ROOTS="<your-lab-workspace>"
export SPATIAL_TOOLKIT_PYTHON="<your-lab-workspace>/envs/spatial/bin/python"
export SPATIAL_TOOLKIT_ENV_SCRIPT="<your-lab-workspace>/env/spatial_env.sh"
export SPATIAL_TOOLKIT_SLURM_ACCOUNT="<your-slurm-account>"
export SPATIAL_TOOLKIT_SLURM_PARTITION="<your-partition>"
```

Your interpreter path is whatever `setup.sh` produced in your workspace; the path above is a placeholder.

Approved roots matter: inputs and outputs must resolve under `SPATIAL_TOOLKIT_WORKSPACE_ROOTS`. Rejected paths name this variable in the error message.

## 6. Launch the app on a compute allocation

Request an interactive allocation. Use your own account and partition:

```bash
salloc --account=<your-slurm-account> --partition=<your-partition> \
       --cpus-per-task=4 --mem=16G --time=04:00:00
```

On the allocated node, activate your environment and start the app bound to loopback only:

```bash
streamlit run apps/streamlit/app.py \
  --server.address 127.0.0.1 \
  --server.port <port> \
  --global.developmentMode=false
```

Do not change the address to `0.0.0.0` or a public interface. The app has no authentication (section 8).

To view the app from your own computer, forward the port over SSH. This example uses placeholders; the forward is only active while the command runs:

```bash
ssh -L <local-port>:<compute-node>:<port> <user>@<login-host>
```

Then open `http://localhost:<local-port>` in your browser. Use the compute node name shown by your allocation for `<compute-node>`.

## 7. Keep your data in your own workspace

- Put raw data, examples, and run outputs under your own workspace. Output roots must lie under `SPATIAL_TOOLKIT_WORKSPACE_ROOTS`.
- To use the prepared example catalog, set `SPATIAL_TOOLKIT_EXAMPLES` to your own example root and prepare it there:

  ```bash
  python -m spatial_omics_toolkit.application.examples --prepare
  ```

  The app reads `quickstart/catalog.json` under that root. If it is missing, the app opens with "Use my own data" and does not substitute placeholder data.
- Nothing from `fixtures/`, `runs/`, `.llm-wiki/`, or `.agents/` is committed. These are local to each clone and are ignored by `.gitignore`. Do not add study data, identifiers, or scratch paths to commits.

## 8. Current limitations

- **No authentication.** Never expose the app publicly. Keep it on `127.0.0.1` and reach it only through an SSH forward.
- **Integration is exploratory.** The only integration workflow is a translation-only serial-section preview of H&E CZI against a fluorescence or HS CZI channel. It does not establish biological pairing, same-cell identity, or an accepted alignment.
- **HS–FLIM and other integration pairs are not implemented.**
- **Nextflow execution is not a guarantee of scientific validity.** Review QC, warnings, and preflight before interpreting results.
- Community Cloud deployment cannot reach HPC filesystems; use an HPC deployment for analysis.

For architecture and extension details, see [DESIGN.md](../DESIGN.md). For contribution rules, see [CONTRIBUTING.md](../CONTRIBUTING.md).
