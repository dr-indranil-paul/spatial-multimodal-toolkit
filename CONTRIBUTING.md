# Contributing

Setup for a new workspace is in [docs/COLLABORATOR_SETUP.md](docs/COLLABORATOR_SETUP.md). This file covers the working rules.

## Branches and pull requests

- Never push to `main`. Open a pull request into `main`.
- Use one short-lived branch per task, named `<name>-<topic>`, for example `yhc-cycif-registration`:

  ```bash
  git fetch origin
  git switch -c yhc-cycif-registration origin/main
  git push -u origin yhc-cycif-registration
  ```

- Delete the branch after its pull request is merged.
- A branch created before the portability changes are merged must run `git fetch origin && git merge origin/main` after that pull request lands.
- The owner must confirm the rules in [Repository rules](#repository-rules-owner) below. Nothing enforces them until the owner enables them.

## Repository rules (owner)

These rules depend on the GitHub plan and settings of the repository owner. They are not on by default.

- **Plan.** Branch protection and required reviews on a private repository need GitHub Pro, Team or Enterprise. Check Settings → Billing and plans. On a free plan, use the fallback: everyone agrees in writing not to push to `main`, and the owner merges every pull request.
- **Branch protection on `main`.** Settings → Branches → Add rule. Require a pull request with at least one approval, require the check `unit` (the job `unit` in the workflow `tests`) to pass, block force-pushes and deletions, and dismiss stale approvals. Select `unit` after the workflow has run once; the check does not appear in the list before that.
- **Self-approval.** GitHub does not count an author's own approval. If the owner is the only reviewer and opens a PR, either allow admins to merge (leave "Do not allow bypassing the above settings" unticked) or add a second reviewer. Otherwise the PR cannot merge.
- **Code owners.** `.github/CODEOWNERS` has no effect until the same rule ticks "Require review from Code Owners".
- **Delete head branches automatically.** Settings → General → Pull Requests.
- **Forking.** Outside collaborators can fork a private repository only if Settings → General → Forking allows it. If it is disabled, invite them as collaborators instead.

## Agreed rule on the free plan

The repository is on a free GitHub plan, so GitHub cannot enforce these rules. They are a team agreement:

- Nobody pushes to `main`. Every change arrives as a pull request from a topic branch.
- The owner reads each pull request and merges it. Contributors don't merge their own work.
- CI (`unit` in `tests`) runs on every pull request. A failing run is not merged.
- Optional local guard: run `git config core.hooksPath .githooks` once in your clone. The `pre-push` hook refuses direct pushes to `main`. It is advisory only; `git push --no-verify` bypasses it, and GitHub does not check it.
- Making the repository public is not a way out of this rule. It would expose the vendored clinical-analysis code and lab paths, and the data-rights approvals for those files are not yet resolved.

## Adding a workflow adapter

Adapters live in `src/spatial_omics_toolkit/application/workflows/`. Each module implements:

- `describe`: declares the workflow, its stages, `depends_on`, conditions, inputs, outputs, tools, source references, and limitations.
- `preflight`: reports missing prerequisites before submission.
- `execute_stage(request, stage_id, upstream, emit)`: performs only that stage's declared operation and returns a `WorkflowResult`.

Registration is discovered by `application/workflows/registry.py`. Registration is not a scientific qualification. Do not split inseparable scientific operations into separate processes. Keep scientific code in reusable domain modules, and preserve the sample-sheet, mapping, and review gates. [DESIGN.md](DESIGN.md) describes the full contract.

## Scientific behavior and tests

- Scientific behavior must have behavioral tests in `tests/unit/`. Test the boundary or invariant that changed, not only that code runs.
- Tests that require private fixtures must skip with an explicit reason (for example `pytest.skip("...")` or `skipif(..., reason="...")`) and never pass silently or fail on a missing file that should have been skipped.

## Running tests

Run tests on an allocated compute node, never on a login node:

```bash
salloc --account=<your-slurm-account> --partition=<your-partition> --cpus-per-task=4 --mem=16G --time=02:00:00
# on the allocated node, with your environment active:
PYTHONPATH=src pytest tests/unit -q
```

The GitHub Actions workflow (`.github/workflows/tests.yml`) runs `tests/unit` on hosted runners. It does not have access to HPC data and is not a substitute for the compute-node run.

## Data and identifiers

- Do not commit study data, participant or specimen identifiers, raw images, run outputs, or scratch paths.
- Do not commit anything under `fixtures/`, `runs/`, `.llm-wiki/`, or `.agents/`.
- Do not hard-code lab paths. Use the `SPATIAL_TOOLKIT_*` variables described in the setup guide.

## Scientific claims in UI text

Do not overstate results. The app must not claim biological pairing, same-cell identity, or a validated alignment from filenames, acquisition groups, or the translation preview. Review new labels and help text against those limits before merging.

## Documentation

Permanent behavior changes update the relevant docs (`README.md`, `DESIGN.md`, or `docs/`). Keep prose short and direct.
