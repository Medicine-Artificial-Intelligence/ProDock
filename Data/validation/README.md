# ProDock revision: workstation and cluster experiments

**No experiment is launched by a dry run.** Local experimental jobs were stopped
when requested. The commands below are for execution by the author on the
cluster. Completed earlier local outputs are preserved separately and are not
substituted for the planned full campaign.

The manuscript keeps its original narrative and Results structure. The twelve
explicit minor wording requests have been checked. Changes and author responses
remain in LoLO Blue. Figure 1 keeps its original artwork, with corrected labels
and larger panels; it no longer uses the unrelated replacement schematic.

## What we still need: four core packages

| Package | Chemical or practical question | Workload |
|---|---|---|
| 1. Preparation and native-pose controls | Does ligand-free preparation distort the occupied EGFR pocket? Does retaining observed crystal coordinates preserve the native binding mode? | 2 structures × 2 preparation policies × 2 programs = **8 docking tasks** |
| 2. Real chemical application | How do two EGFR structures change compound priorities and MET hinge contacts? Are apparent benefits explained by ligand size, flexibility, or the template ligand's footprint? | **1,000 × 2 × Smina = 2,000 tasks**, plus **32 × 2 × QuickVina 2 = 64 tasks** |
| 3. Integrity and provenance | Do coordinates, chemistry, scores, interactions, metadata, failed writes and replay behave as documented? | Small deterministic fixtures; **no docking** |
| 4. Pinned functional comparison | Can stored poses/scores be recovered and the same consensus analysis regenerated using ProDock and native modules from related tools? | Small shared fixture; **no docking** |

**Total planned new docking calls: 2,072**, conditional on package 1 qualifying
its protein inputs. At most five poses are requested per call. Packages 3–4 are
inexpensive software experiments, not chemical screening results.

Common-pocket sensitivity, score/size diagnostics, chemical structures, and
shortlist interpretation reuse package 2's stored poses. **They do not require
another docking campaign.** The 32-compound second-program subset is fixed by
the frozen library order, not chosen using docking outcomes.

This is a scoped software-and-chemistry revision, not an attempt to satisfy every
possible expansion. It does not guarantee acceptance or justify marking every
reviewer point resolved.

## Chemical design and interpretation

The included library is a frozen ChEMBL 37 EGFR binding-assay IC50 cohort:

- 1,000 exact-structure-unique compounds and 638 Bemis–Murcko scaffolds.
- 400 compounds with median IC50 ≤100 nM, 400 intermediate, and 200 with median
  IC50 ≥10 µM. These are descriptive annotations, not experimentally confirmed
  inactive decoys.
- Standardized numeric IC50 records in nM; explicitly annotated variants,
  descriptions matching declared mutant terms, and validity-flagged records are
  filtered. Absence of a variant annotation does not prove wild-type assay use.
- Single-fragment organic inputs, 12–50 heavy atoms and at most 12 rotatable
  bonds, in the supplied chemical state. There is no pH-state or tautomer
  enumeration. Sampling is deterministic, with no docking-based selection.

The two protein structures are **1M17/erlotinib** and **4ZAU/AZD9291
(osimertinib)**. The study concerns noncovalent recognition geometry; ordinary
Vina-family docking does **not** model covalent reaction chemistry or establish
mutant selectivity. Acrylamide substructure flags are annotations only.

The same hinge region has different numbering in source and prepared files.
The analysis obtains the correspondence from the generated protein context,
rather than assuming MET769/MET793 or a fixed renumbering. Native reference
coordinates are transformed with the protein correspondence; the ligand is
never independently fitted to claim binding-pose recovery.

Prespecified analyses include:

1. Paired score ranks and top-decile overlap across structures.
2. Actual ProLIF hinge-residue hydrogen bonds and other contacts, with missing
   or failed profiling separated from a genuine absence of contacts.
3. Dependence of scores/contact summaries on heavy-atom count and flexibility.
4. A sequence-mapped common pocket, with fixed 5/6/8 Å neighborhood sensitivity,
   contrasted with template-footprint overlap.
5. Deterministic inspection lists: stable high ranks with hinge contacts,
   favorable ranks lacking hinge contacts, and structure-sensitive compounds.
   Activity annotations do not choose the examples, thresholds, or weights.

IC50 records come from heterogeneous assays. Their pooled medians are useful
context, **not a homogeneous potency benchmark or prospective validation**.
The chemical-analysis protocol is in [protocols/chemical_usecase.txt](protocols/chemical_usecase.txt).

## Included inputs and offline use

- [inputs/chembl_egfr/library.json](inputs/chembl_egfr/library.json): selected
  structures, activity/assay/document IDs, filtering rules, source page URLs,
  hashes, ChEMBL release metadata, and library checksum.
- [inputs/egfr_structures/manifest.json](inputs/egfr_structures/manifest.json):
  raw PDB coordinates, experimental bound-ligand references, CCD chemical
  definitions, source URLs and input checksums for both structures.
- `inputs/comparison_native/`: exact pinned native helper sources, checksums,
  and their upstream GPL licenses. The cached files retain those licenses.

The cluster execution path uses these included inputs and cached helper files;
it does not need internet access on compute nodes. Refetching ChEMBL through
`fetch_egfr_library.py` is optional and creates a **new** frozen snapshot; do not
silently mix it with an existing campaign. Raw downloaded pages are local
provenance material and are excluded from Git; their URLs/hashes remain in the
selected library manifest.

The supplied `prodock_cluster_package.zip` is a standalone source-and-input snapshot. Extract it and enter its `prodock_cluster/` directory before running the commands below; Git metadata is optional because its checksum manifest and per-stage source hashes identify the snapshot.

Transfer the **current source changes**, not just the old release: batch-write
atomicity and saved-campaign snapshot fixes are part of the integrity package.
Keep the `prodock/dock/binary/` executables and make them executable on the Linux
cluster. These scripts are CPU workflows; no GPU stack is required.

## Environment and dry run

Use your `prodock` conda environment. `requirements.txt` in this directory lists the extra process-monitoring dependency (psutil). The main requirements are the repository's
normal dependencies, RDKit, Meeko/gemmi, OpenMM/PDBFixer, ProLIF/MDAnalysis,
NumPy, pandas, SciPy, matplotlib and psutil. The comparison invokes limited
native modules; it does not install complete DockM8/MolDockLab stacks.

From the repository root:

```bash
conda activate prodock
export PRODOCK_PYTHON="$CONDA_PREFIX/bin/python"
export PRODOCK_WORKDIR="/your/scratch/prodock-revision-v1"
export PRODOCK_WORKERS=8
chmod +x prodock/dock/binary/smina prodock/dock/binary/qvina
bash Data/validation/cluster_revision.sh --dry-run
```

The dry run prints input identities, eight native controls, the cheap fixture
steps, 2,064 application tasks and the protein quality gate. **A blocked large
campaign is expected before the native controls exist.** A dry-run plan is not
proof that simulation libraries, binaries or the final chemical model will work.

Individual metadata-only plans:

```bash
python Data/validation/preparation_sensitivity.py \
  --input-manifest Data/validation/inputs/egfr_structures/manifest.json \
  --output "$PRODOCK_WORKDIR/preparation_quality" --dry-run

python Data/validation/run_revision_experiments.py --plan --suite full \
  --output "$PRODOCK_WORKDIR/evidence/full" \
  --campaign-output "$PRODOCK_WORKDIR/chemical_campaign" \
  --receptors "$PRODOCK_WORKDIR/preparation_quality/ready_receptors.json" \
  --context "$PRODOCK_WORKDIR/preparation_quality/chemical_context.json" \
  --quality-report "$PRODOCK_WORKDIR/preparation_quality/quality_report.json"
```

## Run on a workstation (without Slurm)

The `revision` branch contains the code, frozen inputs and scripts. Manuscript
files under `paper/` are not needed to run these experiments and are not included
in the code commit. From an existing repository checkout on the workstation:

```bash
git fetch origin
git switch revision
conda activate prodock
python -m pip install --no-deps -e .
python -m pip install -r Data/validation/requirements.txt
export PRODOCK_PYTHON="$CONDA_PREFIX/bin/python"
export PRODOCK_WORKDIR="$PWD/Data/validation/runs/workstation_revision_v1"
export PRODOCK_WORKERS=4
bash Data/validation/cluster_revision.sh --dry-run
```

After reviewing the plan, launch the same wrapper directly:

```bash
bash Data/validation/cluster_revision.sh --execute
```

Set the worker count to suit the available CPU and memory. This runs locally on
that workstation; it does not call `sbatch`. It preserves the native-pose quality
gate before the larger chemical campaign. To resume a matching run, keep the
same work directory, set `export PRODOCK_RESUME=1`, and repeat `--execute`.

## Submit to Slurm

Submit **from the repository root**. Set your site-specific account/partition
through `sbatch` options; the template intentionally supplies neither.

```bash
export PRODOCK_REPO="$PWD"
export PRODOCK_WORKDIR="/your/scratch/prodock-revision-v1"
export PRODOCK_CONDA_ENV=prodock
# If conda is not on the batch PATH, also set PRODOCK_CONDA_BASE.
sbatch Data/validation/cluster_revision.sbatch
```

The template requests **one node, eight CPUs, 32 GiB, and 24 hours**. These are
allocation defaults, not measured resource requirements. Change them for your
cluster. Each docking task uses one CPU; the worker count is capped to the
Slurm CPU allocation. Numerical-library/OpenMM thread counts are bounded to
avoid multiplying threads inside each task.

The job first checks input checksums, cached native-source bytes and dependency availability. It runs cheap fixtures, prepares and tests both protein policies, then
starts the full chemical application **only if the computed quality gate
passes**. Before any new outcomes are examined, the native recovery criterion is declared for the two crystal-coordinate-preserving **primary Smina** cases (best-scored RMSD ≤2 Å). Secondary-program and deliberately minimized-baseline outcomes remain reported, with failed/timeout outcomes counted separately. They do not all have to reproduce the native pose for the primary study to proceed. The native controls use seed 42, exhaustiveness 8, five requested
poses, and a 180-second per-task limit. Main docking uses the same settings.
Timeouts remain recorded outcomes; they are not silently dropped or relabeled
as chemical inactivity. Default-model minimization uses the documented
5,000-iteration ceiling, separately from the coordinate-preserving policy.

If the native controls fail, inspect `preparation_quality/summary.csv`, the
pose files and `quality_report.json` before spending on the larger campaign.
Do not tune a box or select successful structures just to recover a target
success rate. Overrides exist for a clearly labeled failure illustration, but
**the cluster wrapper never bypasses the gate automatically**.

## Resume and output checks

For a rerun, use the **same work directory**:

```bash
export PRODOCK_RESUME=1
sbatch Data/validation/cluster_revision.sbatch
```

A completed qualifying quality stage can be reused only when its input, parameters, source and generated-file identities match. Failed/incomplete quality stages are preserved for inspection; use a fresh work directory for a changed preparation protocol. The coordinator checks source/command/result identities. The chemical runner
reuses a task only when its input, settings, binary and output hashes agree.
Missing/truncated or changed outputs are rescheduled. This is a documented
validation-runner policy, **not generic persisted checkpoint/resume in every
ProDock API**. Different preparation policies use separate output/database
namespaces because the current pose key does not preserve parameter variants
under identical protein–ligand–program–rank identifiers.

Inspect:

- `preparation_quality/{report.json,quality_report.json,summary.csv}`.
- `evidence/cheap/`: controlled integrity and pinned native-module results.
- `evidence/full/execution_manifest.json`: stage completion, commands and logs.
- `chemical_campaign/{manifest.json,invocations.jsonl,resources.jsonl}`.
- Task plans, attempts and per-task hashes; `preparation.csv`.
- `chemical_campaign.db`: all parsed valid ranks and rank-1 interaction records.
- `annotated_rank1.csv`, interaction exports, and `analysis_counts.json`.
- Chemical use-case figures, contact/size summaries and inspection shortlists
  under `evidence/full/chemical_usecase/`.

Do not infer a completed campaign from a database file existing. Reconcile all
scheduled tasks, successes, failures/timeouts, parsed ranks and profiled poses.
Resource measurements report wall time, sampled aggregate RSS and storage;
shared memory may be counted more than once in the RSS sum.

## What we can reasonably omit

| Extra request or claim | Scoped response instead of a large new study |
|---|---|
| Relative accuracy of every supported engine | One primary program and a fixed small cross-check; make no engine-superiority claim. |
| Fitted consensus/ML rescoring | Fixed, unfitted descriptive queries; no trained model or performance claim. |
| Prospective hit discovery or enrichment | Retrospective chemical annotations and inspection examples only. |
| Full GPU workflow installations | Execute and identify the native modules actually compared; do not call this a full-platform comparison. |
| Cluster/multi-node scaling or maximum supported size | Report the tested single-node workload and measured limits. |
| Exhaustive protein-preparation survey | Two mechanism-focused controls, plus the optional compact panel below; no population-wide failure-rate claim. |

A reviewer may still require more. The defensible position is to **narrow the
claims and state the limits**, not claim an omitted experiment was completed.

The optional 12-structure preparation panel is implemented in
`preparation_panel.py`; it is not launched by the core cluster wrapper and needs
the additional `Project/benchmark/` inputs from the full repository (not included
in the minimal transfer archive). It has a
200-iteration CPU protocol and a 90-second case budget; timeouts must be
separated from structural exceptions. The archived redocking audit likewise
requires matching original campaign artifacts and is optional on a fresh clone.
The smaller `--suite minimum` uses 250 compounds but is not the selected full
cluster scope.

No script automatically edits manuscript numbers, marks reviewer requests
resolved, publishes, or submits anything. Bring the cluster reports back for
chemical review before updating the marked manuscript and responses.
