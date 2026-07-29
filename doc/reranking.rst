GNINA and DiffDock reranking
============================

ProDock includes a reproducible workflow for combining rank-resolved GNINA and
DiffDock tables and optimizing descriptor thresholds with Optuna. The workflow
can validate and combine one target without optimization, optimize one target,
or process every target discovered in an input directory.

Install the optional dependencies
---------------------------------

From a source checkout, install ProDock and the reranking dependencies:

.. code-block:: bash

   python -m pip install -e ".[reranking]"

The equivalent requirements file is useful when the package is already
installed:

.. code-block:: bash

   python -m pip install -r requirements-reranking.txt

Use ``requirements-dev.txt`` instead when running the repository tests or
linting tools as well.

Input layouts
-------------

The optimizer accepts two engine-specific layouts.

Repository layout
~~~~~~~~~~~~~~~~~

The committed benchmark uses one CSV per engine and target:

.. code-block:: text

   Project/benchmark/
   ├── result_gnina/
   │   └── ABL1_final.csv
   └── result_diffdock/
       └── ABL1_final.csv

For explicit train/test data, use ``ABL1_train_final.csv`` and
``ABL1_test_final.csv``. The unsplit ``ABL1_final.csv`` name is accepted only
as a training input.

Nested export layout
~~~~~~~~~~~~~~~~~~~~

Older campaign exports remain supported:

.. code-block:: text

   all/
   ├── gnina/ABL1/Confidence_score/ABL1_train_final.csv
   └── diffdock/ABL1/Confidence_score/ABL1_train_final.csv

Alternatively, ``--data-dir`` and ``--split-merged-dir`` accept already merged
files named ``{target}_train.csv`` and ``{target}_test.csv``.

Activity labels
---------------

The optimizer never guesses labels silently. Choose one of these contracts:

* Store a binary column in the input and name it with ``--activity-column``.
  The default column name is ``Active``.
* For the committed DUD-E-style tables, pass ``--dude-labels``. Identifiers
  beginning with ``ZINC`` are then decoys and all other identifiers are
  actives.

An embedded activity column takes precedence when it is present. If neither
contract can supply labels, the script stops with an error.

.. warning::

   ``MT1`` contains ``NONE`` as its only non-ZINC identifier. Exclude this
   target from aggregate benchmark runs until a valid active record is
   recovered.

Validate one target
-------------------

Use combine-only mode before a long Optuna run. It loads both engine files,
aligns conformations by compound and rank, resolves labels, and writes the
combined table without optimization:

.. code-block:: bash

   python Project/Optimization_script/optuna_combine_all_structure.py \
     --protein ABL1 \
     --base-dir Project/benchmark \
     --dude-labels \
     --combine-only

The output is ``Combined/ABL1_train_final.csv``. Check the reported molecule,
active, and decoy counts before continuing.

Optimize one target
-------------------

.. code-block:: bash

   python Project/Optimization_script/optuna_combine_all_structure.py \
     --protein ABL1 \
     --base-dir Project/benchmark \
     --dude-labels \
     --scoring-metric affinity \
     --metric roc-auc \
     --n-trials 200 \
     --n-jobs 1 \
     --top-k 10 \
     --output-dir results_all_structure

``--scoring-metric`` selects the final ranking score:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Value
     - Meaning
   * - ``affinity``
     - Use empirical affinity; lower raw affinity is treated as better.
   * - ``cnnaffinity``
     - Use GNINA CNN affinity.
   * - ``cnn-combined``
     - Use the product of CNN pose and CNN affinity.

``--metric`` selects the Optuna objective: ``roc-auc``, ``pr-auc``, or
``logauc``. The result JSON always reports ROC-AUC as a comparison metric in
addition to the selected objective.

Run several or all targets
--------------------------

Preview target discovery without starting optimization:

.. code-block:: bash

   python Project/Optimization_script/run_all_proteins_optimization.py \
     --base-dir Project/benchmark \
     --dude-labels \
     --exclude-proteins MT1 \
     --dry-run

Run the discovered targets:

.. code-block:: bash

   python Project/Optimization_script/run_all_proteins_optimization.py \
     --base-dir Project/benchmark \
     --dude-labels \
     --exclude-proteins MT1 \
     --metric roc-auc \
     --n-trials 100 \
     --parallel 4 \
     --n-jobs 1

``--parallel`` controls how many targets run concurrently.
``--n-jobs`` controls Optuna workers inside each target. Their product is the
approximate maximum worker count, so increase one dimension at a time to avoid
oversubscribing the machine.

Train/test evaluation
---------------------

When explicit split files are available, optimize on training data and replay
the selected thresholds on test data:

.. code-block:: bash

   python Project/Optimization_script/optuna_combine_all_structure.py \
     --protein ABL1 \
     --data-dir split_merged \
     --split train \
     --eval-on-test \
     --activity-column Active

For a batch run, use ``--split-merged-dir split_merged`` together with
``--evaluate-test``. Test metrics are added to the same target result JSON; the
test split does not participate in threshold selection.

Outputs
-------

By default, a single-target optimization writes:

.. code-block:: text

   results_all_structure/
   └── ABL1/
       ├── ABL1_all_structure_results.json
       └── ABL1_all_structure_study.pkl  # only with --save-study

The JSON records the target, objective, optimized and baseline metrics,
improvements, selected thresholds, and metric counts. Batch runs also write
``batch_optimization_results.json`` unless ``--batch-results-file`` selects
another path.

Extract threshold tables after running the nine scoring/objective
configurations:

.. code-block:: bash

   python Project/Analysis_script/extract_optimized_thresholds.py \
     --root configuration-results \
     --output-dir threshold_csv

Optimization results, combined tables, Optuna studies, threshold tables, and
plots are generated artifacts. They are ignored by Git and should be
regenerated from the committed inputs and scripts.

Troubleshooting
---------------

``ModuleNotFoundError: sklearn``
   Install ``requirements-reranking.txt`` or the ``reranking`` package extra.

No GNINA or DiffDock inputs found
   Confirm that both engines have a file for the requested target and split.
   Use ``--dry-run`` to inspect batch discovery.

Activity column not found
   Supply the correct ``--activity-column`` or explicitly opt into
   ``--dude-labels`` for DUD-E-style identifiers.

Only one activity class
   ROC-AUC cannot be estimated from an all-active or all-decoy target. Check
   the labels and exclusions before interpreting the no-information result.

Constant scores or no conformation passed thresholds
   Inspect the combined CSV, descriptor ranges, and ``--top-k``. The optimizer
   reports a no-information baseline for uncomputable trials rather than
   treating them as successful enrichment.

See also
--------

* :doc:`reproducibility` for source inputs, artifact boundaries, and checks.
* :doc:`tutorial/postprocess` for screening metrics and pose post-processing.
* :doc:`api/cli` for the main JSON-driven docking command.
