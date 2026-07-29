Reproducibility
===============

ProDock separates version-controlled inputs from generated campaign artifacts.
A reproducible run records the source revision, effective configuration,
software environment, external docking-engine versions, and output location.

What is version controlled
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 42 32

   * - Workflow
     - Inputs and drivers
     - Generated outputs
   * - Package validation
     - ``prodock/``, ``Test/``, dependency files
     - caches, coverage, built documentation
   * - EGFR case study
     - ``Data/case/`` and ``Data/benchmark/*.py`` or ``*.sh``
     - SQLite databases, logs, tables, figures
   * - GNINA + DiffDock reranking
     - ``Project/benchmark/`` CSVs and ``Project/Optimization_script/``
     - combined CSVs, Optuna studies, result JSON, threshold tables, plots

Generated artifacts are intentionally excluded from Git. The authoritative
manifest is maintained in the repository-level ``REPRODUCIBILITY.md`` file.

Record a docking configuration
------------------------------

Validate split JSON inputs without starting a campaign:

.. code-block:: bash

   prodock \
     --config Data/case/config.json \
     --receptor-json Data/case/receptor.json \
     --ligand-json Data/case/ligand.json \
     --validate-only

Write the fully merged configuration and a compact run summary:

.. code-block:: bash

   prodock \
     --config Data/case/config.json \
     --receptor-json Data/case/receptor.json \
     --ligand-json Data/case/ligand.json \
     --effective-config-json effective-config.json \
     --summary-json run-summary.json

Relative paths inside each JSON input are resolved relative to that input
file. Relative summary and effective-config paths are resolved relative to the
main config file.

Capture software versions
-------------------------

Before a production run, save the package revision and executable versions:

.. code-block:: bash

   git rev-parse HEAD
   python --version
   python -m pip freeze
   vina --version
   smina --version
   qvina --version
   qvina-w --version

Only record engines used by the campaign. GPU workflows should additionally
record the driver, CUDA, GNINA, and DiffDock revisions.

Run repository checks
---------------------

Install the complete test and reranking profile:

.. code-block:: bash

   python -m pip install -r requirements-dev.txt

Then run:

.. code-block:: bash

   python -m compileall -q prodock Project Data/benchmark
   python -m pytest -q Test
   git diff --check

Build the Sphinx documentation with warnings treated as errors:

.. code-block:: bash

   python -m sphinx -W --keep-going -b html doc doc/_build/html

Reproduce the included workflows
--------------------------------

Run the checkpointed EGFR campaign and analysis:

.. code-block:: bash

   Data/benchmark/run_all.sh --no-paper

This workflow requires ``smina``, ``vina``, ``qvina``, and ``qvina-w`` on
``PATH`` and network access for initial structure retrieval.

Validate a committed reranking target without running Optuna:

.. code-block:: bash

   python Project/Optimization_script/optuna_combine_all_structure.py \
     --protein ABL1 \
     --base-dir Project/benchmark \
     --dude-labels \
     --combine-only

See :doc:`reranking` for optimization, batch execution, train/test replay, and
output interpretation.

Checklist for reporting a run
-----------------------------

* Source commit and branch.
* Effective JSON configuration.
* Python environment and ProDock version.
* Docking-engine versions and hardware-relevant settings.
* Random seed, exhaustiveness, pose count, and worker counts.
* Input dataset identity and activity-label contract.
* Output directory and any exclusions.
* Test or validation command used before the run.
