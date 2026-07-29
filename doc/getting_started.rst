Getting Started
===============

.. raw:: html

   <div class="pd-hero-compact">
     <div class="pd-kicker">Quick start</div>
     <h2>Install and run ProDock</h2>
     <p class="pd-lead">
       Create an isolated environment, then install ProDock with
       <strong>Conda</strong>, <strong>Pip</strong>, or <strong>Source</strong>.
     </p>
     <div class="pd-badge-row">
       <span class="pd-install-badge pd-install-conda">
         <span class="pd-install-icon pd-install-icon-conda">◆</span>
         <span><strong>Conda</strong> · easiest</span>
       </span>
       <span class="pd-install-badge pd-install-pip">
         <span class="pd-install-icon pd-install-icon-pip">Py</span>
         <span><strong>Pip</strong> · PyPI</span>
       </span>
       <span class="pd-install-badge pd-install-source">
         <span class="pd-install-icon pd-install-icon-source">Git</span>
         <span><strong>Source</strong> · editable</span>
       </span>
       <span class="pd-install-badge pd-install-env">
         <span class="pd-install-icon pd-install-icon-env">V</span>
         <span><strong>Env first</strong></span>
       </span>
     </div>
   </div>

.. raw:: html

   <div class="pd-install-flow">
     <div class="pd-flow-step">
       <div class="pd-flow-num">1</div>
       <div class="pd-flow-text">
         <strong>🧪 Create env</strong>
         <span>Fresh conda environment</span>
       </div>
     </div>
     <div class="pd-flow-arrow">→</div>
     <div class="pd-flow-step">
       <div class="pd-flow-num">2</div>
       <div class="pd-flow-text">
         <strong>📦 Install</strong>
         <span>Conda, Pip, or Source</span>
       </div>
     </div>
     <div class="pd-flow-arrow">→</div>
     <div class="pd-flow-step">
       <div class="pd-flow-num">3</div>
       <div class="pd-flow-text">
         <strong>🚀 Run</strong>
         <span><code>prodock(...)</code></span>
       </div>
     </div>
   </div>

Installation
------------

.. raw:: html

   <div class="pd-grid-3">

   <div class="pd-install-card pd-install-card-conda">
   <div class="pd-install-card-head">
   <div class="pd-install-mark pd-install-mark-conda">◆</div>
   <div>
   <h3>Conda</h3>
   <p>Recommended.</p>
   </div>
   </div>

.. code-block:: bash

   conda create -n prodock python=3.11
   conda activate prodock
   conda install -c tieulongphan prodock

.. raw:: html

   <div class="pd-note-inline">
     <strong>Use:</strong> quickest install
   </div>
   </div>

   <div class="pd-install-card pd-install-card-pip">
   <div class="pd-install-card-head">
   <div class="pd-install-mark pd-install-mark-pip">Py</div>
   <div>
   <h3>Pip</h3>
   <p>Install dependencies first.</p>
   </div>
   </div>

.. code-block:: bash

   conda create -n prodock python=3.11
   conda activate prodock
   conda install -c conda-forge openmm=8.3.1 pdbfixer
   pip install prodock

.. raw:: html

   <div class="pd-note-inline">
     <strong>Use:</strong> PyPI install
   </div>
   </div>

   <div class="pd-install-card pd-install-card-source">
   <div class="pd-install-card-head">
   <div class="pd-install-mark pd-install-mark-source">Git</div>
   <div>
   <h3>Source</h3>
   <p>Editable install.</p>
   </div>
   </div>

.. code-block:: bash

   git clone https://github.com/Medicine-Artificial-Intelligence/ProDock
   cd ProDock
   conda env create -f prodock-env.yml
   conda activate prodock

.. raw:: html

   <div class="pd-note-inline">
     <strong>Use:</strong> development
   </div>
   </div>
   </div>

Installation profiles
---------------------

The source checkout provides separate dependency profiles so a minimal docking
installation does not need the analysis stack:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Goal
     - Command
     - Includes
   * - Core package
     - ``python -m pip install -e .``
     - Docking, preparation, post-processing, and database APIs
   * - Reranking
     - ``python -m pip install -e ".[reranking]"``
     - NumPy, SciPy, scikit-learn, seaborn, Optuna, and joblib
   * - Repository development
     - ``python -m pip install -r requirements-dev.txt``
     - Reranking dependencies, pytest, flake8, and Black

``prodock-env.yml`` creates the source-development environment and installs the
reranking and test tools. ``environment.yml`` is the larger GPU workflow
environment used by the DiffDock and GNINA campaign scripts.

External docking engines
------------------------

The Python package does not bundle every docking executable. Install the
engines used by your campaign and make their commands available on ``PATH``.
For example:

.. code-block:: bash

   command -v vina
   command -v smina
   command -v qvina
   command -v qvina-w

Only the executables named in the campaign configuration are required.

Verify the installation
-----------------------

.. code-block:: bash

   python -c "import prodock; print(prodock.__version__)"
   prodock --help

For a source checkout, validate the bundled case-study configuration without
starting docking:

.. code-block:: bash

   prodock \
     --config Data/case/config.json \
     --receptor-json Data/case/receptor.json \
     --ligand-json Data/case/ligand.json \
     --validate-only


Quick example
-------------

.. code-block:: python

   from prodock import prodock

   PROJECT = "Demo"

   RECEPTORS = [
       {
           "pdb_id": "4WKQ",
           "receptor_name": "EGFR_4WKQ",
           "ligand_code": "IRE",
           "chains": ["A"],
           "cofactors": [],
       },
   ]

   LIGANDS = [
       {
           "id": "erlotinib",
           "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C",
       },
       {
           "id": "gefitinib",
           "smiles": "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F",
       },
   ]

   result = prodock(
       PROJECT,
       receptors=RECEPTORS,
       ligands=LIGANDS,
       engines=["qvina", "qvina-w"],
       extract_interaction=True,
       db_name="test.db",
   )

   print(result.campaign_json)
   print(result.db_path)
   print(result.merged_df.head())

.. raw:: html

   <div class="pd-grid-2">
     <div class="pd-mini-card">
       <h4>📁 Output paths</h4>
       <p><code>result.campaign_json</code> is the path to the generated campaign config, and <code>result.db_path</code> is the path to the SQLite database.</p>
     </div>
     <div class="pd-mini-card">
       <h4>📊 Output tables</h4>
       <p><code>result.merged_df</code> stores pose-level rows such as receptor, ligand, engine, rank, affinity, and RDKit molecule objects and pose id for downstream analysis.</p>
     </div>
   </div>
