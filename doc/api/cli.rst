CLI
===

The :mod:`prodock.cli` module provides a JSON-first command-line interface for
running end-to-end ProDock workflows. It is designed for reproducibility: a
single configuration file can define receptors, ligands, engine settings,
interaction extraction, and database export, while explicit command-line flags
can override selected run parameters without editing the base config.

Overview
--------

.. raw:: html

   <div class="pd-grid-2">
     <div class="pd-mini">
       <h3>Config-driven</h3>
       <p>Store a full campaign in one JSON file for reproducible reruns and version control.</p>
     </div>
     <div class="pd-mini">
       <h3>Override-friendly</h3>
       <p>Change engines, CPU, exhaustiveness, output database name, and other runtime options from the shell.</p>
     </div>
     <div class="pd-mini">
       <h3>Validation-first</h3>
       <p>Input modes are checked before execution, including receptor mode and ligand mode consistency.</p>
     </div>
     <div class="pd-mini">
       <h3>Pipeline-connected</h3>
       <p>The CLI normalizes user input and forwards it to the core <code>prodock(...)</code> workflow.</p>
     </div>
   </div>

Quick start
-----------

Run a full campaign from a JSON file:

.. code-block:: bash

   python -m prodock --config run.json

Write a compact machine-readable run summary:

.. code-block:: bash

   python -m prodock --config run.json --summary-json outputs/summary.json

Override selected runtime parameters from the shell:

.. code-block:: bash

   python -m prodock \
       --config run.json \
       --engines qvina qvina-w smina \
       --cpu 8 \
       --n-jobs 8 \
       --exhaustiveness 16 \
       --n-poses 20 \
       --db-name campaign.sqlite

Configuration model
-------------------

The CLI supports two mutually exclusive receptor modes and two mutually
exclusive ligand modes:

* ``receptors`` or ``prepared_receptors``
* ``ligands`` or ``ligand_dir``

A typical configuration file is:

.. code-block:: json

   {
     "project_dir": "Data/testcase/Multi",
     "receptors": [
       {
         "pdb_id": "4WKQ",
         "receptor_name": "EGFR_4WKQ",
         "ligand_code": "IRE",
         "chains": ["A"],
         "cofactors": []
       }
     ],
     "ligands": [
       {
         "id": "erlotinib",
         "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C"
       },
       {
         "id": "gefitinib",
         "smiles": "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F"
       }
     ],
     "config": {
       "engines": ["qvina", "qvina-w"],
       "extract_interaction": true,
       "db_name": "test.db",
       "cpu": 8,
       "n_jobs": 8,
       "exhaustiveness": 16,
       "n_poses": 20
     }
   }

The CLI also accepts nested runtime dictionaries under ``config``, ``options``,
or ``run``. These are merged into one normalized argument dictionary before the
pipeline is called.

Common override patterns
------------------------

Switch to prepared receptors and a ligand directory in the base config, then
override only the engines and database output:

.. code-block:: bash

   python -m prodock \
       --config prepared_campaign.json \
       --engines vina smina \
       --db-name prepared.sqlite

Enable interaction extraction and keep the console output quiet except for the
summary file:

.. code-block:: bash

   python -m prodock \
       --config run.json \
       --extract-interaction \
       --summary-json reports/run_summary.json \
       --quiet

Request a traceback for debugging invalid or failing runs:

.. code-block:: bash

   python -m prodock --config run.json --traceback

Notes
-----

* Relative paths inside the JSON file are resolved relative to the config file location.
* Explicit command-line flags take precedence over JSON-provided runtime options.
* The CLI validates that exactly one receptor mode and exactly one ligand mode are selected.
* The final return object is condensed into a JSON-serializable summary for console or file output.

Reference
---------

.. automodule:: prodock.cli
   :members:
   :undoc-members:
   :show-inheritance:
