Automation
==========

.. image:: ../_static/tutorial-automation.svg
   :alt: ProDock automation workflow
   :class: pd-visual


.. raw:: html

   <div class="pd-card-grid pd-card-grid-2">
     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <path d="M8 8h8v8H8z"/>
           <path d="M3 12h5M16 12h5M12 3v5M12 16v5"/>
         </svg>
       </div>
       <h3>One end-to-end entry point</h3>
       <p>
         Run preparation, docking, postprocessing, and optional database export
         through one Python function or one CLI command.
       </p>
     </div>

     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <path d="M6 6h12v12H6z"/>
           <path d="M9 9h6M9 12h6M9 15h4"/>
         </svg>
       </div>
       <h3>JSON-first workflow</h3>
       <p>
         Define campaigns in one all-in-one config or split them across
         config, receptor, and ligand JSON files.
       </p>
     </div>

     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <rect x="4" y="5" width="16" height="14" rx="2"/>
           <path d="M8 9h8M8 13h5"/>
           <circle cx="16.5" cy="13.5" r="1.5"/>
         </svg>
       </div>
       <h3>Override-ready CLI</h3>
       <p>
         Keep stable defaults in JSON and override engines, compute settings,
         paths, and workflow flags directly from the terminal.
       </p>
     </div>

     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <path d="M4 18h16"/>
           <path d="M7 18V9"/>
           <path d="M12 18V6"/>
           <path d="M17 18v-4"/>
         </svg>
       </div>
       <h3>Reproducible effective config</h3>
       <p>
         Validate merged inputs, print the final effective config, and write it
         to disk for debugging and reproducibility.
       </p>
     </div>
   </div>

Automation architecture
-----------------------

The automation layer is the highest-level workflow entry point in ProDock. It
connects the lower-level stages into one reproducible campaign:

- receptor preparation or prepared receptor reuse,
- ligand preparation or prepared ligand reuse,
- single- or multi-engine docking,
- optional interaction extraction,
- optional SQLite export.

The same workflow is exposed through two interfaces:

- :func:`prodock` for Python usage,
- ``prodock --config ...`` or ``python -m prodock --config ...`` for CLI usage.

Configuration patterns
----------------------

The CLI supports two main input patterns:

- **all-in-one JSON** — one file contains project directory, receptor input,
  ligand input, and run options
- **split JSON** — the main config contains project/run options, while receptor
  and ligand definitions are passed separately through
  ``--receptor-json`` and ``--ligand-json``

This makes the automation layer suitable both for simple tutorial runs and for
larger projects where receptor and ligand collections are maintained separately.

Override precedence
-------------------

Configuration values are resolved in this order:

1. ``--config`` provides the base payload
2. ``--receptor-json`` overrides embedded ``receptors`` from ``--config``
3. ``--ligand-json`` overrides embedded ``ligands`` from ``--config``
4. explicit CLI flags override all JSON-derived values

This means you can keep a stable base config and vary engines, compute settings,
or workflow flags from the terminal without editing the JSON files.

Input modes
-----------

After config merging, the final workflow must contain exactly one receptor mode
and exactly one ligand mode.

Supported receptor modes:

- ``receptors`` for raw receptor specifications
- ``prepared_receptors`` for already prepared receptor inputs

Supported ligand modes:

- ``ligands`` for inline ligand dictionaries
- ``ligand_dir`` for a directory of prepared ligand files

This keeps the high-level workflow unambiguous while still supporting both raw
and preprocessed campaigns.

All-in-one JSON
---------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <path d="M6 6h12v12H6z"/>
           <path d="M9 9h6M9 12h6M9 15h4"/>
         </svg>
       </div>
       <div>
         <h2>Single-file campaign config</h2>
         <p>Keep receptors, ligands, and run options together in one JSON file.</p>
       </div>
     </div>
   </div>

The simplest automation layout is one file containing everything:

.. code-block:: json

   {
     "project_dir": "Demo",
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
       "save_to_database": true,
       "db_name": "demo.db",
       "cpu": 8,
       "n_jobs": 8,
       "exhaustiveness": 16,
       "n_poses": 20
     }
   }

Run it with either form:

.. code-block:: bash

   prodock --config run.json

.. code-block:: bash

   python -m prodock --config run.json

This is the most convenient pattern for tutorials, notebooks, and compact
project runs.

Split JSON input
----------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <path d="M5 8h14"/>
           <path d="M5 12h14"/>
           <path d="M5 16h14"/>
           <circle cx="8" cy="8" r="1"/>
           <circle cx="8" cy="12" r="1"/>
           <circle cx="8" cy="16" r="1"/>
         </svg>
       </div>
       <div>
         <h2>Split config, receptor, and ligand files</h2>
         <p>Keep project settings separate from receptor and ligand collections.</p>
       </div>
     </div>
   </div>

For larger campaigns, receptor and ligand definitions can live in separate
files.

Example ``config.json``:

.. code-block:: json

   {
     "project_dir": "Demo",
     "config": {
       "engines": ["qvina", "qvina-w"],
       "extract_interaction": true,
       "save_to_database": true,
       "db_name": "demo.db",
       "cpu": 8,
       "n_jobs": 8,
       "exhaustiveness": 16,
       "n_poses": 20
     }
   }

Example ``receptor.json``:

.. code-block:: json

   {
     "receptors": [
       {
         "pdb_id": "4WKQ",
         "receptor_name": "EGFR_4WKQ",
         "ligand_code": "IRE",
         "chains": ["A"],
         "cofactors": []
       }
     ]
   }

Example ``ligand.json``:

.. code-block:: json

   {
     "ligands": [
       {
         "id": "erlotinib",
         "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C"
       },
       {
         "id": "gefitinib",
         "smiles": "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F"
       }
     ]
   }

Run with split inputs:

.. code-block:: bash

   prodock \
     --config config.json \
     --receptor-json receptor.json \
     --ligand-json ligand.json

If ``--receptor-json`` or ``--ligand-json`` is not supplied, the CLI falls back
to embedded ``receptors`` or ``ligands`` from ``--config``.

Python automation
-----------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <path d="M8 8h8v8H8z"/>
           <path d="M3 12h5M16 12h5M12 3v5M12 16v5"/>
         </svg>
       </div>
       <div>
         <h2>prodock()</h2>
         <p>Run the same automated workflow directly from Python.</p>
       </div>
     </div>
   </div>

Use :func:`prodock` when you want the fastest end-to-end workflow from Python.

Minimal run:

.. code-block:: python

   from prodock import prodock

   result = prodock(
       "Quick_Run",
       receptors=RECEPTORS,
       ligands=LIGANDS,
   )

   print(result.campaign_json)

Interaction-aware run:

.. code-block:: python

   from prodock import prodock

   result = prodock(
       "Interaction_Run",
       receptors=RECEPTORS,
       ligands=LIGANDS,
       engines=["qvina", "qvina-w"],
       extract_interaction=True,
   )

   print(result.pose_df.head())
   print(result.merged_df.head())

Database-focused run:

.. code-block:: python

   from prodock import prodock

   result = prodock(
       "Database_Run",
       receptors=RECEPTORS,
       ligands=LIGANDS,
       engines=["qvina", "qvina-w"],
       extract_interaction=True,
       save_to_database=True,
       db_name="results.db",
   )

   print(result.db_path)

.. raw:: html

   <div class="pd-mini-callout">
     <strong>See API:</strong>
     <a href="../api/core.html">Core automation API reference</a>
   </div>

Prepared-input modes
--------------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <rect x="4" y="5" width="16" height="14" rx="2"/>
           <path d="M8 9h8M8 13h5"/>
           <circle cx="16.5" cy="13.5" r="1.5"/>
         </svg>
       </div>
       <div>
         <h2>Prepared receptors and ligand directories</h2>
         <p>Skip preprocessing when docking-ready inputs already exist.</p>
       </div>
     </div>
   </div>

Automation also supports already prepared inputs.

Prepared receptor mode:

.. code-block:: json

   {
     "project_dir": "DemoPrepared",
     "prepared_receptors": [
       {
         "receptor_id": "4WKQ",
         "receptor_pdbqt": "prepared/4WKQ/4WKQ.pdbqt",
         "center": [5.0, 10.0, 12.0],
         "size": [20.0, 20.0, 20.0]
       }
     ],
     "ligands": [
       {
         "id": "erlotinib",
         "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C"
       }
     ],
     "config": {
       "engines": ["qvina"],
       "save_to_database": true
     }
   }

Ligand directory mode:

.. code-block:: json

   {
     "project_dir": "DemoLigandDir",
     "receptors": [
       {
         "pdb_id": "4WKQ",
         "receptor_name": "EGFR_4WKQ",
         "ligand_code": "IRE",
         "chains": ["A"],
         "cofactors": []
       }
     ],
     "ligand_dir": "prepared_ligands",
     "config": {
       "engines": ["qvina", "vina"],
       "extract_interaction": false
     }
   }

Python prepared-input example:

.. code-block:: python

   from prodock import prodock

   result = prodock(
       "Prepared_Run",
       prepared_receptors=[
           {
               "receptor_id": "4WKQ",
               "receptor_pdbqt": "Prepared_Run/4WKQ/filtered_protein/4WKQ.pdbqt",
               "center": (2.865, 193.257, 21.367),
               "size": (27.091, 27.091, 27.091),
           }
       ],
       ligand_dir="Prepared_Run/ligands",
       engines=["qvina"],
       extract_interaction=True,
       save_to_database=True,
       db_name="prepared.db",
   )

   print(result.db_path)

CLI overrides
-------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <path d="M5 8h14"/>
           <path d="M5 12h14"/>
           <path d="M5 16h14"/>
           <circle cx="8" cy="8" r="1"/>
           <circle cx="8" cy="12" r="1"/>
           <circle cx="8" cy="16" r="1"/>
         </svg>
       </div>
       <div>
         <h2>Override config values from the terminal</h2>
         <p>Keep stable defaults in JSON and vary runtime behavior with CLI flags.</p>
       </div>
     </div>
   </div>

Override engines and compute settings:

.. code-block:: bash

   prodock \
     --config config.json \
     --receptor-json receptor.json \
     --ligand-json ligand.json \
     --engines qvina smina vina \
     --cpu 8 \
     --n-jobs 8 \
     --exhaustiveness 16 \
     --n-poses 20

Boolean workflow flags support both positive and negative forms:

.. code-block:: bash

   prodock --config run.json --progress
   prodock --config run.json --no-progress

   prodock --config run.json --extract-interaction
   prodock --config run.json --no-extract-interaction

   prodock --config run.json --save-to-database
   prodock --config run.json --no-save-to-database

   prodock --config run.json --replace
   prodock --config run.json --no-replace

Interaction-focused overrides:

.. code-block:: bash

   prodock \
     --config run.json \
     --extract-interaction \
     --interaction-batch-size 8 \
     --interaction-n-jobs 4 \
     --interaction-progress \
     --include-fingerprint-columns \
     --include-interaction-events

Use the InteractionProfiler backend explicitly:

.. code-block:: bash

   prodock \
     --config run.json \
     --extract-interaction \
     --use-interaction-profiler

Database-focused overrides:

.. code-block:: bash

   prodock \
     --config run.json \
     --save-to-database \
     --db-name demo.db \
     --replace \
     --replace-interactions

Validation and reproducibility
------------------------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <path d="M4 18h16"/>
           <path d="M7 18V9"/>
           <path d="M12 18V6"/>
           <path d="M17 18v-4"/>
         </svg>
       </div>
       <div>
         <h2>Validate, inspect, and save the merged config</h2>
         <p>Check the final resolved workflow before running the campaign.</p>
       </div>
     </div>
   </div>

The CLI can validate merged inputs without running docking:

.. code-block:: bash

   prodock \
     --config config.json \
     --receptor-json receptor.json \
     --ligand-json ligand.json \
     --validate-only

Print the final merged effective configuration:

.. code-block:: bash

   prodock \
     --config config.json \
     --receptor-json receptor.json \
     --ligand-json ligand.json \
     --print-effective-config

Write the effective configuration to disk:

.. code-block:: bash

   prodock \
     --config config.json \
     --receptor-json receptor.json \
     --ligand-json ligand.json \
     --effective-config-json effective.json

Write a compact run summary:

.. code-block:: bash

   prodock \
     --config run.json \
     --summary-json summary.json

Show a traceback on errors:

.. code-block:: bash

   prodock --config run.json --traceback

These options are especially useful for debugging configuration merges, keeping
reproducible campaign records, and capturing exactly what was executed.

Path resolution notes
---------------------

Relative paths inside each JSON file are resolved relative to the directory of
that JSON file.

Relative paths passed to:

- ``--summary-json``
- ``--effective-config-json``

are resolved relative to the main ``--config`` directory.

Minimal end-to-end example
--------------------------

.. raw:: html

   <div class="pd-example-head">
     <div class="pd-example-badge">Example</div>
     <h2>Run one complete automated campaign from split JSON files</h2>
   </div>

.. code-block:: bash

   prodock \
     --config config.json \
     --receptor-json receptor.json \
     --ligand-json ligand.json \
     --engines qvina qvina-w \
     --extract-interaction \
     --save-to-database \
     --db-name results.db \
     --effective-config-json effective.json \
     --summary-json summary.json

See also
--------

- :doc:`../api/core` — full reference for the high-level automation entry points
- :doc:`../tutorial/preprocess` — prepare inputs explicitly before docking
- :doc:`../tutorial/dock` — run single or batch docking directly
- :doc:`../tutorial/postprocess` — analyze docking outputs after the campaign
- :doc:`../tutorial/database` — store and query campaigns in SQLite