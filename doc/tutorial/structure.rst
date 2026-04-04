Structure
=========

.. raw:: html

   <div class="pd-panel pd-panel-hero">
     <div class="pd-eyebrow">Structure intake</div>
     <h2>Retrieve, clean, and organize experimental protein structures for docking</h2>
     <p>
       The <strong>prodock.structure</strong> module converts raw experimental PDB entries
       into clean receptor and ligand artifacts for downstream docking workflows.
     </p>
   </div>

Workflow at a glance
--------------------

.. raw:: html

   <div class="pd-diagram-wrap">
     <div class="pd-diagram-stage">

       <div class="pd-diagram-main">
         <div class="pd-node pd-node-main pd-node-input">
           <div class="pd-node-kicker">Input</div>
           <h3>PDB entry</h3>
           <p>Experimental structure identifier such as <code>1M17</code>.</p>
         </div>

         <div class="pd-link pd-link-arrow"></div>

         <div class="pd-node pd-node-main">
           <div class="pd-node-kicker">Step 1</div>
           <h3>Fetch structure</h3>
           <p>Download the PDB file and initialize a local receptor workspace.</p>
         </div>

         <div class="pd-link pd-link-arrow"></div>

         <div class="pd-node pd-node-main">
           <div class="pd-node-kicker">Step 2</div>
           <h3>Filter chains</h3>
           <p>Retain only the selected receptor chains for downstream preparation.</p>
         </div>

         <div class="pd-link pd-link-arrow"></div>

         <div class="pd-node pd-node-main pd-node-branch-source">
           <div class="pd-node-kicker">Step 3</div>
           <h3>Extract ligand</h3>
           <p>Save the bound ligand as a reusable structural reference.</p>
           <div class="pd-branch-arrow"></div>
         </div>

         <div class="pd-link pd-link-arrow"></div>

         <div class="pd-node pd-node-main">
           <div class="pd-node-kicker">Step 4</div>
           <h3>Clean structure</h3>
           <p>Remove solvent residues while preserving selected cofactors.</p>
         </div>

         <div class="pd-link pd-link-arrow"></div>

         <div class="pd-node pd-node-main pd-node-output">
           <div class="pd-node-kicker">Output</div>
           <h3>Filtered receptor</h3>
           <p>Clean receptor ready for preprocessing and docking.</p>
         </div>
       </div>

       <div class="pd-branch-output">
         <div class="pd-node pd-node-side">
           <div class="pd-node-kicker">Ligand output</div>
           <h3>Reference ligand</h3>
           <p><code>reference_ligand/&lt;ligand_code&gt;.sdf</code></p>
         </div>
       </div>

     </div>
   </div>

Main objects
------------

.. raw:: html

   <div class="pd-legend-grid">
     <div class="pd-legend-box">
       <div class="pd-legend-icon">⚙️</div>
       <div>
         <h3>PDBEngine</h3>
         <p>Step-wise backend engine for one receptor structure.</p>
       </div>
     </div>
     <div class="pd-legend-box">
       <div class="pd-legend-icon">📦</div>
       <div>
         <h3>PDBQuery</h3>
         <p>Thin public wrapper for single-entry and batch-oriented workflows.</p>
       </div>
     </div>
     <div class="pd-legend-box">
       <div class="pd-legend-icon">🧼</div>
       <div>
         <h3>PDBQTSanitizer</h3>
         <p>Backend-aware validator and sanitizer for generated PDBQT files.</p>
       </div>
     </div>
   </div>

PDBEngine
---------

``PDBEngine`` is the main step-wise backend for preparing a PDB structure for downstream use.

It typically performs:

- validation of runtime requirements and canonical output paths
- fetching the structure into a local working directory
- filtering the structure to selected chains
- extracting the requested ligand as reference and co-crystal files
- removing solvent residues while optionally preserving cofactors
- saving the filtered protein structure

Typical example
~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from prodock.structure.pdb_engine import PDBEngine

   engine = (
       PDBEngine(
           pdb_id="1M17",
           base_out=Path("tutorial/1M17"),
           chains=["A"],
           ligand_code="AQ4",
           cofactors=[],
       )
       .run_all()
   )

   print(engine.filtered_path)
   print(engine.ref_path)
   print(engine.cocrystal_path)

Step-wise usage
~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from prodock.structure.pdb_engine import PDBEngine

   engine = PDBEngine(
       pdb_id="1M17",
       base_out=Path("tutorial/1M17"),
       chains=["A"],
       ligand_code="AQ4",
       cofactors=[],
   )

   (
       engine.validate()
             .fetch()
             .filter_chains()
             .extract_ligand()
             .clean_solvents_and_cofactors()
             .save_filtered_protein()
   )

Generated outputs
~~~~~~~~~~~~~~~~~

- ``fetched_protein/<pdb_id>.pdb``
- ``filtered_protein/<pdb_id>.pdb``
- ``reference_ligand/<ligand_code>.sdf``
- ``cocrystal/<pdb_id>.sdf``

PDBQuery
--------

``PDBQuery`` is a thin public wrapper around ``PDBEngine``. It preserves a compact,
backward-compatible public API while delegating the actual preparation logic to the
underlying engine.

Batch example
~~~~~~~~~~~~~

.. code-block:: python

   from prodock.structure.pdb_query import PDBQuery

   PROJECT = "tutorial"

   RECEPTORS = [
       {
           "pdb_id": "1M17",
           "receptor_name": "EGFR_1M17",
           "ligand_code": "AQ4",
           "chains": ["A"],
           "cofactors": [],
       },
       {
           "pdb_id": "2ITY",
           "receptor_name": "EGFR_2ITY",
           "ligand_code": "IRE",
           "chains": ["A"],
           "cofactors": [],
       },
       {
           "pdb_id": "4WKQ",
           "receptor_name": "EGFR_4WKQ",
           "ligand_code": "IRE",
           "chains": ["A"],
           "cofactors": [],
       },
   ]

   PDBQuery.process_batch(RECEPTORS, output_dir=PROJECT)

Single-receptor example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from prodock.structure import PDBQuery

   pq = PDBQuery(
       pdb_id="1M17",
       output_dir="tutorial/1M17",
       chains=["A"],
       ligand_code="AQ4",
       cofactors=[],
   )

   pq.run_all()

   print(pq.filtered_protein_path)
   print(pq.reference_ligand_path)
   print(pq.cocrystal_ligand_path)

PDBQTSanitizer
--------------

``PDBQTSanitizer`` is a backend-aware validator and sanitizer for PDBQT files.

It is useful when a generated ``.pdbqt`` contains non-canonical element fields,
atom-type-like trailing tokens, or formatting patterns that may cause failures
across docking backends.

Example
~~~~~~~

.. code-block:: python

   from prodock.structure import PDBQTSanitizer

   sanitizer = PDBQTSanitizer("ligand.pdbqt", backend="meeko")
   warnings = sanitizer.validate(strict=True)
   sanitizer.sanitize(rebuild=True, aggressive=False)
   sanitizer.write("ligand.sanitized.pdbqt")

In-place sanitization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from prodock.structure import PDBQTSanitizer

   sanitizer = PDBQTSanitizer("ligand.pdbqt", backend="obabel")
   sanitizer.sanitize_inplace(rebuild=True, aggressive=False, backup=True)

Summary
-------

The structure module provides the earliest stage of a ProDock workflow:

- ``PDBEngine`` for explicit step-wise structure preparation
- ``PDBQuery`` for compact public and batch interfaces
- ``PDBQTSanitizer`` for validating and normalizing generated PDBQT files