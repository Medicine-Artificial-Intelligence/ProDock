Preprocess
==========

.. image:: ../_static/tutorial-preprocess.svg
   :alt: preprocess workflow
   :class: pd-visual


.. raw:: html

   <div class="pd-card-grid pd-card-grid-3">
     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="7" cy="12" r="2.1"/>
           <circle cx="12" cy="7" r="2.1"/>
           <circle cx="17" cy="12" r="2.1"/>
           <circle cx="12" cy="17" r="2.1"/>
           <path d="M8.7 10.3l1.6-1.6M13.7 8.7l1.6 1.6M15.3 13.7l-1.6 1.6M10.3 15.3l-1.6-1.6"/>
         </svg>
       </div>
       <h3>Ligand preparation</h3>
       <p>
         Build 3D ligand structures from SMILES and export docking-ready files
         such as SDF, PDB, or PDBQT.
       </p>
     </div>

     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <path d="M12 3.5l6.5 3.2v4.8c0 4.4-2.9 7.2-6.5 9-3.6-1.8-6.5-4.6-6.5-9V6.7L12 3.5z"/>
           <path d="M9.4 12.1l1.8 1.8 3.7-4.2"/>
         </svg>
       </div>
       <h3>Receptor preparation</h3>
       <p>
         Repair, minimize, clean, and convert receptor structures into
         docking-ready outputs for downstream engines.
       </p>
     </div>

     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <path d="M5 8.5l7-4 7 4v7l-7 4-7-4v-7z"/>
           <path d="M12 4.5v15"/>
           <path d="M5 8.5l7 4 7-4"/>
         </svg>
       </div>
       <h3>Grid box definition</h3>
       <p>
         Estimate the docking search region from ligand coordinates and export
         Vina-compatible box parameters.
       </p>
     </div>
   </div>

Ligand preparation
------------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="7" cy="12" r="2.1"/>
           <circle cx="12" cy="7" r="2.1"/>
           <circle cx="17" cy="12" r="2.1"/>
           <circle cx="12" cy="17" r="2.1"/>
           <path d="M8.7 10.3l1.6-1.6M13.7 8.7l1.6 1.6M15.3 13.7l-1.6 1.6M10.3 15.3l-1.6-1.6"/>
         </svg>
       </div>
       <div>
         <h2>LigandPrep</h2>
         <p>Convert SMILES into 3D ligand structures for docking workflows.</p>
       </div>
     </div>
   </div>

Use :class:`LigandPrep` when you want to prepare ligands from SMILES input and
export them into standard formats such as ``SDF``, ``PDB``, or ``PDBQT``.

Typical use cases include:

- preparing small ligand batches from lists,
- reading ligands from tables or DataFrames,
- generating docking-ready PDBQT files,
- keeping generated MolBlocks in memory.

.. code-block:: python

   from prodock.preprocess import LigandPrep

   ligands = (
       LigandPrep(output_dir="ligands")
       .from_smiles_list(
           [
               "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
               "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
           ],
           names=["gefitinib", "erlotinib"],
       )
       .set_output_format("pdbqt")
       .process_all()
   )

   print(ligands.summary)
   print(ligands.output_paths)

.. raw:: html

   <div class="pd-mini-callout">
     <strong>See API:</strong>
     <a href="../api/preprocess.html#ligandprep">LigandPrep reference</a>
   </div>

Receptor preparation
--------------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <path d="M12 3.5l6.5 3.2v4.8c0 4.4-2.9 7.2-6.5 9-3.6-1.8-6.5-4.6-6.5-9V6.7L12 3.5z"/>
           <path d="M9.4 12.1l1.8 1.8 3.7-4.2"/>
         </svg>
       </div>
       <div>
         <h2>ReceptorPrep</h2>
         <p>Clean, minimize, and export a receptor into a docking-ready artifact.</p>
       </div>
     </div>
   </div>

Use :class:`ReceptorPrep` when you start from a receptor ``PDB`` and want a
prepared output for docking.

The high-level workflow handles:

- receptor fixing,
- minimization,
- conversion to ``PDB`` or ``PDBQT``,
- fallback handling when one preparation route fails.

.. code-block:: python

   from prodock.preprocess import ReceptorPrep

   receptor = ReceptorPrep().prep(
       input_pdb="EGFR_1M17.pdb",
       output_dir="receptor_out",
       out_fmt="pdbqt",
   )

   print(receptor.final_artifact)
   print(receptor.last_simulation_report)

.. raw:: html

   <div class="pd-mini-callout">
     <strong>See API:</strong>
     <a href="../api/preprocess.html#receptorprep">ReceptorPrep reference</a>
   </div>

Grid box computation
--------------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <path d="M5 8.5l7-4 7 4v7l-7 4-7-4v-7z"/>
           <path d="M12 4.5v15"/>
           <path d="M5 8.5l7 4 7-4"/>
         </svg>
       </div>
       <div>
         <h2>GridBox</h2>
         <p>Define the docking search region from ligand geometry.</p>
       </div>
     </div>
   </div>

Use :class:`GridBox` when you want to derive a docking box from one ligand or
from several reference ligands.

This is commonly used for:

- reference-ligand-guided docking,
- reproducible docking box generation,
- Vina-compatible center and size export.

.. code-block:: python

   from prodock.preprocess import GridBox

   box = (
       GridBox()
       .load_ligand("AQ4.sdf")
       .from_ligand_pad(pad=4.0, isotropic=False)
   )

   print(box.center)
   print(box.size)
   print(box.to_vina_lines())

.. raw:: html

   <div class="pd-mini-callout">
     <strong>See API:</strong>
     <a href="../api/preprocess.html#gridbox">GridBox reference</a>
   </div>

Minimal end-to-end example
--------------------------

.. raw:: html

   <div class="pd-example-head">
     <div class="pd-example-badge">Example</div>
     <h2>Preprocess a full docking setup</h2>
   </div>

.. code-block:: python

   from prodock.preprocess import LigandPrep, ReceptorPrep, GridBox

   ligands = (
       LigandPrep(output_dir="project/ligands")
       .from_smiles_list(
           ["CCO", "c1ccccc1"],
           names=["ethanol", "benzene"],
       )
       .set_output_format("pdbqt")
       .process_all()
   )

   receptor = ReceptorPrep().prep(
       input_pdb="EGFR_1M17.pdb",
       output_dir="project/receptor",
       out_fmt="pdbqt",
   )

   box = (
       GridBox()
       .load_ligand("AQ4.sdf")
       .from_ligand_pad(pad=4.0, isotropic=False)
       .snap(step=0.25)
   )

   print("Ligands:", ligands.summary)
   print("Receptor:", receptor.final_artifact)
   print("Box:")
   print(box.to_vina_lines())



API and next steps
------------------

.. raw:: html

   <div class="pd-link-grid">
     <a class="pd-link-tile" href="../api/preprocess.html">
       <strong>Preprocess API</strong>
       <span>Full reference for LigandPrep, ReceptorPrep, and GridBox.</span>
     </a>
     <a class="pd-link-tile" href="../api/structure.html">
       <strong>Structure API</strong>
       <span>Low-level conversion and structure helpers used during preprocessing.</span>
     </a>
     <a class="pd-link-tile" href="../tutorials/docking.html">
       <strong>Docking tutorial</strong>
       <span>Continue to docking after preparing ligands, receptor, and box.</span>
     </a>
   </div>

See also
--------

- :doc:`../api/preprocess` — full reference for ``LigandPrep``, ``ReceptorPrep``, and ``GridBox``
- :doc:`../api/structure` — low-level conversion and structure utilities
- :doc:`../tutorial/dock` — continue to the docking stage