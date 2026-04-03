
Automation
==========

.. image:: ../_static/tutorial-automation.svg
   :alt: automation workflow
   :class: pd-visual

Required input format
---------------------

.. code-block:: python

   RECEPTORS: List[Dict[str, Any]] = [
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
           "ligand_code": "IRE",  # gefitinib (Iressa)
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

   LIGANDS: List[Dict[str, str]] = [
       {
           "id": "erlotinib",
           "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C",
       },
       {
           "id": "gefitinib",
           "smiles": "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F",
       },
   ]

Typical example
---------------

.. code-block:: python

   from prodock.core import ProDockPipeline

   pipeline = ProDockPipeline(project_dir="EGFR_Automation")
   prepared = pipeline.prepare_receptors(RECEPTORS)
   ligands = pipeline.prepare_ligands(LIGANDS)
   campaign = pipeline.build_campaign(prepared, ligands)
   result = pipeline.run(campaign)

Main object
-----------

- ``ProDockPipeline``
