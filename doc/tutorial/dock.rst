
Dock
====

.. image:: ../_static/tutorial-dock.svg
   :alt: docking workflow
   :class: pd-visual

What this stage does
--------------------

- Build tasks from receptors, ligands, and engines.
- Run single jobs or campaign-style batches.
- Keep engine selection registry-based.

Typical example
---------------

.. code-block:: python

   from prodock.dock import BatchDock

   runner = BatchDock(n_jobs=4, progress=True)
   results = runner.run_from_config("campaign.json")

Main objects
------------

- ``SingleDock``
- ``BatchDock`` / ``MatrixDock``
- ``VinaEngine`` / ``SminaEngine`` / ``GninaEngine`` / ``QVinaEngine`` / ``QVinaWEngine``
