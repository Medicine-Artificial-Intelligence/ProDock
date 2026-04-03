
Database
========

.. image:: ../_static/tutorial-database.svg
   :alt: database workflow
   :class: pd-visual

Why store results?
------------------

- Avoid repeated pose conversion.
- Keep scores and interactions normalized.
- Query by receptor, ligand, engine, or rank later.

Typical example
---------------

.. code-block:: python

   from prodock.database import PoseDatabase

   db = PoseDatabase("poses.sqlite")
   db.insert_dataframe(pose_dataframe)

Visual schema
-------------

.. image:: ../_static/db-architecture.svg
   :alt: database overview
   :class: pd-visual

Original detailed schema
------------------------

.. image:: ../fig/db-schema-original.png
   :alt: original full schema
   :class: pd-visual
