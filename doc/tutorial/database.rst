Database
========

.. image:: ../_static/tutorial-database.svg
   :alt: ProDock database workflow
   :class: pd-visual


.. raw:: html

   <div class="pd-card-grid pd-card-grid-2">
     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <ellipse cx="12" cy="5" rx="6.5" ry="2.8"/>
           <path d="M5.5 5v6c0 1.5 2.9 2.8 6.5 2.8s6.5-1.3 6.5-2.8V5"/>
           <path d="M5.5 11v6c0 1.5 2.9 2.8 6.5 2.8s6.5-1.3 6.5-2.8v-6"/>
         </svg>
       </div>
       <h3>Campaign storage</h3>
       <p>
         Store receptors, ligands, engines, poses, scores, and interactions in
         one SQLite database instead of scattered output files.
       </p>
     </div>

     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="11" cy="11" r="5.5"/>
           <path d="M16 16l4 4"/>
           <path d="M8.8 11h4.4M11 8.8v4.4"/>
         </svg>
       </div>
       <h3>Query analysis</h3>
       <p>
         Reopen the campaign later and filter poses by receptor, ligand, engine,
         rank, affinity, or interaction content without rebuilding results.
       </p>
     </div>
   </div>

Database architecture
---------------------

ProDock stores docking campaigns in a compact normalized SQLite schema. The
main design idea is simple:

- write results once,
- query them many times later.

The schema is organized so that receptor, ligand, and engine identifiers are
stored once, while pose-specific, score-specific, and interaction-specific
records remain linked through stable pose keys.

.. image:: ../_static/db-architecture.svg
   :alt: ProDock database architecture
   :class: pd-visual

This gives three practical benefits:

- compact storage across many receptors, ligands, and engines,
- easy reconstruction of analysis tables,
- consistent filtering across identity, score, and interaction layers.

The database is organized into three layers:

- **dimension tables** for receptors, ligands, and engines,
- **pose records** for pose identity, serialized molecules, and metadata,
- **analysis tables** for score payloads and residue-level interactions.

In practice, this means one campaign can be explored later without reparsing
engine logs, reconverting pose files, or recomputing interaction summaries.

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
         <h2>Writer and reader</h2>
         <p>The database layer is split into one writer API and one read/query API.</p>
       </div>
     </div>
   </div>

The two main public objects are:

- :class:`PoseDatabase` — create the schema and insert or update campaign data
- :class:`PoseQuery` — open an existing database and run analysis-friendly queries

This separation keeps workflow generation and downstream analysis cleanly
decoupled.


Pose identity
-------------

A stored pose can be addressed in two ways:

- ``pose_db_id`` — internal SQLite integer primary key
- ``pose_id`` — optional stable external id such as
  ``"1M17__erlotinib__qvina__pose1"``

If no external ``pose_id`` is stored, a pose is still uniquely identified by:

.. code-block:: text

   (receptor_id, ligand_id, engine, pose_rank)

This makes the schema robust both for automated inserts and for human-readable
campaign exports.

Write campaign data
-------------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <ellipse cx="12" cy="5" rx="6.5" ry="2.8"/>
           <path d="M5.5 5v6c0 1.5 2.9 2.8 6.5 2.8s6.5-1.3 6.5-2.8V5"/>
           <path d="M5.5 11v6c0 1.5 2.9 2.8 6.5 2.8s6.5-1.3 6.5-2.8v-6"/>
         </svg>
       </div>
       <div>
         <h2>PoseDatabase</h2>
         <p>Store pose tables, score rows, molecules, and interactions in one campaign database.</p>
       </div>
     </div>
   </div>

Use :class:`PoseDatabase` when you already have a pose dataframe from docking or
postprocessing and want to persist it for later analysis.

Minimal write example:

.. code-block:: python

   from prodock.database import PoseDatabase

   db = PoseDatabase("poses.sqlite")
   db.insert_dataframe(pose_dataframe)

The dataframe is expected to contain the core pose columns:

- ``receptor_id``
- ``ligand_id``
- ``engine``
- ``pose_rank``
- ``affinity``
- ``mol``

A one-step construction pattern is also available:

.. code-block:: python

   from prodock.database import PoseDatabase

   db = PoseDatabase.from_dataframe(
       "poses.sqlite",
       pose_dataframe,
   )

If interaction payloads are already available, they can be stored at import time:

.. code-block:: python

   db.insert_dataframe(
       pose_dataframe,
       interactions_by_pose=interactions_by_pose,
       replace=True,
       replace_interactions=True,
   )

.. raw:: html

   <div class="pd-mini-callout">
     <strong>See API:</strong>
     <a href="../api/database.html">Database API reference</a>
   </div>

Query stored campaigns
----------------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="11" cy="11" r="5.5"/>
           <path d="M16 16l4 4"/>
           <path d="M8.8 11h4.4M11 8.8v4.4"/>
         </svg>
       </div>
       <div>
         <h2>PoseQuery</h2>
         <p>Open an existing database and query poses, scores, and interactions through a read-focused API.</p>
       </div>
     </div>
   </div>

Use :class:`PoseQuery` after a campaign has already been stored. By default, it
opens the database in read-only mode.

Typical query patterns include:

- pose tables filtered by receptor, ligand, engine, or rank,
- score tables without loading molecules,
- exact retrieval of one stored pose,
- interaction-aware filtering,
- interaction summaries and fingerprint matrices,
- campaign-level summaries.

Basic examples:

.. code-block:: python

   from prodock.database import PoseQuery

   q = PoseQuery("poses.sqlite")

   poses = q.poses(
       receptor_id="1M17",
       engine="qvina",
       as_dataframe=True,
   )

   scores = q.scores(
       receptor_id="1M17",
       top_rank=3,
       as_dataframe=True,
   )

   print(poses[["pose_id", "pose_rank", "affinity"]].head())
   print(scores[["pose_id", "affinity"]].head())

Retrieve one exact pose:

.. code-block:: python

   pose = q.pose(
       receptor_id="1M17",
       ligand_id="erlotinib",
       engine="qvina",
       pose_rank=1,
       include_interactions=True,
       interaction_mode="summary",
   )

   print(pose)

.. raw:: html

   <div class="pd-mini-callout">
     <strong>See API:</strong>
     <a href="../api/database.html">Database API reference</a>
   </div>

Interaction querying
--------------------

One advantage of keeping interactions in the same schema is that pose queries
can also filter by interaction content.

.. code-block:: python

   from prodock.database import PoseQuery

   q = PoseQuery("poses.sqlite")

   selected = q.poses(
       receptor_id="1M17",
       interaction_type="Hydrophobic",
       residue_id="LEU23.A",
       as_dataframe=True,
   )

   print(selected[["pose_id", "affinity"]])

Compact summaries can be rebuilt directly from stored interaction rows:

.. code-block:: python

   summary = q.interaction_summary(
       receptor_id="1M17",
       return_by="pose_id",
   )

   print(summary)

Detailed event payloads can also be reconstructed:

.. code-block:: python

   details = q.interaction_details(
       receptor_id="1M17",
       return_by="pose_key",
   )

   print(details)

Fingerprint matrices are also available:

.. code-block:: python

   fp = q.fingerprint(
       receptor_id="1M17",
       mode="binary",
       index_by="pose_key",
   )

   print(fp.head())


Minimal end-to-end example
--------------------------

.. raw:: html

   <div class="pd-example-head">
     <div class="pd-example-badge">Example</div>
     <h2>Write once, query later</h2>
   </div>

.. code-block:: python

   from prodock.database import PoseDatabase, PoseQuery

   # Step 1: write campaign results
   db = PoseDatabase("poses.sqlite")
   db.insert_dataframe(
       pose_dataframe,
       interactions_by_pose=interactions_by_pose,
       replace=True,
       replace_interactions=True,
   )

   # Step 2: reopen with the read/query API
   q = PoseQuery("poses.sqlite")

   best = q.poses(
       receptor_id="1M17",
       top_rank=1,
       include_interactions=True,
       interaction_mode="summary",
       as_dataframe=True,
   )

   fp = q.fingerprint(
       receptor_id="1M17",
       mode="binary",
       index_by="pose_key",
   )

   summary = q.summary()

   print(best.head())
   print(fp.head())
   print(summary.head())

See also
--------

- :doc:`../api/database` — full reference for ``PoseDatabase`` and ``PoseQuery``
- :doc:`../api/postprocess` — build pose tables and interaction payloads before database import
- :doc:`../tutorial/postprocess` — postprocess docking outputs into reusable tables and summaries