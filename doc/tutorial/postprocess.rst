Postprocess
===========

.. image:: ../_static/tutorial-postprocess.svg
   :alt: postprocess workflow
   :class: pd-visual


.. raw:: html

   <div class="pd-card-grid pd-card-grid-2">
     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <path d="M5 19V5"/>
           <path d="M5 19h14"/>
           <path d="M8 15l2.2-3.2 2.2 1.8 3.6-5.1"/>
           <circle cx="8" cy="15" r="1"/>
           <circle cx="10.2" cy="11.8" r="1"/>
           <circle cx="12.4" cy="13.6" r="1"/>
           <circle cx="16" cy="8.5" r="1"/>
         </svg>
       </div>
       <h3>Score extraction</h3>
       <p>
         Parse docking logs and score tables into standardized dataframes for
         ranking, filtering, and downstream analysis.
       </p>
     </div>

     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <path d="M6 6h12v12H6z"/>
           <path d="M9 9h6v6H9z"/>
           <path d="M6 3v3M18 3v3M6 18v3M18 18v3"/>
         </svg>
       </div>
       <h3>Pose crawling</h3>
       <p>
         Discover docked pose files, summarize them into tables, convert them,
         and load molecules for later analysis.
       </p>
     </div>

     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="7" cy="7" r="2"/>
           <circle cx="17" cy="7" r="2"/>
           <circle cx="7" cy="17" r="2"/>
           <circle cx="17" cy="17" r="2"/>
           <path d="M9 7h6M7 9v6M17 9v6M9 17h6"/>
         </svg>
       </div>
       <h3>Interaction profiling</h3>
       <p>
         Compute protein–ligand contacts, fingerprints, and pose-level
         interaction summaries from one structure or a full pose table.
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
       <h3>Metrics and evaluation</h3>
       <p>
         Measure pose quality and virtual screening performance with RMSD,
         AUC, enrichment, BEDROC, and related metrics.
       </p>
     </div>
   </div>

Score extraction
----------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <path d="M5 19V5"/>
           <path d="M5 19h14"/>
           <path d="M8 15l2.2-3.2 2.2 1.8 3.6-5.1"/>
           <circle cx="8" cy="15" r="1"/>
           <circle cx="10.2" cy="11.8" r="1"/>
           <circle cx="12.4" cy="13.6" r="1"/>
           <circle cx="16" cy="8.5" r="1"/>
         </svg>
       </div>
       <div>
         <h2>Extractor</h2>
         <p>Parse docking logs or score tables and normalize them into canonical result tables.</p>
       </div>
     </div>
   </div>

Use :func:`extract_scores` for a simple one-call workflow, or
:class:`Extractor` when you want more explicit control over layouts,
engine filtering, or recursive log discovery.

Typical use cases include:

- parsing one or more engine-specific log trees,
- extracting scores from mixed log or table inputs,
- filtering extracted tables by engine,
- building canonical score dataframes for downstream ranking.

.. code-block:: python

   from prodock.postprocess import extract_scores, Extractor

   scores = extract_scores(
       roots=["logs"],
       layout="engine_tree",
   )

   extractor = Extractor(match_mode="exact")
   vina_scores = extractor.extract_scores(
       roots=["logs"],
       engines=["vina"],
       layout="engine_tree",
   )

   print(scores.head())
   print(vina_scores.head())

.. raw:: html

   <div class="pd-mini-callout">
     <strong>See API:</strong>
     <a href="../api/postprocess.html#module-prodock.postprocess.extract.core">Score Extraction API reference</a>
   </div>

Pose crawling
-------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <path d="M6 6h12v12H6z"/>
           <path d="M9 9h6v6H9z"/>
           <path d="M6 3v3M18 3v3M6 18v3M18 18v3"/>
         </svg>
       </div>
       <div>
         <h2>PoseCrawler</h2>
         <p>Discover pose files, summarize them, select best poses, and convert them into reusable structures.</p>
       </div>
     </div>
   </div>

Use :class:`PoseCrawler` when you want to work with docked pose trees after a
campaign has finished.

Typical use cases include:

- crawling a ProDock pose tree,
- loading pose metadata into a dataframe,
- selecting best-scoring poses per ligand or engine,
- converting discovered ``PDBQT`` poses into ``SDF``.

.. code-block:: python

   from prodock.postprocess.pose import PoseCrawler

   crawler = PoseCrawler(["results"])

   pose_df = crawler.crawl()
   best_df = crawler.best()

   mol_df = crawler.crawl_mols(
       backend="obabel",
       save_sdf=True,
   )

   sdf_paths = crawler.convert(
       out_dir="converted_sdf",
       overwrite=True,
   )

   print(pose_df.head())
   print(best_df.head())
   print(mol_df.head())
   print(sdf_paths[:3])

.. raw:: html

   <div class="pd-mini-callout">
     <strong>See API:</strong>
     <a href="../api/postprocess.html#module-prodock.postprocess.pose.core">Pose Extraction API reference</a>
   </div>

Interaction profiling
---------------------

.. raw:: html

   <div class="pd-panel pd-panel-soft">
     <div class="pd-section-head">
       <div class="pd-section-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="7" cy="7" r="2"/>
           <circle cx="17" cy="7" r="2"/>
           <circle cx="7" cy="17" r="2"/>
           <circle cx="17" cy="17" r="2"/>
           <path d="M9 7h6M7 9v6M17 9v6M9 17h6"/>
         </svg>
       </div>
       <div>
         <h2>InteractionProfiler</h2>
         <p>Compute protein–ligand interaction fingerprints and pose-level summaries.</p>
       </div>
     </div>
   </div>

Use :class:`InteractionProfiler` for one receptor–ligand workflow, or use
:func:`extract_pose_table_interactions` when you already have a pose dataframe
from :class:`PoseCrawler`.

Typical use cases include:

- running interaction analysis for one receptor and one ligand source,
- profiling interactions for a full pose table,
- saving pose-level interaction summaries,
- preparing interaction results for database storage or notebook analysis.

.. code-block:: python

   from prodock.postprocess.pose import PoseCrawler
   from prodock.postprocess.interaction.core import extract_pose_table_interactions

   pose_table = PoseCrawler(["Data/testcase/post"]).crawl_mols()

   receptor_map = {
       "1M17": "Data/testcase/post/1M17/receptor/1M17.pdb",
   }

   pose_result = extract_pose_table_interactions(
       poses=pose_table,
       receptor_pdb_by_id=receptor_map,
       progress=False,
       ultra_safe=True,
   )

   print(pose_result.merged_df.head())
   print(pose_result.summary_df.head())

Single-run interaction analysis is also available:

.. code-block:: python

   from prodock.postprocess.interaction.core import InteractionProfiler

   profiler = InteractionProfiler()
   result = profiler.run(
       receptor_pdb="Data/receptor/egfr.pdb",
       ligands="ligands.sdf",
   )

   print(result.fingerprint_df.head())
   print(result.interaction_df.head())

.. raw:: html

   <div class="pd-mini-callout">
     <strong>See API:</strong>
     <a href="../api/postprocess.html#module-prodock.postprocess.interaction.core">Interaction Extraction API reference</a>
   </div>

Metrics and evaluation
----------------------

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
         <h2>DockEvaluator and ScreenEvaluator</h2>
         <p>Measure pose quality and virtual screening performance with reusable metric helpers.</p>
       </div>
     </div>
   </div>

Use :class:`DockEvaluator` for RMSD-based pose evaluation and
:class:`ScreenEvaluator` for ranking and enrichment metrics.

Typical use cases include:

- comparing docked poses against a reference structure,
- computing screening metrics from ranked score lists,
- measuring early recognition with BEDROC,
- using functional wrappers for quick analysis scripts.

.. code-block:: python

   from prodock.postprocess import DockEvaluator, ScreenEvaluator

   dock_eval = DockEvaluator(engine="rdkit")
   rmsd = dock_eval.rmsd("reference.sdf", "pose.sdf")

   screen_eval = ScreenEvaluator(higher_is_better=False)

   scores = [-10.0, -9.0, -8.0, -7.0, -2.0, -1.0]
   labels = [1, 0, 1, 0, 0, 0]

   print(rmsd)
   print(screen_eval.auc_roc(scores, labels))
   print(screen_eval.pr_auc(scores, labels))
   print(screen_eval.enrichment_factor(scores, labels, fraction=0.1))

Convenience wrappers are also available directly from
:mod:`prodock.postprocess`, including ``rmsd_aligned``, ``rmsd_min``,
``auc_roc``, ``pr_auc``, ``enrichment_factor``, ``bedroc``,
``topn_success``, and ``success_rate``.

.. raw:: html

   <div class="pd-mini-callout">
     <strong>See API:</strong>
     <a href="../api/postprocess.html#id39">Extraction API reference</a>
   </div>

Minimal end-to-end example
--------------------------

.. raw:: html

   <div class="pd-example-head">
     <div class="pd-example-badge">Example</div>
     <h2>Postprocess a completed docking campaign</h2>
   </div>

.. code-block:: python

   from prodock.postprocess import extract_scores
   from prodock.postprocess.pose import PoseCrawler
   from prodock.postprocess.interaction.core import extract_pose_table_interactions

   # 1. Parse docking scores
   score_df = extract_scores(
       roots=["logs"],
       layout="engine_tree",
   )

   # 2. Discover pose files and load molecules
   pose_df = PoseCrawler(["results"]).crawl_mols()

   # 3. Map receptor ids to receptor PDB files
   receptor_map = {
       "1M17": "Data/testcase/post/1M17/receptor/1M17.pdb",
   }

   # 4. Compute interaction tables
   pose_result = extract_pose_table_interactions(
       poses=pose_df,
       receptor_pdb_by_id=receptor_map,
       progress=False,
       ultra_safe=True,
   )

   print(score_df.head())
   print(pose_result.summary_df.head())

See also
--------

- :doc:`../api/postprocess` — full reference for extraction, pose crawling, interaction analysis, and metrics
- :doc:`../api/structure` — low-level conversion and structure utilities used during postprocessing
- :doc:`../tutorial/dock` — continue from docking outputs into analysis workflows