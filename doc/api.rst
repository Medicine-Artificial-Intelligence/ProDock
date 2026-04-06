API
===

Quick access
------------

.. raw:: html

   <div class="pd-grid-3">
     <a class="pd-card pd-card-link" href="tutorial/automation.html">
       <div class="pd-kicker">⚙️ Core</div>
       <h3>Pipeline entry points</h3>
       <p><code>ProDockPipeline</code><br><code>prodock()</code></p>
     </a>
     <a class="pd-card pd-card-link" href="tutorial.html">
       <div class="pd-kicker">🧩 Modules</div>
       <h3>Stage-specific APIs</h3>
       <p><code>structure</code> · <code>preprocess</code><br><code>dock</code> · <code>postprocess</code> · <code>database</code></p>
     </a>
     <a class="pd-card pd-card-link" href="api/full.html">
       <div class="pd-kicker">📘 Reference</div>
       <h3>Complete API page</h3>
       <p>Grouped <code>automodule</code><br>documentation</p>
     </a>
   </div>

Main entry points
-----------------

**⚙️ Core**
   - :class:`prodock.core.ProDockPipeline`
   - :func:`prodock.core.prodock`

**🧬 Structure**
   - :class:`prodock.structure.conversion`
   - :class:`prodock.structure.pdb_query.PDBQuery`
   - :class:`prodock.structure.pdb_engine.PDBEngine`
   - :class:`prodock.structure.pdbqt_sanitizer.PDBQTSanitizer`
  

**🧪 Preprocess**
   - :class:`prodock.preprocess.ligand.prep.LigandPrep`
   - :class:`prodock.preprocess.receptor.prep.ReceptorPrep`
   - :class:`prodock.preprocess.gridbox.gridbox.GridBox`

**🚀 Dock**
   - :class:`prodock.dock.single.SingleDock`
   - :class:`prodock.dock.batch.BatchDock`

**📊 Postprocess**
   - :class:`prodock.postprocess.extract.core.Extractor`
   - :class:`prodock.postprocess.pose.core.PoseCrawler`
   - :class:`prodock.postprocess.interaction.core.InteractionProfiler`

**🗄️ Database**
   - :class:`prodock.database.pose_db.PoseDatabase`
   - :class:`prodock.database.pose_query.PoseQuery`

.. toctree::
   :maxdepth: 1
   :hidden:

   api/full