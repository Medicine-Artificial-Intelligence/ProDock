Tutorial
========

.. image:: _static/tutorial-overview.svg
   :alt: ProDock tutorial workflow overview
   :class: pd-visual

.. raw:: html

   <div class="pd-callout pd-callout-soft">
     <div class="pd-callout-title">Workflow guide</div>
     <p>Follow the full ProDock path from structure collection to docking, interaction analysis, and database-backed reuse.</p>
   </div>

.. raw:: html

   <div class="pd-grid-3">
     <a class="pd-card pd-card-icon" href="tutorial/structure.html">
       <div class="pd-kicker">Step 1</div>
       <h3>🧬 Structure</h3>
       <p>Query PDB entries, select chains, extract co-crystallized ligands, and save a clean receptor structure.</p>
     </a>
     <a class="pd-card pd-card-icon" href="tutorial/preprocess.html">
       <div class="pd-kicker">Step 2</div>
       <h3>🧪 Preprocess</h3>
       <p>Prepare protein and ligand inputs, convert formats, and define docking boxes for downstream engines.</p>
     </a>
     <a class="pd-card pd-card-icon" href="tutorial/dock.html">
       <div class="pd-kicker">Step 3</div>
       <h3>🚀 Dock</h3>
       <p>Run single jobs, batch runs, or receptor–ligand–engine campaigns with reusable configurations.</p>
     </a>
     <a class="pd-card pd-card-icon" href="tutorial/postprocess.html">
       <div class="pd-kicker">Step 4</div>
       <h3>📊 Postprocess</h3>
       <p>Extract scores, crawl poses, profile interactions, compare fingerprints, and compute screening metrics.</p>
     </a>
     <a class="pd-card pd-card-icon" href="tutorial/database.html">
       <div class="pd-kicker">Step 5</div>
       <h3>🗄️ Database</h3>
       <p>Store poses, scores, and interactions once, then query them later for ranking, filtering, and reporting.</p>
     </a>
     <a class="pd-card pd-card-icon" href="tutorial/automation.html">
       <div class="pd-kicker">Step 6</div>
       <h3>⚙️ Automation</h3>
       <p>Use <code>ProDockPipeline</code> with structured receptor and ligand dictionaries for end-to-end campaigns.</p>
     </a>
   </div>

.. raw:: html

   <div class="pd-grid-3">
     <div class="pd-mini-card">
       <h4>🧭 Start here</h4>
       <p>The tutorial is the workflow center of ProDock, designed for practical use rather than low-level API browsing.</p>
     </div>
     <div class="pd-mini-card">
       <h4>🧪 Workflow order</h4>
       <p><strong>Structure</strong> → <strong>Preprocess</strong> → <strong>Dock</strong> → <strong>Postprocess</strong> → <strong>Database</strong> → <strong>Automation</strong></p>
     </div>
     <div class="pd-mini-card">
       <h4>📦 Scope</h4>
       <p>Covers single runs, batch campaigns, multi-receptor and multi-ligand workflows, post-analysis, and persistent database-backed reuse.</p>
     </div>
   </div>

.. toctree::
   :maxdepth: 2
   :caption: Tutorial pages
   :hidden:

   tutorial/structure
   tutorial/preprocess
   tutorial/dock
   tutorial/postprocess
   tutorial/database
   tutorial/automation