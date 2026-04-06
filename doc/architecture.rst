Architecture
============

.. raw:: html

   <div class="pd-hero-compact">
     <div class="pd-kicker">System design</div>
     <h2>Structured Architecture for High-Throughput Docking</h2>
     <p class="pd-lead">
       ProDock structures molecular docking as a scalable campaign, systematically managing <strong>many-to-many relationships across receptors, ligands, and engines</strong>. A pose-centric database ensures rigorous downstream analysis and computational reproducibility.
     </p>
   </div>

System Motivation
-----------------

.. raw:: html

   <div class="pd-grid-3 pd-grid-tight">
     <div class="pd-mini-card pd-mini-card-accent">
       <h4>📂 File-System Robustness</h4>
       <p>Standard directory-based storage fails at scale; ProDock centralizes data to prevent fragmentation.</p>
     </div>
     <div class="pd-mini-card pd-mini-card-accent">
       <h4>🔗 Relational Integrity</h4>
       <p>Explicit linkages between receptors, ligands, search algorithms, and poses are strictly maintained.</p>
     </div>
     <div class="pd-mini-card pd-mini-card-accent">
       <h4>♻️ Retrospective Analysis</h4>
       <p>Persistent pose records enable efficient consensus scoring, interaction profiling, and statistical reporting.</p>
     </div>
   </div>

Architectural Objectives
------------------------

.. raw:: html

   <div class="pd-grid-2">
     <div class="pd-mini-card">
       <h4>🧭 Deterministic Workflow</h4>
       <p>Structure → Preprocess → Dock → Postprocess → Database.</p>
     </div>
     <div class="pd-mini-card">
       <h4>⚙️ Automated Execution</h4>
       <p>Structured entity definitions guarantee exact reproducibility of computational campaigns.</p>
     </div>
     <div class="pd-mini-card">
       <h4>🔌 Engine Agnosticism</h4>
       <p>A unified framework abstracts underlying algorithms, permitting seamless backend integration.</p>
     </div>
     <div class="pd-mini-card">
       <h4>🗄️ Pose-Centric Storage</h4>
       <p>SQLite integration retains discrete outputs as highly queryable, structured records.</p>
     </div>
   </div>

Workflow Architecture
---------------------

.. image:: _static/tutorial-overview.svg
   :alt: workflow overview from structure to database
   :class: pd-visual

Package Dependency Map
----------------------

.. image:: _static/package-map.svg
   :alt: package map
   :class: pd-visual

Relational Database Architecture
--------------------------------

.. raw:: html

   <div class="pd-callout pd-callout-strong">
     <div class="pd-callout-title">Core Architectural Paradigm</div>
     <p>
       The database maps the comprehensive <strong>receptor–ligand–engine–pose</strong> phase space, elevating ProDock into a robust <strong>many-to-many querying framework</strong>.
     </p>
   </div>

.. raw:: html

   <div class="pd-grid-4 pd-grid-tight">
     <div class="pd-mini-card pd-mini-card-accent">
       <h4>🧬 Receptors</h4>
       <p>Prepared target libraries.</p>
     </div>
     <div class="pd-mini-card pd-mini-card-accent">
       <h4>🧪 Ligands</h4>
       <p>High-throughput screening sets.</p>
     </div>
     <div class="pd-mini-card pd-mini-card-accent">
       <h4>⚙️ Engines</h4>
       <p>Diverse docking algorithms.</p>
     </div>
     <div class="pd-mini-card pd-mini-card-accent">
       <h4>📍 Poses</h4>
       <p>Ranked structural predictions.</p>
     </div>
   </div>

.. image:: _static/db-architecture.svg
   :alt: database architecture
   :class: pd-visual

Database Infrastructure
-----------------------

.. raw:: html

   <div class="pd-note-inline">
     <strong>Architectural Imperative:</strong> ProDock strictly decouples execution from analysis. The persistent database supports rigorous comparative queries without iterative re-computation.
   </div>