Architecture
============

.. raw:: html

   <div class="pd-arch-hero">
     <div class="pd-arch-hero-kicker">System design</div>
     <h2>Structured architecture for campaign-scale docking</h2>
     <p>
       ProDock is designed around one central idea: docking is not just a single
       run, but a reproducible campaign spanning many receptors, many ligands,
       and one or more docking engines. The architecture separates execution,
       analysis, and persistence so workflows stay scalable, queryable, and
       reproducible.
     </p>
   </div>

.. raw:: html

   <div class="pd-card-grid pd-card-grid-3">
     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <rect x="4" y="5" width="16" height="14" rx="2"/>
           <path d="M8 9h8M8 13h5"/>
           <circle cx="16.5" cy="13.5" r="1.5"/>
         </svg>
       </div>
       <h3>Deterministic workflow</h3>
       <p>
         Structure intake, preprocessing, docking, postprocessing, and database
         export follow one stable campaign-oriented sequence.
       </p>
     </div>

     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <path d="M6 6h12v12H6z"/>
           <path d="M9 9h6M9 12h6M9 15h4"/>
         </svg>
       </div>
       <h3>Engine-agnostic execution</h3>
       <p>
         The docking layer abstracts backend-specific details so campaigns can
         run across multiple engines with one consistent interface.
       </p>
     </div>

     <div class="pd-icon-card">
       <div class="pd-icon-wrap" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
           <ellipse cx="12" cy="5" rx="6.5" ry="2.8"/>
           <path d="M5.5 5v6c0 1.5 2.9 2.8 6.5 2.8s6.5-1.3 6.5-2.8V5"/>
           <path d="M5.5 11v6c0 1.5 2.9 2.8 6.5 2.8s6.5-1.3 6.5-2.8v-6"/>
         </svg>
       </div>
       <h3>Pose-centric persistence</h3>
       <p>
         A normalized SQLite database stores poses, scores, and interactions so
         downstream analysis does not depend on fragile folder parsing.
       </p>
     </div>
   </div>

Why the architecture looks like this
------------------------------------

ProDock is built for workflows that grow beyond a single receptor–ligand test
case. At that scale, a folder-only approach becomes difficult to maintain.

The architecture is meant to solve three recurring problems:

.. raw:: html

   <div class="pd-arch-triptych">
     <div class="pd-arch-triptych-card">
       <h3>File-system fragility</h3>
       <p>
         Logs, poses, converted structures, and interaction outputs become
         scattered across engines and folders, making reuse difficult.
       </p>
     </div>
     <div class="pd-arch-triptych-card">
       <h3>Relational complexity</h3>
       <p>
         Real campaigns create many-to-many relationships across receptors,
         ligands, engines, and pose ranks that flat files do not model well.
       </p>
     </div>
     <div class="pd-arch-triptych-card">
       <h3>Retrospective analysis</h3>
       <p>
         Consensus scoring, residue filtering, interaction queries, and campaign
         reporting should be possible later without rerunning the heavy workflow.
       </p>
     </div>
   </div>

Workflow architecture
---------------------

.. raw:: html

   <div class="pd-arch-flow">
     <div class="pd-arch-flow-step">
       <div class="pd-arch-flow-num">1</div>
       <div class="pd-arch-flow-body">
         <h3>Structure</h3>
         <p>Obtain and normalize structural inputs and conversions.</p>
       </div>
     </div>
     <div class="pd-arch-flow-arrow">→</div>
     <div class="pd-arch-flow-step">
       <div class="pd-arch-flow-num">2</div>
       <div class="pd-arch-flow-body">
         <h3>Preprocess</h3>
         <p>Prepare receptors, ligands, and docking boxes.</p>
       </div>
     </div>
     <div class="pd-arch-flow-arrow">→</div>
     <div class="pd-arch-flow-step">
       <div class="pd-arch-flow-num">3</div>
       <div class="pd-arch-flow-body">
         <h3>Dock</h3>
         <p>Run one or more engines over many receptor–ligand pairs.</p>
       </div>
     </div>
     <div class="pd-arch-flow-arrow">→</div>
     <div class="pd-arch-flow-step">
       <div class="pd-arch-flow-num">4</div>
       <div class="pd-arch-flow-body">
         <h3>Postprocess</h3>
         <p>Extract scores, crawl poses, and compute interactions.</p>
       </div>
     </div>
     <div class="pd-arch-flow-arrow">→</div>
     <div class="pd-arch-flow-step">
       <div class="pd-arch-flow-num">5</div>
       <div class="pd-arch-flow-body">
         <h3>Database</h3>
         <p>Persist campaign outputs for later querying and reuse.</p>
       </div>
     </div>
   </div>

.. image:: _static/tutorial-overview.svg
   :alt: workflow overview from structure to database
   :class: pd-visual

This stage order matters because it separates heavy generation work from later
analysis. Once a campaign has finished, most downstream questions should become
query problems rather than rerun problems.

Package dependency map
----------------------

.. image:: _static/package-map.svg
   :alt: package map
   :class: pd-visual

The package layout mirrors the workflow:

- ``structure`` handles intake and low-level conversion
- ``preprocess`` prepares receptors, ligands, and box definitions
- ``dock`` runs single or batch docking through registered engines
- ``postprocess`` parses logs, crawls poses, and computes interactions
- ``database`` stores and queries campaign outputs
- ``core`` and automation entry points tie the layers together

This modular organization allows two usage styles:

- use the entire stack end-to-end,
- or reuse one stage independently inside a notebook or script.

Many-to-many campaign model
---------------------------

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
         <h2>Core architectural paradigm</h2>
         <p>
           ProDock models docking as a many-to-many campaign across receptors,
           ligands, engines, and pose ranks rather than as isolated output files.
         </p>
       </div>
     </div>
   </div>

.. raw:: html

   <div class="pd-arch-entity-band">
     <div class="pd-arch-entity">Receptors</div>
     <div class="pd-arch-entity-join">×</div>
     <div class="pd-arch-entity">Ligands</div>
     <div class="pd-arch-entity-join">×</div>
     <div class="pd-arch-entity">Engines</div>
     <div class="pd-arch-entity-join">→</div>
     <div class="pd-arch-entity pd-arch-entity-strong">Poses</div>
   </div>

In this model:

- one receptor can be docked against many ligands,
- one ligand can be tested across many receptors,
- one receptor–ligand pair can be evaluated by many engines,
- one receptor–ligand–engine combination can produce many ranked poses.

That is why ProDock treats the **pose** as the central stored result. Scores,
interaction rows, and later analyses all attach naturally to that level.

Relational database architecture
--------------------------------

.. image:: _static/db-architecture.svg
   :alt: database architecture
   :class: pd-visual

The database is normalized so that receptor, ligand, and engine identifiers are
stored once, while pose-specific and interaction-specific records remain linked
through stable keys.

This gives three practical benefits:

.. raw:: html

   <div class="pd-card-grid pd-card-grid-3">
     <div class="pd-icon-card">
       <h3>Compact storage</h3>
       <p>
         Shared identifiers are not duplicated across every downstream analysis row.
       </p>
     </div>
     <div class="pd-icon-card">
       <h3>Relational integrity</h3>
       <p>
         Scores, molecules, and interactions remain linked to the same pose identity.
       </p>
     </div>
     <div class="pd-icon-card">
       <h3>Queryable analysis</h3>
       <p>
         Identity, score, and interaction filters can be combined inside one query layer.
       </p>
     </div>
   </div>

The practical result is that ProDock can answer questions such as:

- which ligands produce the best-ranked poses for one receptor,
- which poses satisfy both affinity and residue-contact constraints,
- how interaction fingerprints vary across engines,
- how one campaign compares across many receptor–ligand pairs.

Execution and analysis are separated
------------------------------------

.. raw:: html

   <div class="pd-arch-split">
     <div class="pd-arch-split-card">
       <h3>Execution layer</h3>
       <ul>
         <li>prepare receptors</li>
         <li>prepare ligands</li>
         <li>run docking engines</li>
         <li>generate logs and poses</li>
       </ul>
     </div>
     <div class="pd-arch-split-card">
       <h3>Analysis layer</h3>
       <ul>
         <li>extract score tables</li>
         <li>crawl pose trees</li>
         <li>compute interactions</li>
         <li>query stored SQLite records</li>
       </ul>
     </div>
   </div>

A central architectural rule in ProDock is that **execution is decoupled from
analysis**.

Heavy stages generate artifacts. Later stages transform those artifacts into
tables, summaries, and persistent records. Once stored, downstream work becomes
lighter, more reproducible, and easier to query.


System summary
--------------

.. raw:: html

   <div class="pd-mini-callout">
     <strong>Architectural summary:</strong>
     ProDock is a campaign-oriented docking system with modular execution stages,
     engine-agnostic docking, and a pose-centric relational database. Its main
     goal is to make large docking studies reproducible during execution and
     queryable after execution.
   </div>

See also
--------

- :doc:`tutorial/preprocess` — prepare receptors, ligands, and docking boxes
- :doc:`tutorial/dock` — run single and batch docking workflows
- :doc:`tutorial/postprocess` — parse scores, crawl poses, and compute interactions
- :doc:`tutorial/database` — store and query campaign outputs in SQLite