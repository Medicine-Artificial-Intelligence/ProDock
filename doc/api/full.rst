Full API
========

.. raw:: html

   <div class="pd-api-intro">
     <p class="pd-api-lead">
       ProDock exposes a workflow-oriented API for structure handling,
       preprocessing, docking, postprocessing, and database-backed analysis.
     </p>
   </div>

   <div class="pd-api-grid-2x3">

     <a class="pd-api-card pd-core" href="core.html">
       <div class="pd-api-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="12" cy="12" r="3.2"/>
           <path d="M12 2.8v2.2M12 19v2.2M21.2 12H19M5 12H2.8M18.5 5.5l-1.6 1.6M7.1 16.9l-1.6 1.6M18.5 18.5l-1.6-1.6M7.1 7.1 5.5 5.5"/>
         </svg>
       </div>
       <div class="pd-api-body">
         <div class="pd-api-title">Core</div>
         <div class="pd-api-text">
           High-level workflow.
         </div>
       </div>
     </a>

     <a class="pd-api-card pd-structure" href="structure.html">
       <div class="pd-api-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
           <path d="M7 6.5 12 3l5 3.5v6L12 16l-5-3.5z"/>
           <path d="M7 17.5 12 21l5-3.5"/>
           <path d="M12 3v6.5M7 6.5l5 3 5-3"/>
         </svg>
       </div>
       <div class="pd-api-body">
         <div class="pd-api-title">Structure</div>
         <div class="pd-api-text">
           Protein and ligand structure parsing, preparation, and utilities.
         </div>
       </div>
     </a>

     <a class="pd-api-card pd-preprocess" href="preprocess.html">
       <div class="pd-api-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
           <path d="M4 5h16"/>
           <path d="M6.5 5 10 12v5.5l4 2V12L17.5 5"/>
         </svg>
       </div>
       <div class="pd-api-body">
         <div class="pd-api-title">Preprocess</div>
         <div class="pd-api-text">
           Input cleaning, normalization, and preparation before docking.
         </div>
       </div>
     </a>

     <a class="pd-api-card pd-dock" href="dock.html">
       <div class="pd-api-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="12" cy="12" r="7.5"/>
           <circle cx="12" cy="12" r="3.2"/>
           <path d="M12 4.5v2.2M12 17.3v2.2M4.5 12h2.2M17.3 12h2.2"/>
         </svg>
       </div>
       <div class="pd-api-body">
         <div class="pd-api-title">Dock</div>
         <div class="pd-api-text">
           Docking engines, campaigns, and batch execution workflows.
         </div>
       </div>
     </a>

     <a class="pd-api-card pd-postprocess" href="postprocess.html">
       <div class="pd-api-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
           <path d="M5 18V9"/>
           <path d="M10 18V5"/>
           <path d="M15 18v-7"/>
           <path d="M20 18V11"/>
           <path d="M3.5 18.5h17"/>
         </svg>
       </div>
       <div class="pd-api-body">
         <div class="pd-api-title">Postprocess</div>
         <div class="pd-api-text">
           Pose extraction, interaction analysis, similarity, and visualization.
         </div>
       </div>
     </a>

     <a class="pd-api-card pd-database" href="database.html">
       <div class="pd-api-icon" aria-hidden="true">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
           <ellipse cx="12" cy="5.5" rx="6.5" ry="2.5"/>
           <path d="M5.5 5.5v5c0 1.4 2.9 2.5 6.5 2.5s6.5-1.1 6.5-2.5v-5"/>
           <path d="M5.5 10.5v5c0 1.4 2.9 2.5 6.5 2.5s6.5-1.1 6.5-2.5v-5"/>
         </svg>
       </div>
       <div class="pd-api-body">
         <div class="pd-api-title">Database</div>
         <div class="pd-api-text">
           Structured storage and many-to-many querying across docking results.
         </div>
       </div>
     </a>

   </div>

.. toctree::
   :maxdepth: 2
   :hidden:

   core
   structure
   preprocess
   dock
   postprocess
   database