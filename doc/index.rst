
Overview
========

.. raw:: html

   <div class="pd-hero">
     <div>
       <div class="pd-eyebrow">workflow-first molecular docking</div>
       <p class="pd-hero-title">Automate docking, then reuse the same results for analysis</p>
       <p>Prepare structures, build docking inputs, run campaigns across engines, and keep poses, scores, and interactions queryable in one SQLite database.</p>
       <div class="pd-actions">
         <a class="pd-btn pd-btn--primary" href="getting_started.html">Get started</a>
         <a class="pd-btn" href="tutorial.html">Tutorial</a>
         <a class="pd-btn" href="architecture.html">Architecture</a>
         <a class="pd-btn" href="api.html">API</a>
       </div>
       <div class="pd-badges">
         <a href="https://pypi.org/project/prodock/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/prodock.svg"></a>
         <a href="https://anaconda.org/tieulongphan/prodock"><img alt="Conda" src="https://img.shields.io/conda/vn/tieulongphan/prodock.svg?label=conda"></a>
         <a href="https://hub.docker.com/r/tieulongphan/prodock"><img alt="Docker pulls" src="https://img.shields.io/docker/pulls/tieulongphan/prodock.svg"></a>
         <a href="https://hub.docker.com/r/tieulongphan/prodock"><img alt="Docker image version" src="https://img.shields.io/docker/v/tieulongphan/prodock/latest?label=container"></a>
         <a href="https://github.com/Medicine-Artificial-Intelligence/prodock/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Medicine-Artificial-Intelligence/prodock.svg"></a>
         <a href="https://github.com/Medicine-Artificial-Intelligence/prodock/releases"><img alt="Release" src="https://img.shields.io/github/v/release/Medicine-Artificial-Intelligence/prodock.svg"></a>
         <a href="https://github.com/Medicine-Artificial-Intelligence/prodock/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/Medicine-Artificial-Intelligence/prodock.svg"></a>
         <a href="https://github.com/Medicine-Artificial-Intelligence/prodock/actions/workflows/test-and-lint.yml"><img alt="CI" src="https://github.com/Medicine-Artificial-Intelligence/prodock/actions/workflows/test-and-lint.yml/badge.svg?branch=main"></a>
         <a href="https://github.com/Medicine-Artificial-Intelligence/prodock/pulls?q=is%3Apr+label%3Adependencies"><img alt="Dependency PRs" src="https://img.shields.io/github/issues-pr-raw/Medicine-Artificial-Intelligence/prodock?label=dependency%20PRs"></a>
         <a href="https://github.com/Medicine-Artificial-Intelligence/prodock/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/Medicine-Artificial-Intelligence/prodock.svg?style=social&label=Star"></a>
       </div>
     </div>
     <div class="pd-hero-media">
       <img src="_static/hero-workflow.svg" alt="workflow overview" />
     </div>
   </div>

.. raw:: html

   <div class="pd-grid-3">
     <a class="pd-card" href="tutorial.html"><div class="pd-kicker">Tutorial</div><h3>Follow the workflow</h3><p>Structure, preprocess, dock, postprocess, database, and automation pages stay grouped under one tutorial hub.</p></a>
     <a class="pd-card" href="architecture.html"><div class="pd-kicker">Architecture</div><h3>Understand the design</h3><p>See why ProDock uses a pose-centric database and how the package is split into reusable layers.</p></a>
     <a class="pd-card" href="api.html"><div class="pd-kicker">API</div><h3>Scan the package</h3><p>Start with a short API page, then move to a larger automodule-based reference under the same API section.</p></a>
   </div>

Store once, analyze many times
------------------------------

.. image:: _static/analysis-flow.svg
   :alt: database to analysis flow
   :class: pd-visual

.. raw:: html

   <div class="pd-grid-2">
     <div class="pd-panel pd-panel--blue">
       <h3>Package map</h3>
       <p>Core orchestration stays compact while structure, preprocess, docking, postprocess, and database modules remain reusable.</p>
       <div class="pd-visual"><img src="_static/package-map.svg" alt="package map"></div>
     </div>
     <div class="pd-panel pd-panel--stone">
       <h3>Database overview</h3>
       <p>Catalog tables feed a central poses table, which branches into score and interaction storage.</p>
       <div class="pd-visual"><img src="_static/db-architecture.svg" alt="database architecture"></div>
     </div>
   </div>

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   tutorial
   architecture
   api
   reference
