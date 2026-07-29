IO
==

The :mod:`prodock.io` namespace collects shared utilities for molecular text
parsing, RDKit-based format conversion, defensive environment cleanup, and
structured logging. These helpers are intended to support the rest of the
ProDock workflow stack while remaining reusable in standalone scripts and
notebooks.

Overview
--------

.. raw:: html

   <div class="pd-grid-2">
     <div class="pd-mini">
       <h3>Parsing</h3>
       <p>Read molecular text blocks such as SDF, PDB, MOL2, and XYZ into RDKit molecules with defensive fallbacks.</p>
     </div>
     <div class="pd-mini">
       <h3>Conversion</h3>
       <p>Convert between SMILES, SDF, and PDB while optionally generating and optimizing 3D coordinates.</p>
     </div>
     <div class="pd-mini">
       <h3>Utilities</h3>
       <p>Provide best-effort runtime helpers such as graceful PyMOL shutdown and module cleanup.</p>
     </div>
     <div class="pd-mini">
       <h3>Logging</h3>
       <p>Expose structured logging, colored console formatters, JSON logs, step decorators, and timing helpers.</p>
     </div>
   </div>

Submodules
----------

``prodock.io.parser``
~~~~~~~~~~~~~~~~~~~~~

Small defensive parsers for text blocks used by grid-box and related modules.
These functions attempt to recover an RDKit molecule from common chemical file
formats and return ``None`` instead of raising when parsing fails, making them
well suited for fallback-heavy preprocessing code.

.. automodule:: prodock.io.parser
   :members:
   :undoc-members:
   :private-members:
   :show-inheritance:

``prodock.io.rdkit``
~~~~~~~~~~~~~~~~~~~~

RDKit-focused molecular I/O functions for common format conversions. This
module covers SMILES-to-molecule conversion, SDF and PDB writing, file-to-RDKit
loading, and optional 3D embedding and optimization. When available, it can use
an internal ProDock conformer engine before falling back to plain RDKit methods.

.. automodule:: prodock.io.rdkit
   :members:
   :undoc-members:
   :show-inheritance:

``prodock.io.utils``
~~~~~~~~~~~~~~~~~~~~

General utility helpers that do not belong to a chemistry-specific conversion
module. The current utility surface focuses on runtime environment cleanup for
PyMOL-based workflows.

.. automodule:: prodock.io.utils
   :members:
   :undoc-members:
   :show-inheritance:

``prodock.io.logging``
~~~~~~~~~~~~~~~~~~~~~~

The logging package provides a compact but flexible structured logging layer for
scripts, notebooks, and automation runs. It includes a logger manager,
structured adapters, console and JSON formatters, step decorators, and elapsed
time measurement helpers.

.. automodule:: prodock.io.logging
   :members:
   :undoc-members:
   :show-inheritance:

``prodock.io.logging.manager``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: prodock.io.logging.manager
   :members:
   :undoc-members:
   :show-inheritance:

``prodock.io.logging.decorators``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: prodock.io.logging.decorators
   :members:
   :undoc-members:
   :show-inheritance:

``prodock.io.logging.formatters``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: prodock.io.logging.formatters
   :members:
   :undoc-members:
   :show-inheritance:

``prodock.io.logging.timing``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: prodock.io.logging.timing
   :members:
   :undoc-members:
   :show-inheritance:

``prodock.io.logging.compat_logging``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: prodock.io.logging.compat_logging
   :members:
   :exclude-members: StructuredAdapter
   :undoc-members:
   :show-inheritance:
