# from __future__ import annotations

# """Interaction analysis utilities for protein-ligand docking poses.

# This subpackage contains the high-level interaction extraction workflow used in
# ProDock. It wraps receptor loading, ligand preparation, ProLIF fingerprint
# execution, flattened interaction-event generation, and pose-level summary
# construction.

# Typical usage
# -------------
# .. code-block:: python

#     from prodock.postprocess.interaction import (
#         InteractionProfiler,
#         extract_pose_table_interactions,
#     )
# """

# from .core import InteractionProfiler, extract_pose_table_interactions
# from .exceptions import InteractionError, MissingDependencyError, VisualizationError
# from .models import InteractionRunResult
# from .visualize import make_barcode, save_barcode

# __all__ = [
#     "InteractionProfiler",
#     "extract_pose_table_interactions",
#     "InteractionRunResult",
#     "InteractionError",
#     "MissingDependencyError",
#     "VisualizationError",
#     "make_barcode",
#     "make_heatmap",
#     "save_barcode",
#     "save_heatmap",
# ]
