from .registry import available, factory, register, register_many
from .single import SingleDock, SingleResult
from .batch import BatchDock, DockResult, DockTask, MatrixDock
from .config import (
    BatchConfig,
    Box,
    CampaignConfig,
    DockRow,
    LigandSpec,
    LigandTask,
    ReceptorSpec,
    SingleConfig,
    SoftwareSpec,
)
from .vina import VinaEngine
from .smina import SminaEngine
from .gnina import GninaEngine
from .qvina import QVinaEngine
from .qvina_w import QVinaWEngine

register_many(
    [
        ("vina", lambda: VinaEngine()),
        ("smina", lambda: SminaEngine()),
        ("gnina", lambda: GninaEngine()),
        ("qvina", lambda: QVinaEngine()),
        ("qvina-w", lambda: QVinaWEngine()),
    ]
)

__all__ = [
    "available",
    "register",
    "register_many",
    "factory",
    "SingleDock",
    "SingleResult",
    "BatchDock",
    "MatrixDock",
    "DockTask",
    "DockResult",
    "VinaEngine",
    "SminaEngine",
    "GninaEngine",
    "QVinaEngine",
    "QVinaWEngine",
    "Box",
    "SingleConfig",
    "BatchConfig",
    "CampaignConfig",
    "DockRow",
    "LigandSpec",
    "LigandTask",
    "SoftwareSpec",
    "ReceptorSpec",
]
