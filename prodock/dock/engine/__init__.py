from .registry import register, factory
from .single import SingleDock
from .batch import BatchDock
from .config import Box, SingleConfig, BatchConfig, LigandTask

# Engines
from .vina_binding import VinaBindingEngine
from .vina import VinaEngine
from .smina import SminaEngine
from .gnina import GninaEngine
from .qvina import QVinaEngine
from .qvina_w import QVinaWEngine

# Register built-ins
register("vina_binding", lambda: VinaBindingEngine())
register("vina", lambda: VinaEngine())
register("smina", lambda: SminaEngine())
register("gnina", lambda: GninaEngine())
register("qvina", lambda: QVinaEngine())
register("qvina-w", lambda: QVinaWEngine())

__all__ = [
    "register",
    "factory",
    "SingleDock",
    "BatchDock",
    "VinaBindingEngine",
    "VinaCLIEngine",
    "SminaEngine",
    "GninaEngine",
    "QVinaEngine",
    "QVinaWEngine",
    "Box",
    "SingleConfig",
    "BatchConfig",
    "LigandTask",
]
