from importlib.metadata import PackageNotFoundError, version

from .core import prodock
from .cli import main

try:
    __version__ = version("prodock")
except PackageNotFoundError:
    __version__ = "0.5.0"

__all__ = [
    "__version__",
    "prodock",
    "main",
]
