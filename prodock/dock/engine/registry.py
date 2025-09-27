from __future__ import annotations
from typing import Callable, Dict, Any

_REGISTRY: Dict[str, Callable[[], Any]] = {}


def register(name: str, factory: Callable[[], Any]) -> None:
    """
    Register a docking backend factory.

    :param name: Registry key (case-insensitive), e.g. ``"smina"``.
    :param factory: Zero-arg callable returning a backend instance.
    """
    _REGISTRY[name.lower()] = factory


def factory(name: str) -> Callable[[], Any]:
    """
    Retrieve a previously registered factory by name.

    :param name: Registry key.
    :returns: Factory callable.
    :raises KeyError: If the engine is unknown.

    Examples
    --------
    .. code-block:: python

        from prodock.dock.engine import registry

        f = registry.factory("smina")
        backend = f()
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown docking engine: {name}")
    return _REGISTRY[key]
