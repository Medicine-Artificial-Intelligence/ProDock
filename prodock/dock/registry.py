from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

_REGISTRY: Dict[str, Callable[[], Any]] = {}


def register(name: str, factory: Callable[[], Any]) -> None:
    """
    Register a docking-backend factory under a case-insensitive key.

    The registry stores backend factories using the lowercase form of
    ``name``. Re-registering the same name replaces the previous factory.

    :param name:
        Human-readable backend name, such as ``"vina"`` or ``"smina"``.
        Registration is case-insensitive.
    :type name: str

    :param factory:
        Zero-argument callable returning a backend instance or other
        engine-specific object.
    :type factory: Callable[[], Any]

    :returns:
        ``None``.
    :rtype: None

    Example
    -------
    ::

        class VinaBackend:
            pass

        register("vina", lambda: VinaBackend())
    """
    _REGISTRY[name.lower()] = factory


def factory(name: str) -> Callable[[], Any]:
    """
    Return the registered factory for a backend name.

    Lookup is case-insensitive. If the backend name is not present in the
    registry, a :class:`KeyError` is raised with a message listing the
    currently available engine keys.

    :param name:
        Backend name to resolve.
    :type name: str

    :raises KeyError:
        If no backend factory is registered under ``name``.

    :returns:
        The registered zero-argument backend factory.
    :rtype: Callable[[], Any]

    Example
    -------
    ::

        fac = factory("vina")
        backend = fac()
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<empty>"
        raise KeyError(
            f"Unknown docking engine: {name!r}. Available engines: {available}"
        )
    return _REGISTRY[key]


def available() -> List[str]:
    """
    Return the sorted list of registered backend keys.

    The returned keys are the canonical lowercase names stored in the
    internal registry.

    :returns:
        Sorted list of registered engine names.
    :rtype: List[str]

    Example
    -------
    ::

        names = available()
        print(names)
    """
    return sorted(_REGISTRY)


def register_many(items: Iterable[tuple[str, Callable[[], Any]]]) -> None:
    """
    Register multiple backend factories in one call.

    Each item in ``items`` must be a ``(name, factory)`` pair. Registration is
    delegated to :func:`register`, so names remain case-insensitive and later
    entries overwrite earlier ones with the same normalized key.

    :param items:
        Iterable of ``(name, factory)`` pairs to register.
    :type items: Iterable[tuple[str, Callable[[], Any]]]

    :returns:
        ``None``.
    :rtype: None

    Example
    -------
    ::

        register_many([
            ("vina", lambda: VinaBackend()),
            ("smina", lambda: SminaBackend()),
        ])
    """
    for name, fac in items:
        register(name, fac)
