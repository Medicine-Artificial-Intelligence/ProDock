"""
Custom exceptions for :mod:`prodock.postprocess.interaction`.

This module defines the exception hierarchy used by the interaction
postprocessing workflow. The custom exceptions make it easier to distinguish
between dependency issues, invalid inputs, visualization failures, and general
processing errors.

Example
-------
.. code-block:: python

    from prodock.postprocess.interaction.exceptions import (
        InteractionError,
        MissingDependencyError,
    )

    try:
        raise MissingDependencyError("prolif is required for interaction analysis")
    except InteractionError as exc:
        print(f"Interaction pipeline failed: {exc}")
"""


class InteractionError(RuntimeError):
    """
    Base exception for interaction postprocessing failures.

    All custom exceptions raised by
    :mod:`prodock.postprocess.interaction` should inherit from this class so
    callers can catch a single umbrella error type when appropriate.

    :param message:
        Human-readable error message describing the failure.
    :type message: str

    Example
    -------
    .. code-block:: python

        raise InteractionError("Unexpected interaction processing failure")
    """


class MissingDependencyError(InteractionError):
    """
    Raised when an optional runtime dependency is not available.

    This exception is intended for cases where a feature depends on an external
    package or executable that is not installed or cannot be imported at
    runtime.

    :param message:
        Human-readable error message describing the missing dependency.
    :type message: str

    :raises MissingDependencyError:
        Raised when a required optional dependency is unavailable.

    Example
    -------
    .. code-block:: python

        try:
            import prolif
        except ImportError as exc:
            raise MissingDependencyError(
                "prolif is required for interaction fingerprint generation"
            ) from exc
    """


class InvalidLigandInputError(InteractionError):
    """
    Raised when the ligand input cannot be interpreted.

    This exception should be used when a ligand file, ligand object, or ligand
    identifier is malformed, missing required content, or cannot be converted
    into the expected internal representation.

    :param message:
        Human-readable error message describing why the ligand input is invalid.
    :type message: str

    :raises InvalidLigandInputError:
        Raised when ligand input parsing or validation fails.

    Example
    -------
    .. code-block:: python

        if ligand_mol is None:
            raise InvalidLigandInputError("Ligand molecule could not be parsed")
    """


class VisualizationError(InteractionError):
    """
    Raised when a visualisation helper cannot complete.

    This exception is intended for failures during rendering, drawing,
    exporting, or formatting interaction visualizations.

    :param message:
        Human-readable error message describing the visualization failure.
    :type message: str

    :raises VisualizationError:
        Raised when an interaction visualization step fails.

    Example
    -------
    .. code-block:: python

        if output_path is None:
            raise VisualizationError("Output path is required for image export")
    """


class InteractionProcessingError(InteractionError):
    """
    Raised when a batch or pose-table interaction run fails.

    This exception is intended for high-level failures in the interaction
    processing pipeline, such as errors encountered while iterating over
    multiple poses, receptors, or ligand entries.

    :param message:
        Human-readable error message describing the processing failure.
    :type message: str

    :raises InteractionProcessingError:
        Raised when a batch-level or pose-table interaction workflow fails.

    Example
    -------
    .. code-block:: python

        try:
            results = extract_pose_table_interactions(df, receptor_map)
        except Exception as exc:
            raise InteractionProcessingError(
                "Failed to process interaction table"
            ) from exc
    """
