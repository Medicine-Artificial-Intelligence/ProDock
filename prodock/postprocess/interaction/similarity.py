from __future__ import annotations

"""
Similarity utilities for interaction fingerprints.

This module provides helper functions for computing pairwise similarity
matrices from interaction fingerprint vectors. The main use case is comparing
pose-level or ligand-level fingerprints generated during interaction
postprocessing.

The implementation is designed to work with RDKit fingerprint objects,
including explicit bit vectors and sparse count vectors, so it can support both
binary and count-based interaction representations.

Example
-------
.. code-block:: python

    from prodock.postprocess.interaction.similarity import (
        tanimoto_similarity_matrix,
    )

    sim = tanimoto_similarity_matrix(result.bitvectors, result.molecule_names)
    print(sim)
"""

from typing import Any, Sequence

import pandas as pd

from .exceptions import MissingDependencyError


def tanimoto_similarity_matrix(
    vectors: Sequence[Any],
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Compute a pairwise Tanimoto similarity matrix.

    This function computes the all-against-all Tanimoto similarity between
    fingerprint vectors and returns the result as a square
    :class:`pandas.DataFrame`. It is suitable for RDKit explicit bit vectors as
    well as sparse count vectors, which makes it useful for both boolean and
    count-based ProLIF fingerprints.

    When ``names`` is not provided, default labels of the form
    ``mol_0000``, ``mol_0001``, and so on are generated automatically.

    :param vectors:
        Sequence of RDKit-compatible fingerprint vectors. Each element should be
        accepted by :func:`rdkit.DataStructs.TanimotoSimilarity`.
    :type vectors: Sequence[Any]
    :param names:
        Optional labels to use for both the row index and column names of the
        returned matrix. When omitted, default molecule labels are generated
        from the vector order.
    :type names: Sequence[str] | None

    :returns:
        Square dataframe whose ``(i, j)`` entry contains the Tanimoto
        similarity between ``vectors[i]`` and ``vectors[j]``.
    :rtype: pandas.DataFrame

    :raises MissingDependencyError:
        Raised when RDKit is not installed or cannot be imported.

    Example
    -------
    .. code-block:: python

        sim = tanimoto_similarity_matrix(
            vectors=result.bitvectors,
            names=result.molecule_names,
        )
        print(sim.iloc[:5, :5])

    Another Example
    ---------------
    .. code-block:: python

        sim = tanimoto_similarity_matrix(vectors)
        assert sim.shape == (len(vectors), len(vectors))
    """
    try:
        from rdkit.DataStructs import TanimotoSimilarity
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "RDKit is required to compute Tanimoto similarity matrices."
        ) from exc

    size = len(vectors)
    labels = list(names) if names is not None else [f"mol_{i:04d}" for i in range(size)]

    matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(float(TanimotoSimilarity(vectors[i], vectors[j])))
        matrix.append(row)

    return pd.DataFrame(matrix, index=labels, columns=labels)
