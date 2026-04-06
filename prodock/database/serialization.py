from __future__ import annotations

"""Serialization helpers for SQLite payloads and RDKit molecules."""

import json
import re
import zlib
import pandas as pd
from typing import Any, Mapping, Optional, Sequence, Union

from rdkit.Chem import rdchem

StringOrMany = Optional[Union[str, Sequence[str]]]

_RESIDUE_ID_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_]+?)(?P<number>-?\d+)?(?:\.(?P<chain>[A-Za-z0-9_]+))?$"
)


def json_dumps(value: Optional[Mapping[str, Any]]) -> str:
    """
    Serialize a mapping-like object into a JSON object string.

    This helper is intended for SQLite text fields that store structured
    metadata. Missing input values are normalized to an empty JSON object.

    :param value:
        Mapping payload to serialize. If ``None``, an empty mapping is used.
        The input is converted with :class:`dict` before serialization.
    :type value: Optional[Mapping[str, Any]]

    :returns:
        JSON object string. When ``value`` is ``None`` or empty, the function
        returns ``"{}"``.
    :rtype: str

    :raises TypeError:
        If ``value`` cannot be converted to a dictionary-compatible object.

    Example:
        >>> json_dumps({"engine": "vina", "rank": 1})
        '{"engine": "vina", "rank": 1}'
        >>> json_dumps(None)
        '{}'
    """
    if value is None:
        return "{}"
    try:
        if pd.isna(value):
            return "{}"
    except Exception:
        pass
    return json.dumps(dict(value), ensure_ascii=False, default=str)


def json_dumps_list(value: Optional[Sequence[Any]]) -> str:
    """
    Serialize a sequence-like object into a JSON array string.

    This helper is useful for storing lists of indices, labels, or other
    ordered payloads in SQLite text columns. Missing input values are
    normalized to an empty JSON array.

    :param value:
        Sequence payload to serialize. If ``None``, an empty list is used.
        The input is converted with :class:`list` before serialization.
    :type value: Optional[Sequence[Any]]

    :returns:
        JSON array string. When ``value`` is ``None`` or empty, the function
        returns ``"[]"``.
    :rtype: str

    :raises TypeError:
        If ``value`` is not sequence-like.

    Example:
        >>> json_dumps_list([1, 2, 3])
        '[1, 2, 3]'
        >>> json_dumps_list(None)
        '[]'
    """
    return json.dumps(list(value or []), ensure_ascii=False, default=str)


def json_loads_dict(value: Optional[str]) -> dict[str, Any]:
    """
    Deserialize JSON text into a Python dictionary.

    The function is tolerant of missing values. If the input is ``None``,
    empty, or decodes to a JSON value that is not an object, an empty
    dictionary is returned.

    :param value:
        JSON text expected to represent an object.
    :type value: Optional[str]

    :returns:
        Deserialized dictionary. Returns an empty dictionary when the input is
        missing or when the decoded JSON payload is not a dictionary.
    :rtype: dict[str, Any]

    :raises json.JSONDecodeError:
        If ``value`` is non-empty but is not valid JSON.

    Example:
        >>> json_loads_dict('{"a": 1}')
        {'a': 1}
        >>> json_loads_dict(None)
        {}
        >>> json_loads_dict('["not", "a", "dict"]')
        {}
    """
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def json_loads_list(value: Optional[str]) -> list[Any]:
    """
    Deserialize JSON text into a Python list.

    The function is tolerant of missing values. If the input is ``None``,
    empty, or decodes to a JSON value that is not an array, an empty list is
    returned.

    :param value:
        JSON text expected to represent an array.
    :type value: Optional[str]

    :returns:
        Deserialized list. Returns an empty list when the input is missing or
        when the decoded JSON payload is not a list.
    :rtype: list[Any]

    :raises json.JSONDecodeError:
        If ``value`` is non-empty but is not valid JSON.

    Example:
        >>> json_loads_list('[1, 2, 3]')
        [1, 2, 3]
        >>> json_loads_list(None)
        []
        >>> json_loads_list('{"not": "a list"}')
        []
    """
    if not value:
        return []
    loaded = json.loads(value)
    return loaded if isinstance(loaded, list) else []


def json_loads_int_list(value: Optional[str]) -> list[int]:
    """
    Deserialize JSON text into a list of integers.

    This helper first decodes the input with :func:`json_loads_list`, then
    attempts to coerce each element to :class:`int`. Elements that cannot be
    converted are silently skipped.

    :param value:
        JSON text expected to represent an array.
    :type value: Optional[str]

    :returns:
        List of successfully converted integer values.
    :rtype: list[int]

    :raises json.JSONDecodeError:
        If ``value`` is non-empty but is not valid JSON.

    Example:
        >>> json_loads_int_list('[1, "2", 3.0, "x"]')
        [1, 2, 3]
        >>> json_loads_int_list(None)
        []
    """
    out: list[int] = []
    for item in json_loads_list(value):
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def as_many(value: StringOrMany) -> Optional[list[str]]:
    """
    Normalize a scalar-or-sequence string input into a list of strings.

    This helper is commonly used for query filters that may accept either a
    single value or multiple values. ``None`` is preserved to distinguish
    between "no filter" and an explicitly provided empty list.

    :param value:
        Either a single string, a sequence of strings, or ``None``.
    :type value: Optional[Union[str, Sequence[str]]]

    :returns:
        A list of strings when input is provided, otherwise ``None``.
    :rtype: Optional[list[str]]

    Example:
        >>> as_many("vina")
        ['vina']
        >>> as_many(["vina", "smina"])
        ['vina', 'smina']
        >>> as_many(None) is None
        True
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def make_pose_key(
    receptor_id: str,
    ligand_id: str,
    engine: str,
    pose_rank: int,
) -> str:
    """
    Build a deterministic human-readable pose identifier.

    The generated key is intended to be stable across exports, imports, and
    database round-trips, making it suitable as an external pose reference.

    :param receptor_id:
        Receptor identifier, for example ``"1M17"``.
    :type receptor_id: str
    :param ligand_id:
        Ligand identifier, for example ``"erlotinib"``.
    :type ligand_id: str
    :param engine:
        Docking engine name, for example ``"vina"`` or ``"qvina"``.
    :type engine: str
    :param pose_rank:
        One-based pose rank within the docking result set.
    :type pose_rank: int

    :returns:
        Pose key in the canonical form
        ``receptor__ligand__engine__pose<rank>``.
    :rtype: str

    Example:
        >>> make_pose_key("1M17", "erlotinib", "qvina", 1)
        '1M17__erlotinib__qvina__pose1'
    """
    return f"{receptor_id}__{ligand_id}__{engine}__pose{int(pose_rank)}"


def parse_residue_id(
    residue_id: Optional[str],
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse a composite residue identifier into structured fields.

    The supported compact format is typically of the form
    ``<residue_name><residue_number>.<chain_id>``, for example ``LEU149.A``.
    The residue number and chain identifier are optional. If parsing fails,
    all fields are returned as ``None``.

    :param residue_id:
        Composite residue identifier to parse.
    :type residue_id: Optional[str]

    :returns:
        Tuple ``(residue_name, residue_number, chain_id)``. Missing or invalid
        components are returned as ``None``.
    :rtype: tuple[Optional[str], Optional[int], Optional[str]]

    Example:
        >>> parse_residue_id("LEU149.A")
        ('LEU', 149, 'A')
        >>> parse_residue_id("GLY24")
        ('GLY', 24, None)
        >>> parse_residue_id(None)
        (None, None, None)
    """
    if residue_id is None:
        return None, None, None
    text = str(residue_id).strip()
    if not text:
        return None, None, None
    match = _RESIDUE_ID_RE.match(text)
    if match is None:
        return None, None, None
    residue_name = match.group("name") or None
    residue_number_text = match.group("number")
    residue_number = int(residue_number_text) if residue_number_text else None
    chain_id = match.group("chain") or None
    return residue_name, residue_number, chain_id


def compose_residue_id(
    residue_name: Optional[str],
    residue_number: Optional[int],
    chain_id: Optional[str],
) -> Optional[str]:
    """
    Compose a compact residue identifier from structured residue fields.

    The resulting format is compatible with identifiers such as
    ``LEU149.A``. The residue name is required. The residue number and chain
    identifier are appended only when provided.

    :param residue_name:
        Residue name, such as ``"LEU"``.
    :type residue_name: Optional[str]
    :param residue_number:
        Residue number, such as ``149``.
    :type residue_number: Optional[int]
    :param chain_id:
        Chain identifier, such as ``"A"``.
    :type chain_id: Optional[str]

    :returns:
        Composite residue identifier, or ``None`` when ``residue_name`` is
        missing.
    :rtype: Optional[str]

    Example:
        >>> compose_residue_id("LEU", 149, "A")
        'LEU149.A'
        >>> compose_residue_id("GLY", 24, None)
        'GLY24'
        >>> compose_residue_id(None, 24, "A") is None
        True
    """
    if not residue_name:
        return None
    out = str(residue_name)
    if residue_number is not None:
        out += str(int(residue_number))
    if chain_id:
        out += f".{chain_id}"
    return out


def serialize_mol(
    mol: rdchem.Mol,
    *,
    compress: bool = True,
    include_props: bool = True,
) -> bytes:
    """
    Serialize an RDKit molecule into a binary payload suitable for SQLite
    storage.

    The molecule is converted to RDKit's binary representation via
    :meth:`rdkit.Chem.rdchem.Mol.ToBinary`. The payload can optionally be
    compressed with :mod:`zlib` to reduce storage size. RDKit properties and
    conformer-related data may be preserved depending on ``include_props``.

    :param mol:
        RDKit molecule to serialize.
    :type mol: rdchem.Mol
    :param compress:
        If ``True``, compress the binary payload with :func:`zlib.compress`
        before returning it.
    :type compress: bool
    :param include_props:
        If ``True``, preserve RDKit molecule properties in the serialized
        payload. If ``False``, properties are omitted.
    :type include_props: bool

    :returns:
        Serialized binary payload ready to be stored in a SQLite ``BLOB``
        column.
    :rtype: bytes

    :raises ValueError:
        If ``mol`` is ``None``.

    Example:
        >>> # blob = serialize_mol(mol)
        >>> # cursor.execute("INSERT INTO poses (mol_blob) VALUES (?)", (blob,))
    """
    if mol is None:
        raise ValueError("mol must not be None")

    flags = (
        rdchem.PropertyPickleOptions.AllProps
        if include_props
        else rdchem.PropertyPickleOptions.NoProps
    )
    payload = mol.ToBinary(flags)
    return zlib.compress(payload) if compress else payload


def deserialize_mol(blob: bytes, *, compressed: bool = True) -> rdchem.Mol:
    """
    Reconstruct an RDKit molecule from a stored binary payload.

    This function reverses :func:`serialize_mol`. The payload may optionally be
    decompressed with :mod:`zlib` before being passed to the RDKit molecule
    constructor.

    :param blob:
        Stored binary molecule payload, typically retrieved from a SQLite
        ``BLOB`` column.
    :type blob: bytes
    :param compressed:
        If ``True``, first decompress the payload with
        :func:`zlib.decompress`.
    :type compressed: bool

    :returns:
        Reconstructed RDKit molecule object.
    :rtype: rdchem.Mol

    :raises ValueError:
        If ``blob`` is empty.
    :raises ValueError:
        If the molecule cannot be reconstructed from the payload.
    :raises zlib.error:
        If ``compressed`` is ``True`` and the payload is not valid compressed
        data.

    Example:
        >>> # mol = deserialize_mol(row["mol_blob"])
        >>> # smiles = Chem.MolToSmiles(mol)
    """
    if not blob:
        raise ValueError("blob must not be empty")

    payload = zlib.decompress(blob) if compressed else blob
    mol = rdchem.Mol(payload)
    if mol is None:
        raise ValueError("Could not deserialize RDKit molecule")
    return mol
