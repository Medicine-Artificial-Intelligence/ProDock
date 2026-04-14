from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Literal

from prodock.io.logging import get_logger

logger = get_logger(__name__)

SanitizeBackend = Literal["meeko", "obabel"]


class PDBQTSanitizer:
    """
    Backend-aware PDBQT sanitizer and validator.

    This sanitizer is designed for ligand PDBQT compatibility with older
    Vina/QuickVina-family parsers that are sensitive to fixed-column formatting.

    Main behavior
    -------------
    - rebuilds ATOM/HETATM lines into one consistent fixed-width format
    - preserves legacy AutoDock/Vina atom types such as ``A``, ``OA``, ``NA``,
      ``SA``, and ``HD``
    - selectively downgrades unsupported pseudo-types such as ``CG0`` and ``G0``
    - keeps torsion-tree records unchanged

    Recommended usage
    -----------------
    .. code-block:: python

        sanitizer = PDBQTSanitizer("ligand.pdbqt", backend="meeko")
        sanitizer.validate(strict=False)
        sanitizer.sanitize(rebuild=True, aggressive=False)
        sanitizer.write("ligand.sanitized.pdbqt")

    :param path:
        Optional path to a PDBQT file to load immediately.
    :type path: Optional[str | pathlib.Path]

    :param backend:
        Sanitizer behavior profile.
    :type backend: Literal["meeko", "obabel"]
    """

    _ATOM_RE = re.compile(r"^(ATOM|HETATM)\b")
    _FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")
    _TAG_WHITELIST = {
        "REMARK",
        "ROOT",
        "ENDROOT",
        "BRANCH",
        "ENDBRANCH",
        "TORSDOF",
        "MODEL",
        "ENDMDL",
        "TER",
        "END",
    }

    _VALID_ELEMENTS = {
        "H",
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
        "Mg",
        "Zn",
        "Fe",
        "K",
        "Na",
        "Ca",
        "Mn",
        "Cu",
    }

    _VALID_PDBQT_TYPES = {
        "H",
        "HD",
        "HS",
        "C",
        "A",
        "N",
        "NA",
        "NS",
        "OA",
        "OS",
        "F",
        "P",
        "S",
        "SA",
        "Cl",
        "Br",
        "I",
        "Mg",
        "Zn",
        "Fe",
        "Ca",
        "Mn",
        "Cu",
        "Na",
        "K",
    }

    _TYPE_ALIAS_MAP_MEEKO: Dict[str, str] = {
        "CG0": "C",
        "G0": "C",
        "CG": "C",
        "G": "C",
        "C0": "C",
        "AA": "A",
        "OH": "OA",
        "OD": "OA",
        "SD": "SA",
        "HG": "HD",
        "HG1": "HD",
        "HA": "H",
        "HB": "H",
        "HC": "H",
        "CL1": "Cl",
        "BR1": "Br",
        "CL": "Cl",
        "BR": "Br",
        "MG": "Mg",
        "ZN": "Zn",
        "FE": "Fe",
        "OA": "OA",
        "OS": "OS",
        "HD": "HD",
        "A": "A",
        "NA": "NA",
        "SA": "SA",
        "C": "C",
        "N": "N",
        "O": "O",
        "S": "S",
        "H": "H",
        "F": "F",
        "P": "P",
        "I": "I",
        "K": "K",
        "NA+": "Na",
    }

    _TYPE_ALIAS_MAP_OBABEL: Dict[str, str] = {
        "CL": "Cl",
        "BR": "Br",
        "MG": "Mg",
        "ZN": "Zn",
        "FE": "Fe",
        "OA": "OA",
        "OS": "OS",
        "HD": "HD",
        "A": "A",
        "NA": "NA",
        "SA": "SA",
        "C": "C",
        "N": "N",
        "O": "O",
        "S": "S",
        "H": "H",
        "F": "F",
        "P": "P",
        "I": "I",
        "K": "K",
    }

    _ELEMENT_TO_DEFAULT_PDBQT = {
        "H": "H",
        "C": "C",
        "N": "N",
        "O": "O",
        "F": "F",
        "P": "P",
        "S": "S",
        "Cl": "Cl",
        "Br": "Br",
        "I": "I",
        "Mg": "Mg",
        "Zn": "Zn",
        "Fe": "Fe",
        "K": "K",
        "Na": "Na",
        "Ca": "Ca",
        "Mn": "Mn",
        "Cu": "Cu",
    }

    def __init__(
        self,
        path: Optional[str | Path] = None,
        *,
        backend: SanitizeBackend = "meeko",
    ) -> None:
        self._path: Optional[Path] = None if path is None else Path(path)
        self.backend: SanitizeBackend = backend
        self.lines: List[str] = []
        self.sanitized_lines: List[str] = []
        self.warnings: List[str] = []
        self._sanitized: bool = False

        if self._path is not None:
            self.read(self._path)

    # -------------------------
    # I/O
    # -------------------------
    def read(self, path: str | Path) -> "PDBQTSanitizer":
        """
        Load a PDBQT file into memory.

        :param path:
            Input file path.
        :type path: str | pathlib.Path

        :returns:
            Current instance.
        :rtype: PDBQTSanitizer
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        self._path = p
        text = p.read_text(encoding="utf-8", errors="replace")
        self.lines = text.splitlines()
        self.sanitized_lines = []
        self.warnings = []
        self._sanitized = False
        return self

    def write(self, out_path: str | Path) -> Path:
        """
        Write sanitized content to disk.

        :param out_path:
            Output file path.
        :type out_path: str | pathlib.Path

        :returns:
            Written path.
        :rtype: pathlib.Path
        """
        if not self._sanitized:
            raise RuntimeError("Call sanitize(...) before write().")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(self.sanitized_lines) + "\n", encoding="utf-8")
        logger.info("Wrote sanitized PDBQT to %s", out)
        return out

    def sanitize_inplace(
        self,
        rebuild: bool = True,
        aggressive: bool = False,
        backup: bool = True,
    ) -> Path:
        """
        Sanitize and overwrite the loaded file.

        :param rebuild:
            Rebuild ATOM/HETATM lines into fixed-width PDBQT format.
        :type rebuild: bool

        :param aggressive:
            Allow stronger fallback heuristics for malformed lines.
        :type aggressive: bool

        :param backup:
            Create ``.bak`` backup when overwriting.
        :type backup: bool

        :returns:
            Written path.
        :rtype: pathlib.Path
        """
        if self._path is None:
            raise RuntimeError("No file loaded. Call read(path) first.")
        self.sanitize(rebuild=rebuild, aggressive=aggressive)
        if backup:
            bak = self._path.with_suffix(self._path.suffix + ".bak")
            if not bak.exists():
                bak.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
                logger.debug("Created backup %s", bak)
        return self.write(self._path)

    @classmethod
    def sanitize_file(
        cls,
        path: str | Path,
        out_path: Optional[str | Path] = None,
        *,
        backend: SanitizeBackend = "meeko",
        rebuild: bool = True,
        aggressive: bool = False,
        backup: bool = True,
    ) -> Path:
        """
        Convenience wrapper for sanitizing a file.

        :param path:
            Input PDBQT path.
        :type path: str | pathlib.Path

        :param out_path:
            Output path. If ``None``, overwrite original.
        :type out_path: Optional[str | pathlib.Path]

        :param backend:
            Sanitizer behavior profile.
        :type backend: Literal["meeko", "obabel"]

        :param rebuild:
            Rebuild ATOM/HETATM lines into fixed-width PDBQT format.
        :type rebuild: bool

        :param aggressive:
            Allow stronger fallback heuristics for malformed lines.
        :type aggressive: bool

        :param backup:
            Create backup if overwriting.
        :type backup: bool

        :returns:
            Sanitized file path.
        :rtype: pathlib.Path
        """
        p = Path(path)
        s = cls(p, backend=backend)
        s.validate(strict=False)
        s.sanitize(rebuild=rebuild, aggressive=aggressive)

        if out_path is None:
            if backup:
                bak = p.with_suffix(p.suffix + ".bak")
                if not bak.exists():
                    bak.write_text("\n".join(s.lines) + "\n", encoding="utf-8")
            s.write(p)
            return p

        outp = Path(out_path)
        s.write(outp)
        return outp

    # -------------------------
    # Backend helpers
    # -------------------------
    def _type_alias_map(self) -> Dict[str, str]:
        if self.backend == "meeko":
            return self._TYPE_ALIAS_MAP_MEEKO
        return self._TYPE_ALIAS_MAP_OBABEL

    @staticmethod
    def _strip_digits(s: str) -> str:
        return re.sub(r"\d+", "", s)

    @staticmethod
    def _canonicalize_element(token: str) -> str:
        t = (token or "").strip()
        if not t:
            return ""
        if len(t) == 1:
            return t.upper()
        return t[0].upper() + t[1:].lower()

    @classmethod
    def _is_valid_element_token(cls, token: str) -> bool:
        return token in cls._VALID_ELEMENTS

    @classmethod
    def _is_valid_pdbqt_type(cls, token: str) -> bool:
        return token in cls._VALID_PDBQT_TYPES

    @classmethod
    def _looks_like_chain_id(cls, token: str) -> bool:
        t = (token or "").strip()
        if len(t) != 1:
            return False
        return not bool(cls._FLOAT_RE.match(t))

    @staticmethod
    def _to_float(s: str, default: float = 0.0) -> float:
        try:
            return float(s)
        except Exception:
            return default

    def _normalize_pdbqt_type(self, raw: str) -> str:
        """
        Normalize raw trailing token into a legacy-compatible PDBQT atom type.

        :param raw:
            Raw trailing token.
        :type raw: str

        :returns:
            Normalized PDBQT atom type or empty string.
        :rtype: str
        """
        r = (raw or "").strip()
        if not r:
            return ""

        if self._is_valid_pdbqt_type(r):
            return r

        amap = self._type_alias_map()
        if r in amap:
            return amap[r]

        r2 = self._strip_digits(r).upper()
        if r2 in amap:
            return amap[r2]

        elem = self._canonicalize_element(r2)
        if elem in self._ELEMENT_TO_DEFAULT_PDBQT:
            return self._ELEMENT_TO_DEFAULT_PDBQT[elem]

        return ""

    def _default_type_from_element(self, elem: str) -> str:
        """
        Convert chemical element to conservative default PDBQT type.

        :param elem:
            Element token.
        :type elem: str

        :returns:
            Default PDBQT type or empty string.
        :rtype: str
        """
        return self._ELEMENT_TO_DEFAULT_PDBQT.get(self._canonicalize_element(elem), "")

    def _infer_element_from_atom_name(self, atom_name: str) -> str:
        """
        Infer chemical element from atom name.

        :param atom_name:
            Atom name.
        :type atom_name: str

        :returns:
            Inferred element or empty string.
        :rtype: str
        """
        a = self._strip_digits(atom_name or "").strip()
        if not a:
            return ""

        cand2 = self._canonicalize_element(a[:2])
        if cand2 in self._VALID_ELEMENTS:
            return cand2

        cand1 = a[:1].upper()
        if cand1 in self._VALID_ELEMENTS:
            return cand1

        return ""

    # -------------------------
    # Parsing helpers
    # -------------------------
    def _parse_fixed_atom_line(self, ln: str) -> Dict[str, str]:
        """
        Parse a standard fixed-column PDBQT/PDB atom line.

        Returns an empty dict when the line is not plausibly fixed-column.
        """
        if not self._ATOM_RE.match(ln):
            return {}

        if len(ln) < 54:
            return {}

        x = ln[30:38].strip()
        y = ln[38:46].strip()
        z = ln[46:54].strip()

        # Only trust fixed-column parsing when x/y/z slices are all floats.
        if not (
            self._FLOAT_RE.match(x)
            and self._FLOAT_RE.match(y)
            and self._FLOAT_RE.match(z)
        ):
            return {}

        return {
            "record": ln[0:6].strip()
            or ("HETATM" if ln.startswith("HETATM") else "ATOM"),
            "serial": ln[6:11].strip() or "0",
            "atom_name": ln[12:16].strip(),
            "alt_loc": ln[16:17].strip(),
            "res_name": ln[17:20].strip() or "UNK",
            "chain_id": ln[21:22].strip(),
            "res_seq": ln[22:26].strip() or "1",
            "i_code": ln[26:27].strip(),
            "x": x,
            "y": y,
            "z": z,
            "occupancy": ln[54:60].strip() if len(ln) >= 60 else "",
            "temp_factor": ln[60:66].strip() if len(ln) >= 66 else "",
            "charge": ln[70:76].strip() if len(ln) >= 76 else "",
            "pdbqt_type": ln[78:80].strip() if len(ln) >= 80 else "",
        }

    def _extract_atom_fields(self, ln: str) -> Dict[str, str]:
        """
        Best-effort extraction of common atom fields.

        Prefer fixed-column parsing for real PDB/PDBQT lines and fall back
        to token parsing for minimal whitespace-delimited records.
        """
        fixed = self._parse_fixed_atom_line(ln)
        if fixed:
            fixed["float_idx"] = ""
            return fixed

        toks = ln.split()

        record = toks[0] if len(toks) > 0 else ""
        serial = toks[1] if len(toks) > 1 else "0"
        atom_name = toks[2] if len(toks) > 2 else ""
        res_name = toks[3] if len(toks) > 3 else "UNK"

        chain_id = ""
        res_seq = "1"

        if len(toks) > 5 and self._looks_like_chain_id(toks[4]):
            chain_id = toks[4]
            res_seq = toks[5]
        elif len(toks) > 4:
            res_seq = toks[4]

        float_idx: Optional[int] = None
        start = 6 if chain_id else 5
        for idx in range(start, len(toks) - 2):
            if (
                self._FLOAT_RE.match(toks[idx])
                and self._FLOAT_RE.match(toks[idx + 1])
                and self._FLOAT_RE.match(toks[idx + 2])
            ):
                float_idx = idx
                break

        return {
            "record": record,
            "serial": serial,
            "atom_name": atom_name,
            "alt_loc": "",
            "res_name": res_name,
            "chain_id": chain_id,
            "res_seq": res_seq,
            "i_code": "",
            "float_idx": "" if float_idx is None else str(float_idx),
        }

    def _extract_trailing_type(self, ln: str) -> str:
        """
        Extract likely trailing PDBQT atom type token.

        Prefer fixed columns 79-80 only when present.
        Otherwise fall back to token parsing.
        """
        fixed = self._parse_fixed_atom_line(ln)
        if fixed and fixed["pdbqt_type"]:
            return fixed["pdbqt_type"]

        toks = ln.split()
        if not toks:
            return ""

        for tok in reversed(toks[1:]):
            if not self._FLOAT_RE.match(tok):
                return tok
        return ""

    def _extract_fixed_element(self, ln: str) -> str:
        """
        Extract fixed-column element from columns 77-78 if present.

        :param ln:
            Input ATOM/HETATM line.
        :type ln: str

        :returns:
            Fixed-column element token or empty string.
        :rtype: str
        """
        if len(ln) >= 78:
            return ln[76:78].strip()
        return ""

    def _extract_xyz_and_tail_floats(
        self,
        ln: str,
    ) -> tuple[Optional[float], Optional[float], Optional[float], List[str], List[str]]:
        """
        Extract x/y/z and floating tail after z.

        For common ligand PDBQT forms, this accepts patterns like:
        - ATOM ... x y z q type
        - ATOM ... x y z occ temp type
        - ATOM ... x y z occ temp charge type
        """
        fixed = self._parse_fixed_atom_line(ln)
        if fixed:
            try:
                x = float(fixed["x"])
                y = float(fixed["y"])
                z = float(fixed["z"])
            except Exception:
                return None, None, None, [], ln.split()

            float_tail: List[str] = []
            for key in ("occupancy", "temp_factor", "charge"):
                val = fixed.get(key, "")
                if val and self._FLOAT_RE.match(val):
                    float_tail.append(val)

            return x, y, z, float_tail, ln.split()

        toks = ln.split()
        fields = self._extract_atom_fields(ln)
        float_idx_str = fields["float_idx"]
        if float_idx_str == "":
            return None, None, None, [], toks

        idx = int(float_idx_str)
        if idx + 2 >= len(toks):
            return None, None, None, [], toks

        try:
            x = float(toks[idx])
            y = float(toks[idx + 1])
            z = float(toks[idx + 2])
        except Exception:
            return None, None, None, [], toks

        tail = toks[idx + 3 :]  # noqa
        float_tail = [t for t in tail if self._FLOAT_RE.match(t)]
        return x, y, z, float_tail, toks

    def _extract_charge_occ_temp(self, ln: str) -> tuple[float, float, float]:
        """
        Extract ``charge``, ``occupancy``, and ``tempFactor`` from a line.

        Heuristic rules
        ---------------
        - 1 float after z:
            treat as charge
        - 2 floats after z:
            treat as occupancy, tempFactor; charge=0.0
        - 3+ floats after z:
            treat as occupancy, tempFactor, charge using first three floats
        """
        fixed = self._parse_fixed_atom_line(ln)
        if fixed:
            occ = self._to_float(fixed.get("occupancy", ""), 0.0)
            temp = self._to_float(fixed.get("temp_factor", ""), 0.0)
            charge = self._to_float(fixed.get("charge", ""), 0.0)
            return charge, occ, temp

        _, _, _, float_tail, _ = self._extract_xyz_and_tail_floats(ln)

        if len(float_tail) == 1:
            return self._to_float(float_tail[0]), 0.00, 0.00
        if len(float_tail) == 2:
            return 0.00, self._to_float(float_tail[0]), self._to_float(float_tail[1])
        if len(float_tail) >= 3:
            return (
                self._to_float(float_tail[2]),
                self._to_float(float_tail[0]),
                self._to_float(float_tail[1]),
            )

        return 0.00, 0.00, 0.00

    def _choose_pdbqt_type(
        self,
        *,
        trailing_tok: str,
        fixed_elem: str,
        atom_name: str,
        aggressive: bool = False,
    ) -> str:
        """
        Choose the best PDBQT atom type.

        Priority
        --------
        1. valid trailing PDBQT type
        2. normalized trailing type
        3. default type from valid fixed element
        4. default type from atom-name-inferred element
        5. ``C`` fallback if aggressive=True
        """
        if trailing_tok and self._is_valid_pdbqt_type(trailing_tok):
            return trailing_tok

        mapped = self._normalize_pdbqt_type(trailing_tok)
        if mapped:
            return mapped

        if fixed_elem and self._is_valid_element_token(fixed_elem):
            from_elem = self._default_type_from_element(fixed_elem)
            if from_elem:
                return from_elem

        inferred = self._infer_element_from_atom_name(atom_name)
        if inferred:
            from_name = self._default_type_from_element(inferred)
            if from_name:
                return from_name

        if aggressive:
            return "C"

        return ""

    # -------------------------
    # Formatting
    # -------------------------
    def _format_atom_name(self, atom_name: str, element: str = "") -> str:
        """
        Format atom name into 4-character PDB atom-name field.

        :param atom_name:
            Atom name.
        :type atom_name: str

        :param element:
            Optional element hint.
        :type element: str

        :returns:
            4-character atom name.
        :rtype: str
        """
        name = (atom_name or "").strip()[:4]
        if len(name) == 4:
            return name
        if len(name) == 1:
            return f" {name}  "
        if len(name) == 2:
            return f" {name} "
        if len(name) == 3:
            return f" {name}"
        return "    "

    def _rebuild_atom_line(
        self,
        *,
        record: str,
        serial: str,
        atom_name: str,
        res_name: str,
        chain_id: str,
        res_seq: str,
        alt_loc: str,
        i_code: str,
        x: float,
        y: float,
        z: float,
        occupancy: float,
        temp_factor: float,
        charge: float,
        pdbqt_type: str,
    ) -> str:
        """
        Rebuild a fixed-width, qvina-friendly PDBQT ATOM/HETATM line.
        """
        try:
            serial_i = int(serial)
        except Exception:
            serial_i = 0

        try:
            res_seq_i = int(res_seq)
        except Exception:
            res_seq_i = 1

        atype = (pdbqt_type or "C")[:2]
        element_hint = self._canonicalize_element(self._strip_digits(atype))
        atom_name_fmt = self._format_atom_name(atom_name, element_hint)
        res_name_fmt = (res_name or "UNL")[:3]

        alt_loc = (alt_loc or " ")[:1]
        chain_id = (chain_id or " ")[:1]
        i_code = (i_code or " ")[:1]

        line = (
            f"{record:<6}"
            f"{serial_i:>5d}"
            f" "
            f"{atom_name_fmt}"
            f"{alt_loc}"
            f"{res_name_fmt:>3s}"
            f" "
            f"{chain_id}"
            f"{res_seq_i:>4d}"
            f"{i_code}"
            f"   "
            f"{x:>8.3f}"
            f"{y:>8.3f}"
            f"{z:>8.3f}"
            f"{occupancy:>6.2f}"
            f"{temp_factor:>6.2f}"
            f"    "
            f"{charge:>6.3f}"
            f"  "
            f"{atype:>2s}"
        )
        return line

    # -------------------------
    # Validation
    # -------------------------
    def validate(self, strict: bool = False) -> List[str]:
        """
        Validate the loaded PDBQT and collect warnings.

        :param strict:
            Emit stronger warnings when type inference is needed.
        :type strict: bool

        :returns:
            Warning list.
        :rtype: List[str]
        """
        if not self.lines:
            raise RuntimeError("No file loaded. Call read(path) first.")

        self.warnings = []

        for i, ln in enumerate(self.lines, start=1):
            if not ln.strip():
                continue

            if not self._ATOM_RE.match(ln):
                first = ln.strip().split()[0]
                if (
                    first.isalpha()
                    and first not in self._TAG_WHITELIST
                    and not first.isdigit()
                ):
                    self.warnings.append(f"Line {i}: unknown top-level tag '{first}'")
                continue

            x, y, z, _, _ = self._extract_xyz_and_tail_floats(ln)
            if x is None or y is None or z is None:
                self.warnings.append(f"Line {i}: could not parse x/y/z coordinates")
                continue

            atom_name = self._extract_atom_fields(ln).get("atom_name", "")
            fixed_elem = self._extract_fixed_element(ln)
            trailing = self._extract_trailing_type(ln)

            if trailing:
                if not self._is_valid_pdbqt_type(trailing):
                    mapped = self._normalize_pdbqt_type(trailing)
                    if mapped:
                        self.warnings.append(
                            f"Line {i}: trailing PDBQT type '{trailing}' is non-canonical; suggested='{mapped}'."
                        )
                    else:
                        self.warnings.append(
                            f"Line {i}: trailing token '{trailing}' is not a recognized PDBQT type."
                        )
            elif strict:
                guess = self._choose_pdbqt_type(
                    trailing_tok="",
                    fixed_elem=fixed_elem,
                    atom_name=atom_name,
                    aggressive=False,
                )
                if guess:
                    self.warnings.append(
                        f"Line {i}: missing trailing PDBQT atom type; suggested='{guess}'."
                    )
                else:
                    self.warnings.append(
                        f"Line {i}: missing trailing PDBQT atom type and could not infer one."
                    )

        return list(self.warnings)

    # -------------------------
    # Sanitization
    # -------------------------
    def sanitize(
        self,
        rebuild: bool = True,
        aggressive: bool = False,
    ) -> "PDBQTSanitizer":
        """
        Produce sanitized content.

        For older qvina, ``rebuild=True`` is strongly recommended.

        :param rebuild:
            Rebuild ATOM/HETATM lines into fixed-width PDBQT format.
        :type rebuild: bool

        :param aggressive:
            Allow stronger fallback heuristics for atom-type inference.
        :type aggressive: bool

        :returns:
            Current instance.
        :rtype: PDBQTSanitizer
        """
        if not self.lines:
            raise RuntimeError("No file loaded. Call read(path) first.")

        out_lines: List[str] = []
        self.warnings = []

        for i, ln in enumerate(self.lines, start=1):
            if not ln.strip():
                out_lines.append(ln)
                continue

            if not self._ATOM_RE.match(ln):
                out_lines.append(ln)
                continue

            toks = ln.split()
            if len(toks) < 6 and not self._parse_fixed_atom_line(ln):
                out_lines.append(ln)
                self.warnings.append(f"Line {i}: short ATOM/HETATM left unchanged")
                continue

            fields = self._extract_atom_fields(ln)
            record = fields["record"]
            serial = fields["serial"]
            atom_name = fields["atom_name"]
            res_name = fields["res_name"]
            chain_id = fields.get("chain_id", "")
            res_seq = fields["res_seq"]
            alt_loc = fields.get("alt_loc", "")
            i_code = fields.get("i_code", "")

            x, y, z, _, _ = self._extract_xyz_and_tail_floats(ln)
            if x is None or y is None or z is None:
                out_lines.append(ln)
                self.warnings.append(
                    f"Line {i}: could not parse coordinates, left unchanged"
                )
                continue

            trailing = self._extract_trailing_type(ln)
            fixed_elem = self._extract_fixed_element(ln)
            pdbqt_type = self._choose_pdbqt_type(
                trailing_tok=trailing,
                fixed_elem=fixed_elem,
                atom_name=atom_name,
                aggressive=aggressive,
            )

            if not pdbqt_type:
                out_lines.append(ln)
                self.warnings.append(
                    f"Line {i}: could not infer PDBQT atom type, left unchanged"
                )
                continue

            charge, occupancy, temp_factor = self._extract_charge_occ_temp(ln)

            rebuilt = self._rebuild_atom_line(
                record=record,
                serial=serial,
                atom_name=atom_name,
                res_name=res_name,
                chain_id=chain_id,
                res_seq=res_seq,
                alt_loc=alt_loc,
                i_code=i_code,
                x=x,
                y=y,
                z=z,
                occupancy=occupancy,
                temp_factor=temp_factor,
                charge=charge,
                pdbqt_type=pdbqt_type,
            )
            out_lines.append(rebuilt)

            if rebuilt.rstrip() != ln.rstrip():
                if rebuild:
                    self.warnings.append(
                        f"Line {i}: rebuilt ATOM/HETATM; "
                        f"charge={charge:.3f} type='{pdbqt_type}' backend='{self.backend}'"
                    )
                else:
                    self.warnings.append(
                        f"Line {i}: normalized ATOM/HETATM in fixed-width mode; "
                        f"charge={charge:.3f} type='{pdbqt_type}' backend='{self.backend}'"
                    )

        self.sanitized_lines = out_lines
        self._sanitized = True
        return self

    def __repr__(self) -> str:
        """
        Return concise object representation.

        :returns:
            Representation string.
        :rtype: str
        """
        return (
            f"<PDBQTSanitizer path={self._path.name if self._path else None} "
            f"backend={self.backend} lines={len(self.lines)} sanitized={self._sanitized}>"
        )

    def help(self) -> str:
        """
        Return brief usage help.

        :returns:
            Usage string.
        :rtype: str
        """
        return (
            "PDBQTSanitizer(path, backend='meeko'|'obabel'). "
            "validate(strict=False) -> warnings; "
            "sanitize(rebuild=True) -> produce sanitized_lines; "
            "write(path) -> save file."
        )
