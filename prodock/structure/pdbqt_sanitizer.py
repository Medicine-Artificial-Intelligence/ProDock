from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Optional, Literal

from prodock.io.logging import get_logger

logger = get_logger(__name__)

SanitizeBackend = Literal["meeko", "obabel"]


class PDBQTSanitizer:
    """
    Backend-aware PDBQT sanitizer and validator.

    This sanitizer is tuned for backend-specific behavior:

    - ``backend="meeko"``:
      more tolerant alias mapping and more active repair of atom-type-like
      trailing tokens such as ``OA``, ``HD``, ``CG0``, ``G0``, ``A``, ``CL1``,
      and related variants.

    - ``backend="obabel"``:
      lighter validation and sanitization, preferring canonical element fields
      and avoiding unnecessary rewriting unless something is clearly malformed.

    Features
    --------
    - validation of fixed-column element fields (PDB columns 77--78)
    - validation of trailing atom-type-like tokens
    - backend-aware alias mapping
    - optional rebuilding of ``ATOM`` / ``HETATM`` records into fixed-width
      PDB-style lines
    - in-place overwrite with optional backup

    Example
    -------
    .. code-block:: python

        sanitizer = PDBQTSanitizer("ligand.pdbqt", backend="meeko")
        warnings = sanitizer.validate(strict=True)
        sanitizer.sanitize(rebuild=True, aggressive=False)
        sanitizer.write("ligand.sanitized.pdbqt")

    :param path:
        Optional path to a PDBQT file to load immediately.
    :type path: Optional[str | pathlib.Path]

    :param backend:
        Sanitizer behavior profile.
    :type backend: Literal["meeko", "obabel"]
    """

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
    }

    _ATOM_RE = re.compile(r"^(ATOM|HETATM)\b")
    _FLOAT_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")
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

    _COMMON_ALIAS_MAP_MEEKO: Dict[str, str] = {
        "OA": "O",
        "OH": "O",
        "OD": "O",
        "OS": "O",
        "HD": "H",
        "HG": "H",
        "HG1": "H",
        "HA": "H",
        "HB": "H",
        "HC": "H",
        "CG": "C",
        "CG0": "C",
        "G0": "C",
        "G": "C",
        "A": "C",
        "AA": "C",
        "C0": "C",
        "NA": "N",
        "N0": "N",
        "SA": "S",
        "SD": "S",
        "CL": "Cl",
        "CL1": "Cl",
        "BR": "Br",
        "BR1": "Br",
    }

    _COMMON_ALIAS_MAP_OBABEL: Dict[str, str] = {
        "CL": "Cl",
        "BR": "Br",
        "NA": "Na",
        "CA": "Ca",
        "MG": "Mg",
        "ZN": "Zn",
        "FE": "Fe",
        "K": "K",
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
    # I/O helpers
    # -------------------------
    def read(self, path: str | Path) -> "PDBQTSanitizer":
        """
        Load a PDBQT file into memory.
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
        Write sanitized content.
        """
        if not self._sanitized:
            raise RuntimeError("Call sanitize(...) before write().")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(self.sanitized_lines) + "\n", encoding="utf-8")
        logger.info("Wrote sanitized PDBQT to %s", out)
        return out

    def set_backend(self, backend: SanitizeBackend) -> "PDBQTSanitizer":
        """
        Set sanitizer backend profile.
        """
        self.backend = backend
        return self

    def sanitize_inplace(
        self,
        rebuild: bool = True,
        aggressive: bool = False,
        backup: bool = True,
    ) -> Path:
        """
        Sanitize and overwrite original file.
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

    # -------------------------
    # backend helpers
    # -------------------------
    def _alias_map(self) -> Dict[str, str]:
        if self.backend == "meeko":
            return self._COMMON_ALIAS_MAP_MEEKO
        if self.backend == "obabel":
            return self._COMMON_ALIAS_MAP_OBABEL
        return {}

    def _prefer_trailing_mapping(self) -> bool:
        return self.backend == "meeko"

    def _default_fallback_element(self) -> str:
        return "C"

    # -------------------------
    # internal heuristics
    # -------------------------
    @classmethod
    def _canonicalize_element(cls, token: str) -> str:
        t = (token or "").strip()
        if not t:
            return ""
        if len(t) == 1:
            return t.upper()
        return t[0].upper() + t[1:].lower()

    @classmethod
    def _strip_digits(cls, s: str) -> str:
        return re.sub(r"\d+", "", s)

    def _is_valid_element_token(self, token: str) -> bool:
        if not token:
            return False
        t = token.strip()
        m = re.fullmatch(r"[A-Z][a-z]?", t)
        return bool(m and t in self._VALID_ELEMENTS)

    def _map_alias(self, raw: str, atomname: str = "") -> str:
        """
        Map raw token to element using backend-aware alias maps and heuristics.
        """
        r = (raw or "").strip()
        if not r:
            return ""

        amap = self._alias_map()

        if r in amap:
            return amap[r]

        r2 = self._strip_digits(r).upper()
        if r2 in amap:
            return amap[r2]

        can = self._canonicalize_element(r2)
        if can in self._VALID_ELEMENTS:
            return can

        cand2 = r2[:2].capitalize()
        if cand2 in self._VALID_ELEMENTS:
            return cand2

        cand1 = r2[:1].upper()
        if cand1 in self._VALID_ELEMENTS:
            return cand1

        if atomname:
            a = self._strip_digits(atomname).strip()
            if a:
                can_atom = self._canonicalize_element(a[:2])
                if can_atom in self._VALID_ELEMENTS:
                    return can_atom
                one = a[:1].upper()
                if one in self._VALID_ELEMENTS:
                    return one

        return ""

    def _extract_atom_fields(self, ln: str) -> Dict[str, str]:
        """
        Best-effort extraction of common atom fields from a whitespace-split PDBQT line.
        """
        toks = ln.split()

        record = toks[0] if len(toks) > 0 else ""
        serial = toks[1] if len(toks) > 1 else "0"
        atom_name = toks[2] if len(toks) > 2 else ""
        res_name = toks[3] if len(toks) > 3 else ""
        res_seq = toks[4] if len(toks) > 4 else "1"

        float_idx = None
        for idx in range(5, len(toks)):
            if self._FLOAT_RE.match(toks[idx]):
                float_idx = idx
                break

        return {
            "record": record,
            "serial": serial,
            "atom_name": atom_name,
            "res_name": res_name,
            "res_seq": res_seq,
            "float_idx": str(float_idx) if float_idx is not None else "",
        }

    def _extract_trailing_token(self, ln: str) -> str:
        """
        Try to extract the most likely trailing atom-type token.
        """
        toks = ln.split()
        if not toks:
            return ""

        float_idx = None
        for idx in range(5, len(toks)):
            if self._FLOAT_RE.match(toks[idx]):
                float_idx = idx
                break

        if float_idx is not None:
            tail_start = float_idx + 5
            if tail_start < len(toks):
                last = toks[-1]
                if not self._FLOAT_RE.match(last):
                    return last
                if len(toks) >= 2 and not self._FLOAT_RE.match(toks[-2]):
                    return toks[-2]
                return ""

        last = toks[-1]
        if not self._FLOAT_RE.match(last):
            return last
        return ""

    def _fixed_element(self, ln: str) -> str:
        if len(ln) >= 78:
            return ln[76:78].strip()
        return ""

    def _choose_element(
        self,
        *,
        fixed_elem: str,
        trailing_tok: str,
        atom_name: str,
    ) -> str:
        """
        Backend-aware element selection.
        """
        if self._is_valid_element_token(fixed_elem):
            return fixed_elem

        if self._prefer_trailing_mapping() and trailing_tok:
            mapped = self._map_alias(trailing_tok, atomname=atom_name)
            if mapped:
                return mapped

        mapped_atom = self._map_alias(atom_name, atomname=atom_name)
        if mapped_atom:
            return mapped_atom

        if trailing_tok:
            mapped = self._map_alias(trailing_tok, atomname=atom_name)
            if mapped:
                return mapped

        return self._default_fallback_element()

    # -------------------------
    # Validation
    # -------------------------
    def validate(self, strict: bool = False) -> List[str]:
        """
        Validate the loaded PDBQT and collect warnings.
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

            atom_name = ln[12:16].strip() if len(ln) >= 16 else ""
            fixed_element = self._fixed_element(ln)
            trailing = self._extract_trailing_token(ln)

            if fixed_element:
                if not self._is_valid_element_token(fixed_element):
                    suggestion = (
                        self._map_alias(trailing, atomname=atom_name) or "<none>"
                    )
                    self.warnings.append(
                        f"Line {i}: fixed-column element token '{fixed_element}' is invalid; "
                        f"suggested='{suggestion}'."
                    )
                elif strict and trailing:
                    if not self._is_valid_element_token(trailing):
                        mapped = self._map_alias(trailing, atomname=atom_name)
                        if mapped and mapped != fixed_element:
                            self.warnings.append(
                                f"Line {i}: trailing token '{trailing}' differs from "
                                f"fixed element '{fixed_element}'; mapped='{mapped}'."
                            )
            else:
                if trailing:
                    if self._is_valid_element_token(trailing):
                        pass
                    else:
                        mapped = self._map_alias(trailing, atomname=atom_name)
                        if mapped:
                            self.warnings.append(
                                f"Line {i}: trailing token '{trailing}' is non-canonical; "
                                f"suggested='{mapped}'."
                            )
                        else:
                            self.warnings.append(
                                f"Line {i}: trailing token '{trailing}' cannot be mapped; "
                                f"atom='{atom_name}'."
                            )
                elif strict:
                    self.warnings.append(
                        f"Line {i}: no element detected (fixed-column or trailing)."
                    )

            if atom_name and not re.match(r"^[A-Za-z0-9_\-\.]+$", atom_name):
                self.warnings.append(f"Line {i}: suspicious atom name '{atom_name}'.")

        return list(self.warnings)

    # -------------------------
    # Sanitization / Rebuild
    # -------------------------
    def sanitize(
        self,
        rebuild: bool = True,
        aggressive: bool = False,
    ) -> "PDBQTSanitizer":
        """
        Produce sanitized content.

        Notes
        -----
        backend="meeko":
            more likely to rewrite trailing atom-type aliases into a canonical element

        backend="obabel":
            prefers existing valid fixed-column element when available and avoids
            unnecessary changes
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
            if len(toks) < 6:
                out_lines.append(ln)
                self.warnings.append(f"Line {i}: short ATOM/HETATM left unchanged")
                continue

            fields = self._extract_atom_fields(ln)
            record = fields["record"]
            serial = fields["serial"]
            atom_name = fields["atom_name"]
            res_name = fields["res_name"]
            res_seq = fields["res_seq"]

            float_idx_str = fields["float_idx"]
            if float_idx_str == "":
                out_lines.append(ln)
                self.warnings.append(
                    f"Line {i}: cannot parse coordinates, left unchanged"
                )
                continue

            float_idx = int(float_idx_str)
            if float_idx + 2 >= len(toks):
                out_lines.append(ln)
                self.warnings.append(
                    f"Line {i}: incomplete coordinates, left unchanged"
                )
                continue

            try:
                x = float(toks[float_idx])
                y = float(toks[float_idx + 1])
                z = float(toks[float_idx + 2])
            except Exception:
                out_lines.append(ln)
                self.warnings.append(
                    f"Line {i}: invalid numeric coordinates, left unchanged"
                )
                continue

            occ = toks[float_idx + 3] if len(toks) > float_idx + 3 else "0.00"
            temp = toks[float_idx + 4] if len(toks) > float_idx + 4 else "0.00"
            trailing = (
                toks[float_idx + 5 :] if len(toks) > float_idx + 5 else []  # noqa
            )  # noqa
            trailing_tok = trailing[-1] if trailing else ""

            fixed_elem = self._fixed_element(ln)
            element = self._choose_element(
                fixed_elem=fixed_elem,
                trailing_tok=trailing_tok,
                atom_name=atom_name,
            )

            if aggressive:
                up = element.upper()
                if up == "CL":
                    element = "Cl"
                elif up == "BR":
                    element = "Br"
                elif up == "NA" and self.backend == "obabel":
                    element = "Na"
                elif up == "CA" and self.backend == "obabel":
                    element = "Ca"
                elif up == "MG":
                    element = "Mg"
                elif up == "ZN":
                    element = "Zn"
                elif up == "FE":
                    element = "Fe"

            if rebuild:
                try:
                    serial_i = int(serial)
                except Exception:
                    serial_i = 0
                try:
                    res_seq_i = int(res_seq)
                except Exception:
                    res_seq_i = 1
                try:
                    occ_f = float(occ)
                except Exception:
                    occ_f = 0.00
                try:
                    temp_f = float(temp)
                except Exception:
                    temp_f = 0.00

                name_fmt = atom_name if len(atom_name) <= 4 else atom_name[:4]
                alt_loc = " "
                chain_id = " "
                i_code = " "

                base = (
                    f"{record:<6}{serial_i:>5} {name_fmt:^4}{alt_loc}{res_name:>3}"
                    f" {chain_id}{res_seq_i:>4}{i_code}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occ_f:6.2f}{temp_f:6.2f}"
                )

                if len(base) < 76:
                    base = base + " " * (76 - len(base))

                element_field = f"{element:>2}"
                rebuilt = base[:76] + element_field
                out_lines.append(rebuilt)

                if rebuilt.rstrip() != ln.rstrip():
                    self.warnings.append(
                        f"Line {i}: rebuilt ATOM/HETATM; element='{element}' backend='{self.backend}'"
                    )
            else:
                if trailing_tok:
                    mapped = self._map_alias(trailing_tok, atomname=atom_name)
                    should_replace = (
                        mapped and mapped != trailing_tok and self.backend == "meeko"
                    )
                    if should_replace:
                        core = " ".join(toks[: float_idx + 5])
                        newln = core + " " + mapped
                        out_lines.append(newln)
                        self.warnings.append(
                            f"Line {i}: replaced trailing '{trailing_tok}' -> '{mapped}'"
                        )
                    else:
                        out_lines.append(ln)
                else:
                    out_lines.append(ln)

        self.sanitized_lines = out_lines
        self._sanitized = True
        return self

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

        :param path: input PDBQT
        :param out_path: output path; if None, overwrite original
        :param backend: "meeko" or "obabel"
        :param rebuild: rebuild ATOM/HETATM records
        :param aggressive: apply stronger capitalization/element normalization
        :param backup: create .bak when overwriting
        :returns: path to sanitized file
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

    def __repr__(self) -> str:
        return (
            f"<PDBQTSanitizer path={self._path.name if self._path else None} "
            f"backend={self.backend} lines={len(self.lines)} sanitized={self._sanitized}>"
        )

    def help(self) -> str:
        return (
            "PDBQTSanitizer(path, backend='meeko'|'obabel').validate(strict=False) -> warnings; "
            "sanitize(rebuild=True) -> produce sanitized_lines; write(path) -> save file."
        )
