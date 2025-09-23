# prodock/io/pdbqt_sanitizer.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Optional

from prodock.io.logging import get_logger

logger = get_logger(__name__)


class PDBQTSanitizer:
    """
    PDBQT sanitizer and validator tuned for Meeko -> smina/qvina compatibility.

    This sanitizer offers:
      - strict validation of fixed-column element (PDB cols 77-78) and trailing tokens
      - conservative alias mapping for Meeko/rdkit/OpenBabel artifacts (e.g. CG0, OA, HD)
      - ability to *rebuild* ATOM/HETATM into fixed-width PDB lines with an element
        in the fixed-column so smina/qvina's stricter parser accepts the file.
      - option to run in strict audit mode (warn but don't change) or aggressive sanitize.

    Save as: prodock/io/pdbqt_sanitizer.py

    Typical usage::

        s = PDBQTSanitizer("ligand.pdbqt")
        warnings = s.validate(strict=True)
        s.sanitize(rebuild=True, aggressive=False)
        s.write("ligand.sanitized.pdbqt")

    :param path: optional path to PDBQT to read immediately (if provided).
    :type path: Optional[str | pathlib.Path]
    """

    # conservative list of element symbols acceptable in element column
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

    # default alias map (conservative — edit for your project)
    _COMMON_ALIAS_MAP: Dict[str, str] = {
        "OA": "O",
        "OH": "O",
        "OD": "O",
        "HD": "H",
        "HG": "H",
        "HG1": "H",
        "HA": "H",
        "HB": "H",
        "CG": "C",
        "CG0": "C",
        "G0": "C",
        "G": "C",
        "A": "C",
        "AA": "C",
        "NA": "N",
        "CL": "Cl",
        "CL1": "Cl",
        "BR": "Br",
    }

    _ATOM_RE = re.compile(r"^(ATOM|HETATM)\b")
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

    def __init__(self, path: Optional[str | Path] = None) -> None:
        """
        Create sanitizer instance and optionally load file.

        :param path: optional path to PDBQT to read now.
        :type path: Optional[str | pathlib.Path]
        :returns: None
        :rtype: None
        :example:

        >>> s = PDBQTSanitizer("ligand.pdbqt")
        >>> isinstance(s, PDBQTSanitizer)
        True
        """
        self._path: Optional[Path] = None if path is None else Path(path)
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

        :param path: path to PDBQT file.
        :type path: str | pathlib.Path
        :returns: self
        :rtype: PDBQTSanitizer
        :raises FileNotFoundError: if the provided path does not exist
        :example:

        >>> s = PDBQTSanitizer()
        >>> s.read("ligand.pdbqt")
        <PDBQTSanitizer ...>
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
        Write sanitized content (must call sanitize() first).

        :param out_path: file path to write sanitized file.
        :type out_path: str | pathlib.Path
        :returns: Path to written file.
        :rtype: pathlib.Path
        :raises RuntimeError: if sanitize() has not been called
        :example:

        >>> s = PDBQTSanitizer("ligand.pdbqt")
        >>> s.sanitize()
        >>> s.write("ligand.sanitized.pdbqt")
        PosixPath('ligand.sanitized.pdbqt')
        """
        if not self._sanitized:
            raise RuntimeError("Call sanitize(rebuild=True/False) before write().")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(self.sanitized_lines) + "\n", encoding="utf-8")
        logger.info("Wrote sanitized PDBQT to %s", out)
        return out

    def sanitize_inplace(
        self, rebuild: bool = True, aggressive: bool = False, backup: bool = True
    ) -> Path:
        """
        Sanitize and overwrite original file.

        :param rebuild: if True, rebuild ATOM/HETATM into fixed-width PDB format.
        :type rebuild: bool
        :param aggressive: if True, apply aggressive alias heuristics (e.g. NA -> Na).
        :type aggressive: bool
        :param backup: if True, create .bak copy before overwrite.
        :type backup: bool
        :returns: Path to overwritten file.
        :rtype: pathlib.Path
        :raises RuntimeError: if no file has been loaded via read() or constructor.
        :example:

        >>> s = PDBQTSanitizer("ligand.pdbqt")
        >>> s.sanitize_inplace(rebuild=True, aggressive=False, backup=True)
        PosixPath('ligand.pdbqt')
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
    # internal heuristics
    # -------------------------
    @classmethod
    def _canonicalize_element(cls, token: str) -> str:
        """
        Normalize capitalization of element token (e.g. 'cl'->'Cl').

        :param token: raw token.
        :type token: str
        :returns: canonicalized token or empty string.
        :rtype: str
        """
        t = (token or "").strip()
        if not t:
            return ""
        if len(t) == 1:
            return t.upper()
        return t[0].upper() + t[1:].lower()

    @classmethod
    def _strip_digits(cls, s: str) -> str:
        """
        Remove digits from string (used to map tokens like CG0 -> CG).

        :param s: string
        :type s: str
        :returns: string without digits
        :rtype: str
        """
        return re.sub(r"\d+", "", s)

    def _map_alias(self, raw: str, atomname: str = "") -> str:
        """
        Map a raw token to a likely element using alias map and heuristics.

        The mapping uses the conservative _COMMON_ALIAS_MAP, numeric-stripping,
        capitalization heuristics, and finally a fallback to the first letter of
        the atom name when reasonable.

        :param raw: raw trailing token or atomname.
        :type raw: str
        :param atomname: atom name (fallback).
        :type atomname: str
        :returns: element symbol or empty string if none found.
        :rtype: str
        """
        r = (raw or "").strip()
        if not r:
            return ""
        if r in self._COMMON_ALIAS_MAP:
            return self._COMMON_ALIAS_MAP[r]
        r2 = self._strip_digits(r).upper()
        if r2 in self._COMMON_ALIAS_MAP:
            return self._COMMON_ALIAS_MAP[r2]
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
            a = atomname[0].upper()
            if a in self._VALID_ELEMENTS:
                return a
        return ""

    def _is_valid_element_token(self, token: str) -> bool:
        """
        Exact check if token is an allowed element symbol.

        :param token: token
        :type token: str
        :returns: True if token exactly matches allowed element
        :rtype: bool
        """
        if not token:
            return False
        m = re.fullmatch(r"[A-Z][a-z]?", token.strip())
        return bool(m and token in self._VALID_ELEMENTS)

    # -------------------------
    # Validation
    # -------------------------
    def validate(self, strict: bool = False) -> List[str]:
        """
        Validate the loaded PDBQT and collect warnings.

        This inspects:
          - fixed-column PDB element (cols 77-78) if present
          - trailing token styles (Meeko-style tokens after coords)
          - strange atom names and unknown top-level tags

        :param strict: if True, warn on any non-canonical trailing token (audit mode).
        :type strict: bool
        :returns: list of warning strings (may be empty).
        :rtype: List[str]
        :raises RuntimeError: if no file has been loaded via read() or constructor.
        :example:

        >>> s = PDBQTSanitizer("ligand.pdbqt")
        >>> warnings = s.validate(strict=True)
        >>> isinstance(warnings, list)
        True
        """
        if not self.lines:
            raise RuntimeError("No file loaded. Call read(path) first.")

        self.warnings = []
        float_re = re.compile(r"^[+-]?\d+(\.\d+)?$")

        for i, ln in enumerate(self.lines, start=1):
            if not ln.strip():
                continue
            # Non-ATOM top-level tags check
            if not self._ATOM_RE.match(ln):
                first = ln.strip().split()[0]
                if (
                    first.isalpha()
                    and first not in self._TAG_WHITELIST
                    and not first.isdigit()
                ):
                    self.warnings.append(f"Line {i}: unknown top-level tag '{first}'")
                continue

            # For ATOM/HETATM lines, attempt to read fixed-column element if present
            fixed_element = ""
            if len(ln) >= 78:
                fixed_element = ln[76:78].strip()
            # atom-name attempt (cols 13-16 PDB) if line long enough
            atom_name = ln[12:16].strip() if len(ln) >= 16 else ""
            # whitespace trailing token attempt
            toks = ln.split()
            trailing = ""
            # find first float idx after typical residue tokens
            # assume tokens: RECORD serial name resName resSeq x y z occ temp [more...]
            float_idx = None
            for idx in range(5, len(toks)):
                if float_re.match(toks[idx]):
                    float_idx = idx
                    break
            if float_idx is not None and float_idx + 2 < len(toks):
                # trailing tokens are anything after z, occ, temp (i.e. from float_idx+5 onward)
                tail_start = float_idx + 5
                if tail_start < len(toks):
                    trailing = toks[-1]  # last token is typical candidate
            else:
                # fallback: if last token not numeric, consider it trailing
                if not float_re.match(toks[-1]):
                    trailing = toks[-1]

            # First check fixed-column element
            if fixed_element:
                if not self._is_valid_element_token(fixed_element):
                    suggestion = (
                        self._map_alias(trailing, atomname=atom_name) or "<none>"
                    )
                    self.warnings.append(
                        f"Line {i}: fixed-column element token '{fixed_element}' is not valid"
                        + f". Suggested element (map from trailing/atom): '{suggestion}'."
                    )
                else:
                    # if valid but trailing token present and strict, warn about mismatch
                    if (
                        strict
                        and trailing
                        and not self._is_valid_element_token(trailing)
                        and trailing != fixed_element
                    ):
                        mapped = self._map_alias(trailing, atomname=atom_name)
                        if mapped and mapped != fixed_element:
                            self.warnings.append(
                                f"Line {i}: trailing token '{trailing}' differs from"
                                + f" fixed element '{fixed_element}'; mapped trailing"
                                + f"->'{mapped}'."
                            )
            else:
                # no fixed element, analyze trailing
                if trailing:
                    if self._is_valid_element_token(trailing):
                        # ok (element provided as token)
                        if strict and trailing not in self._VALID_ELEMENTS:
                            self.warnings.append(
                                f"Line {i}: trailing token '{trailing}' is not a canonical element."
                            )
                    else:
                        mapped = self._map_alias(trailing, atomname=atom_name)
                        if mapped:
                            # mapping exists
                            msg = (
                                f"Line {i}: trailing token '{trailing}' is non-canonical"
                                + f"; suggested element '{mapped}'."
                            )
                            if strict:
                                self.warnings.append(msg)
                            else:
                                # non-strict: append mild info to inspect
                                self.warnings.append(msg)
                        else:
                            self.warnings.append(
                                f"Line {i}: trailing token '{trailing}' cannot be mapped;"
                                + f" inspect atom='{atom_name}'."
                            )
                else:
                    if strict:
                        self.warnings.append(
                            f"Line {i}: no element (fixed-column or trailing) detected."
                        )

            # atom name odd characters
            if atom_name and not re.match(r"^[A-Za-z0-9_\-\.]+$", atom_name):
                self.warnings.append(
                    f"Line {i}: suspicious atom name '{atom_name}' (non-alphanumeric)."
                )

        return list(self.warnings)

    # -------------------------
    # Sanitization / Rebuild
    # -------------------------
    def sanitize(
        self, rebuild: bool = True, aggressive: bool = False
    ) -> "PDBQTSanitizer":
        """
        Produce sanitized content.

        If ``rebuild`` is True the sanitizer reconstructs ATOM/HETATM lines into fixed-column PDB
        format placing the element in cols 77-78. If ``rebuild`` is False, only whitespace-token
        trailing mapping is applied.

        :param rebuild: if True rebuild ATOM/HETATM into fixed-width form.
        :type rebuild: bool
        :param aggressive: if True apply aggressive heuristics (e.g. NA->Na).
        :type aggressive: bool
        :returns: self
        :rtype: PDBQTSanitizer
        :example:

        >>> s = PDBQTSanitizer("ligand.pdbqt")
        >>> s.validate()
        >>> s.sanitize(rebuild=True)
        <PDBQTSanitizer ...>
        """
        out_lines: List[str] = []
        self.warnings = []

        float_re = re.compile(r"^[+-]?\d+(\.\d+)?$")

        for i, ln in enumerate(self.lines, start=1):
            if not ln.strip():
                out_lines.append(ln)
                continue

            if not self._ATOM_RE.match(ln):
                out_lines.append(ln)
                continue

            toks = ln.split()
            # minimal tokens fallback
            if len(toks) < 6:
                out_lines.append(ln)
                self.warnings.append(f"Line {i}: short ATOM/HETATM left unchanged")
                continue

            record = toks[0]
            serial = toks[1] if len(toks) > 1 else "0"
            atom_name = toks[2] if len(toks) > 2 else ""
            resName = toks[3] if len(toks) > 3 else ""
            resSeq = toks[4] if len(toks) > 4 else "1"

            # find index of first float (x coordinate)
            float_idx = None
            for idx in range(5, len(toks)):
                if float_re.match(toks[idx]):
                    float_idx = idx
                    break
            if float_idx is None or float_idx + 2 >= len(toks):
                # can't parse coords reliably; keep line unchanged but warn
                out_lines.append(ln)
                self.warnings.append(
                    f"Line {i}: cannot parse coordinates, left unchanged"
                )
                continue

            x = float(toks[float_idx])
            y = float(toks[float_idx + 1])
            z = float(toks[float_idx + 2])
            occ = toks[float_idx + 3] if len(toks) > float_idx + 3 else "0.00"
            temp = toks[float_idx + 4] if len(toks) > float_idx + 4 else "0.00"
            # fmt: off
            trailing = toks[float_idx + 5:] if len(toks) > float_idx + 5 else []
            # fmt: on
            trailing_tok = trailing[-1] if trailing else ""

            # decide element: prefer fixed-column element if present and valid
            fixed_elem = ln[76:78].strip() if len(ln) >= 78 else ""
            element = ""
            if fixed_elem and self._is_valid_element_token(fixed_elem):
                element = fixed_elem
            else:
                # map trailing or atom name
                if trailing_tok:
                    element = self._map_alias(trailing_tok, atomname=atom_name)
                if not element:
                    element = self._map_alias(atom_name, atomname=atom_name)
                if not element:
                    element = "C"  # conservative fallback

            # aggressive heuristic
            if aggressive and element.upper() == "NA":
                element = "Na"

            if rebuild:
                # Build fixed-width PDB-style ATOM/HETATM line with element in cols 77-78.
                # We'll format base up to tempFactor, then pad to column 76 and set element.
                try:
                    serial_i = int(serial)
                except Exception:
                    serial_i = 0
                try:
                    resSeq_i = int(resSeq)
                except Exception:
                    resSeq_i = 1
                name_fmt = atom_name if len(atom_name) <= 4 else atom_name[:4]
                altLoc = " "
                chainID = " "
                iCode = " "
                base = (
                    f"{record:<6}{serial_i:>5} {name_fmt:^4}{altLoc}{resName:>3}"
                    + f" {chainID}{resSeq_i:>4}{iCode}   {x:8.3f}{y:8.3f}{z:8.3f}"
                    + f"{float(occ):6.2f}{float(temp):6.2f}"
                )
                if len(base) < 76:
                    base = base + " " * (76 - len(base))
                element_field = f"{element:>2}"
                rebuilt = base[:76] + element_field
                out_lines.append(rebuilt)
                if rebuilt.rstrip() != ln.rstrip():
                    self.warnings.append(
                        f"Line {i}: rebuilt ATOM/HETATM; element set to '{element}'"
                    )
            else:
                # minimal sanitize: replace trailing token if it's non-canonical
                if trailing_tok:
                    mapped = self._map_alias(trailing_tok, atomname=atom_name)
                    if mapped and mapped != trailing_tok:
                        # remove trailing token and append mapped element as last token
                        core = " ".join(toks[: float_idx + 5])  # upto temp
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
        rebuild: bool = True,
        aggressive: bool = False,
        backup: bool = True,
    ) -> Path:
        """
        Convenience: sanitize a file and write output.

        :param path: input path.
        :type path: str | pathlib.Path
        :param out_path: if None overwrite original (create .bak when backup=True).
        :type out_path: Optional[str | pathlib.Path]
        :param rebuild: if True rebuild ATOM/HETATM into fixed-width PDB form.
        :type rebuild: bool
        :param aggressive: if True apply aggressive heuristics.
        :type aggressive: bool
        :param backup: if True when overwriting create .bak copy.
        :type backup: bool
        :returns: Path to sanitized file.
        :rtype: pathlib.Path
        :raises FileNotFoundError: if input path does not exist.
        :example:

        >>> PDBQTSanitizer.sanitize_file("ligand.pdbqt", out_path="ligand.sanitized.pdbqt")
        PosixPath('ligand.sanitized.pdbqt')
        """
        p = Path(path)
        s = cls(p)
        s.validate(strict=False)
        s.sanitize(rebuild=rebuild, aggressive=aggressive)
        if out_path is None:
            if backup:
                bak = p.with_suffix(p.suffix + ".bak")
                if not bak.exists():
                    bak.write_text("\n".join(s.lines) + "\n", encoding="utf-8")
            s.write(p)
            return p
        else:
            outp = Path(out_path)
            s.write(outp)
            return outp

    def __repr__(self) -> str:
        return (
            f"<PDBQTSanitizer path={self._path.name if self._path else None}"
            + f" lines={len(self.lines)} sanitized={self._sanitized}>"
        )

    def help(self) -> str:
        """
        Short help text.

        :returns: help text string.
        :rtype: str
        """
        return (
            "PDBQTSanitizer(path).validate(strict=False) -> warnings; "
            "sanitize(rebuild=True) -> produce sanitized_lines; write(path) -> save file."
        )
