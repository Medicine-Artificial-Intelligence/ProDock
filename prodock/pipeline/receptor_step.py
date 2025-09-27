# prodock/pipeline/receptor_step.py
from __future__ import annotations

from pathlib import Path
from typing import Type, Union

import logging

# Project import must be at top (fixes E402)
from prodock.preprocess.receptor.receptor import ReceptorPrep as DefaultReceptorPrep

logger = logging.getLogger("prodock.pipeline.receptor_step")


class ReceptorStep:
    """
    Thin wrapper around a configurable ``ReceptorPrep`` implementation.

    :example:

    >>> from tempfile import TemporaryDirectory
    >>> class _MiniReceptorPrep:
    ...     def __init__(self, enable_logging=True):  # signature compatibility
    ...         self.expected_output_path = None
    ...     def prep(self, input_pdb, output_dir, **kwargs):
    ...         out = Path(output_dir) / "prepared.pdbqt"
    ...         out.write_text("PDBQT")
    ...         self.expected_output_path = str(out)
    >>> tmp = TemporaryDirectory()
    >>> step = ReceptorStep(receptor_prep_cls=_MiniReceptorPrep)
    >>> inp = Path(tmp.name, "rec.pdb"); inp.write_text("ATOM")
    >>> out = step.prepare(inp, tmp.name)
    >>> out.name == "prepared.pdbqt"
    True
    """

    def __init__(self, receptor_prep_cls: Type = DefaultReceptorPrep) -> None:
        """
        :param receptor_prep_cls: Class implementing ``prep(...)`` and setting
                                  ``expected_output_path`` upon success.
        """
        self._cls = receptor_prep_cls

    def prepare(
        self,
        input_pdb: Union[str, Path],
        output_dir: Union[str, Path],
        energy_diff: float = 10.0,
        max_minimization_steps: int = 5000,
        minimize_in_water: bool = False,
        out_fmt: str = "pdbqt",
        enable_logging: bool = True,
    ) -> Path:
        """
        Prepare receptor.

        :param input_pdb: Input receptor (PDB).
        :param output_dir: Destination directory.
        :param energy_diff: Energy filtering threshold (kcal/mol).
        :param max_minimization_steps: Maximum minimization steps.
        :param minimize_in_water: Whether to minimize in implicit water.
        :param out_fmt: Output format (default ``"pdbqt"``).
        :param enable_logging: Toggle for verbose prep logging.
        :returns: Path to the prepared receptor.
        :raises FileNotFoundError: When ``input_pdb`` does not exist.
        :raises RuntimeError: When the prep class does not produce output.
        """
        input_pdb = Path(input_pdb)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_pdb.exists():
            raise FileNotFoundError(f"Input receptor not found: {input_pdb}")

        rp = self._cls(enable_logging=enable_logging)
        rp.prep(
            input_pdb=str(input_pdb),
            output_dir=str(output_dir),
            energy_diff=energy_diff,
            max_minimization_steps=max_minimization_steps,
            minimize_in_water=minimize_in_water,
            enable_logging=enable_logging,
            out_fmt=out_fmt,
        )
        if not getattr(rp, "expected_output_path", None):
            raise RuntimeError("ReceptorPrep did not produce expected_output_path.")
        logger.info("Receptor prepared at %s", rp.expected_output_path)
        return Path(rp.expected_output_path).resolve()
