from pathlib import Path
from typing import Union, Optional, List, Tuple, Iterable
import numpy as np
from vina import Vina


class VinaDock:
    """
    Object-oriented wrapper for AutoDock Vina (Python API).

    Provides CLI-style logging, robust parsing of returned scores,
    and helpers for single and batch docking.

    :param sf_name: Scoring function ('vina', 'vinardo', 'ad4')
    :param cpu: Number of CPU cores to use
    :param seed: Random seed for reproducibility (int or None)
    :param no_refine: If True, skip the final refinement step in Vina
    :param verbosity: Verbosity level passed to the underlying Vina wrapper
    """

    def __init__(
        self,
        sf_name: str = "vina",
        cpu: int = 1,
        seed: Optional[int] = None,
        no_refine: bool = False,
        verbosity: int = 1,
    ):
        if seed is not None:
            self._vina = Vina(
                sf_name=sf_name,
                cpu=int(cpu),
                seed=int(seed),
                no_refine=bool(no_refine),
                verbosity=int(verbosity),
            )
        else:
            # avoid passing seed=None into the C++ binding
            self._vina = Vina(
                sf_name=sf_name,
                cpu=int(cpu),
                no_refine=bool(no_refine),
                verbosity=int(verbosity),
            )

        self.receptor: Optional[str] = None
        self.ligand: Optional[str] = None
        self.center: Optional[Tuple[float, float, float]] = None
        self.size: Optional[Tuple[float, float, float]] = None
        self._scores: Optional[List[Tuple[float, float, float]]] = None
        self._last_poses = None

    # -------------------- helpers -------------------- #
    @staticmethod
    def _normalize_scores(raw_scores: Iterable) -> List[Tuple[float, float, float]]:
        """
        Convert vina.energies(...) return value into a list of (affinity, rmsd_lb, rmsd_ub).

        This is defensive: the Python binding may return a numpy array, list of arrays,
        1D array, or nested lists. This helper attempts to coerce all common shapes.

        :param raw_scores: whatever was returned by self._vina.energies(...)
        :return: list of (affinity, rmsd_lb, rmsd_ub) as floats
        """
        arr = np.asarray(list(raw_scores))
        # case: (n, 3)  -> straightforward
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return [tuple(map(float, arr[i, :3])) for i in range(arr.shape[0])]

        # case: 1D array flattened but length is multiple of 3
        if arr.ndim == 1 and arr.size % 3 == 0 and arr.size > 0:
            n = arr.size // 3
            reshaped = arr.reshape((n, 3))
            return [tuple(map(float, reshaped[i])) for i in range(n)]

        # fallback: iterate original raw_scores and coerce each item
        out: List[Tuple[float, float, float]] = []
        for item in raw_scores:
            it = np.asarray(item).flatten()
            if it.size >= 3:
                out.append((float(it[0]), float(it[1]), float(it[2])))
            elif it.size == 2:
                out.append((float(it[0]), float(it[1]), 0.0))
            elif it.size == 1:
                out.append((float(it[0]), 0.0, 0.0))
            else:
                # Very defensive: if item is empty or unexpected, raise informative error
                raise ValueError(f"Unable to parse docking score item: {item!r}")
        return out

    # -------------------- setup -------------------- #
    def set_receptor(self, receptor_path: Union[str, Path]) -> "VinaDock":
        """
        Load receptor (PDBQT).

        :param receptor_path: path to receptor in PDBQT format
        :return: self
        """
        self._vina.set_receptor(str(receptor_path))
        self.receptor = str(receptor_path)
        return self

    def set_ligand(self, ligand_path: Union[str, Path]) -> "VinaDock":
        """
        Load ligand from PDBQT file.

        :param ligand_path: path to ligand in PDBQT format
        :return: self
        """
        self._vina.set_ligand_from_file(str(ligand_path))
        self.ligand = str(ligand_path)
        return self

    def set_ligand_from_string(self, pdbqt_str: str) -> "VinaDock":
        """
        Load ligand from PDBQT string.

        :param pdbqt_str: ligand PDBQT as string
        :return: self
        """
        self._vina.set_ligand_from_string(pdbqt_str)
        self.ligand = "<pdbqt-string>"
        return self

    def set_ligand_rdkit(self, rdkit_mol) -> "VinaDock":
        """
        Load ligand directly from RDKit Mol (if RDKit bindings available).

        :param rdkit_mol: rdkit.Chem.Mol instance
        :return: self
        """
        self._vina.set_ligand_from_rdkit(rdkit_mol)
        self.ligand = "<rdkit-mol>"
        return self

    def define_box(
        self, center: Tuple[float, float, float], size: Tuple[float, float, float]
    ) -> "VinaDock":
        """
        Compute the Vina maps for the search box and store parameters.

        :param center: (x, y, z)
        :param size: (size_x, size_y, size_z)
        :return: self
        """
        self._vina.compute_vina_maps(center=center, box_size=size)
        self.center = tuple(map(float, center))
        self.size = tuple(map(float, size))
        return self

    # -------------------- run -------------------- #
    def dock(self, exhaustiveness: int = 8, n_poses: int = 9) -> "VinaDock":
        """
        Run docking (wraps Vina.dock).

        :param exhaustiveness: search exhaustiveness
        :param n_poses: number of poses to request
        :return: self
        """
        # call docking
        self._vina.dock(exhaustiveness=int(exhaustiveness), n_poses=int(n_poses))
        # normalize and store scores
        raw = self._vina.energies(n_poses=int(n_poses))
        self._scores = self._normalize_scores(raw)
        # also store raw poses if needed later
        try:
            self._last_poses = self._vina.poses(n_poses=int(n_poses))
        except Exception:
            self._last_poses = None
        return self

    def score(self) -> float:
        """
        Score the current ligand pose.

        :return: score (float)
        """
        s = self._vina.score()
        # score() may return array-like; coerce to float
        if isinstance(s, (list, tuple, np.ndarray)):
            return float(np.asarray(s).flat[0])
        return float(s)

    def optimize(self) -> float:
        """
        Local optimization of the current ligand pose.

        :return: optimized score (float)
        """
        s = self._vina.optimize()
        if isinstance(s, (list, tuple, np.ndarray)):
            return float(np.asarray(s).flat[0])
        return float(s)

    # -------------------- outputs -------------------- #
    def write_poses(
        self, out_path: Union[str, Path], n_poses: int = 9, overwrite: bool = True
    ) -> "VinaDock":
        """
        Write docked poses to file.

        :param out_path: output path (PDBQT)
        :param n_poses: number of poses to write
        :param overwrite: overwrite if exists
        :return: self
        """
        self._vina.write_poses(
            str(out_path), n_poses=int(n_poses), overwrite=bool(overwrite)
        )
        return self

    def write_log(
        self, log_path: Union[str, Path], seed: Optional[int] = None
    ) -> "VinaDock":
        """
        Write a CLI-style log (and print to console).

        :param log_path: path to the log file
        :param seed: optional seed to show in header (defaults to None)
        :return: self
        """
        if self._scores is None:
            raise RuntimeError("No docking results to log. Call .dock() first.")

        with open(log_path, "w") as f:

            def log_print(msg: str):
                print(msg, flush=True)
                f.write(msg + "\n")

            log_print("Computing Vina grid ... done.")
            seed_txt = seed if seed is not None else "auto"
            log_print(f"Performing docking (random seed: {seed_txt}) ... ")
            log_print("0%   10   20   30   40   50   60   70   80   90   100%")
            log_print("|----|----|----|----|----|----|----|----|----|----|")
            log_print("*" * 51 + "\n")

            log_print("mode |   affinity | dist from best mode")
            log_print("     | (kcal/mol) | rmsd l.b.| rmsd u.b.")
            log_print("-----+------------+----------+----------")

            for i, (e, rmsd_lb, rmsd_ub) in enumerate(self._scores, start=1):
                log_print(f"{i:4d}{e:12.3f}{rmsd_lb:10.3f}{rmsd_ub:10.3f}")

        return self

    # -------------------- batch helper -------------------- #
    def dock_many(
        self,
        ligand_paths: Iterable[Union[str, Path]],
        out_dir: Union[str, Path],
        log_dir: Union[str, Path],
        exhaustiveness: int = 8,
        n_poses: int = 9,
        overwrite: bool = True,
    ) -> "VinaDock":
        """
        Dock multiple ligands with the currently defined receptor & maps.

        Reuses the same Vina instance and precomputed maps for speed.

        :param ligand_paths: iterable of ligand file paths
        :param out_dir: directory to write poses (one file per ligand)
        :param log_dir: directory to write logs (one log per ligand)
        :param exhaustiveness: search exhaustiveness for each ligand
        :param n_poses: number of poses per ligand
        :param overwrite: overwrite existing files if True
        :return: self
        """
        out_dir = Path(out_dir)
        log_dir = Path(log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        if self.receptor is None or self.center is None or self.size is None:
            raise RuntimeError(
                "Receptor or box not defined. Call .set_receptor(...) and .define_box(...) first."
            )

        results = []
        for lig in ligand_paths:
            lig = Path(lig)
            out_path = out_dir / f"{lig.stem}_docked.pdbqt"
            log_path = log_dir / f"{lig.stem}.log"

            # set ligand and run docking
            self.set_ligand(str(lig))
            self.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            self.write_poses(out_path, n_poses=n_poses, overwrite=overwrite)
            self.write_log(log_path)
            results.append(
                {
                    "ligand": str(lig),
                    "out": str(out_path),
                    "log": str(log_path),
                    "scores": self.scores,
                }
            )

        # after batch, last _scores/_last_poses correspond to final ligand
        return self

    # -------------------- properties -------------------- #
    @property
    def scores(self) -> Optional[List[Tuple[float, float, float]]]:
        """
        Return docking scores as list of tuples: (affinity, rmsd_lb, rmsd_ub).

        :return: list of scores or None if docking not run
        """
        return None if self._scores is None else list(self._scores)

    def get_best(self) -> Optional[Tuple[float, float, float]]:
        """
        Return best mode (first pose) score tuple or None.

        :return: (affinity, rmsd_lb, rmsd_ub) or None
        """
        return None if not self._scores else self._scores[0]

    def __repr__(self) -> str:
        return (
            f"<VinaDock receptor={self.receptor} ligand={self.ligand} "
            f"center={self.center} size={self.size} "
            f"scores={'yes' if self._scores is not None else 'no'}>"
        )

    # small help function
    def help(self) -> None:
        """
        Print short usage help.
        """
        print(
            "VinaDock usage example:\n"
            "  vd = VinaDock(cpu=4, seed=42)\n"
            "  vd.set_receptor('rec.pdbqt').define_box((x,y,z),(sx,sy,sz))\n"
            "  vd.set_ligand('lig.pdbqt').dock().write_poses('out.pdbqt').write_log('log.txt')"
        )
