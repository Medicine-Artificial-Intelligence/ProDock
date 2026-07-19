**ProDock — Semi-automated Rank-Resolved Multi-Engine Docking Pipeline for Virtual Screening**
**ProDock** is a semi-automated Python package for protein preparation, ligand preparation, docking execution, and rank-resolved post-docking analysis.

The workflow combines global docking with **DiffDock** and local docking with **GNINA**, preserving multiple poses per ligand instead of reducing each docking run to a single top-ranked conformer. Each pose rank is evaluated independently using engine-specific scores and additional descriptors.

For **GNINA**, these descriptors include:

- empirical affinity;
- `CNNpose`;
- `CNNaffinity`;
- reference-based protein–ligand interaction-fingerprint similarity;
- steric clash counts; and
- optional electrostatic solvation energy.

For **DiffDock**, the evaluated descriptors include:

- confidence score;
- a measurement of the unoccupied region, representing the absence of an interaction surface between the protein and ligand; and
- a ligand-localization metric represented by the percentage of ligand atoms remaining within the binding site.

**ProDock** builds on established cheminformatics, visualization, and molecular-simulation libraries, including [RDKit](https://www.rdkit.org/), [PyMOL](https://www.pymol.org/), [Open Babel](https://openbabel.org/), [OpenMM](https://openmm.org/), [MDAnalysis](https://www.mdanalysis.org/), [ProLIF](https://prolif.readthedocs.io/), [Biopython](https://biopython.org/), and [APBS](https://www.poissonboltzmann.org/).

The package writes structured intermediate files and target-specific output directories so that protein preparation, ligand preparation, docking, re-ranking, and downstream inspection can be repeated from explicit inputs.

The package writes structured intermediate files and target-specific output directories so that protein preparation, ligand preparation, docking, re-ranking, and downstream inspection can be repeated from explicit inputs.
![ProDock graphic](fig/Graphic.png)
Overall workflow
![ProDock flow](fig/Flow.png)
Protein Preparation
![ProDock protein_prep](fig/Protein-prep.png)
Ligand Preparation
![ProDock ligand_prep](fig/Ligand-prep.png)
