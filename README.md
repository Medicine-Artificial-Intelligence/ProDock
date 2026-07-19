**ProDock — Semi-automated Rank-Resolved Multi-Engine Docking Pipeline for Virtual Screening**
![ProDock graphic](fig/Graphic.png)
**ProDock** is a semi-automated Python package for protein preparation, ligand preparation, docking execution, and rank-resolved post-docking analysis.

The workflow combines global docking with **DiffDock** and local docking with **GNINA**, preserving multiple poses per ligand instead of reducing each docking run to a single top-ranked conformer. Each pose rank is evaluated independently using engine-specific scores and additional descriptors.

For **GNINA**, these descriptors include:

- empirical affinity;
- `CNNpose`;
- `CNNaffinity`;
- Reference-based protein–ligand interaction-fingerprint similarity: `Similarity-type1` and `Similarity-type2`
- Steric clash counts; and
- Optional electrostatic solvation energy.

For **DiffDock**, the evaluated descriptors include:
- Confidence score;
- Unoccupied region, representing the absence of an interaction surface between the protein and ligand; and
- Percentage of ligand atoms remaining within the binding site and percentage of Occupancy.

**ProDock** builds on established cheminformatics, visualization, and molecular-simulation libraries, including [RDKit](https://www.rdkit.org/), [PyMOL](https://www.pymol.org/), [Open Babel](https://openbabel.org/), [OpenMM](https://openmm.org/), [MDAnalysis](https://www.mdanalysis.org/), [ProLIF](https://prolif.readthedocs.io/), [Biopython](https://biopython.org/), and [APBS](https://www.poissonboltzmann.org/).

The package writes structured intermediate files and target-specific output directories so that protein preparation, ligand preparation, docking, re-ranking, and downstream inspection can be repeated from explicit inputs.

**Overall workflow**
![ProDock flow](fig/Flow.png)
Protein Preparation
![ProDock protein_prep](fig/Protein-prep.png)
Ligand Preparation
![ProDock ligand_prep](fig/Ligand-prep.png)

## License
This project is licensed under MIT License - see the [License](LICENSE) file for details.

## Authors & Contributors
- [Lai Hoang Son Le](https://github.com/lelaihoangson)
- [Thanh-An Pham](https://github.com/Thanh-An-Pham)
- [Tieu-Long Phan](https://tieulongphan.github.io/)

## Acknowledgments
This work has received support from the Korea International Cooperation Agency (KOICA) under the project entitled ``Education and Research Capacity Building Project at University of Medicine and Pharmacy at Ho Chi Minh City,'' conducted from 2024 to 2025 (Project No. 2021-00020-3). TLP and PFS have received funding from the European Union's Horizon Europe Doctoral Network programme under the Marie Sk{\l}odowska-Curie grant agreement No.~101072930 (TACsy - Training Alliance for Computational systems chemistry).
