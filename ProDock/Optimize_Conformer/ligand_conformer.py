import os
import shutil
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count


# from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy,RDKitMolCreate
class Ligand_Conformer:
    @staticmethod
    def list_smi(text: str) -> list[str]:
        """
        Read a text file containing SMILES strings and return a list of RDKit Mol objects.

        Parameters
        ----------
        text : str
            The path to the text file containing the SMILES strings.

        Returns
        -------
        list[Chem.Mol]
            A list of RDKit Mol objects. The list will contain all the molecules
            that can be successfully created from the SMILES strings.

        Notes
        -----
        The function reads the text file line by line, strips each line of any
        trailing newline characters, and then attempts to create an RDKit Mol object
        from the line. If the creation is successful, the Mol object is added to the list.
        Finally, the function prints the number of SMILES strings read from the text file
        and the number of Mol objects created, and then returns the list.
        """

        with open(text, "r") as f:
            smi = [smiles.strip("\r\n") for smiles in f]
            mol_list = [
                Chem.MolFromSmiles(smiles)
                for smiles in smi
                if Chem.MolFromSmiles(smiles)
            ]
        logging.info("Input Smi", len(smi))
        logging.info("Sucess embbeding 2D Mol", len(mol_list))
        return mol_list

    @staticmethod
    def mol_embbeding_3d(mol: Chem.Mol) -> Chem.Mol:
        """
        Embed a molecule in 3D space using the RDKit's ETKDG method.

        Parameters
        ----------
        mol : Chem.Mol
            The RDKit molecule object to be embedded in 3D space.

        Returns
        -------
        Chem.Mol
            The RDKit molecule object with 3D coordinates embedded.

        Notes
        -----
        The function first adds hydrogens to the molecule. Then it attempts to
        embed the molecule in 3D space using the ETKDG method with the ET version 2.
        If the embedding fails, the function will attempt to embed the molecule again
        without the ET version 2. Finally, the function returns the embedded molecule.

        """
        mol = Chem.AddHs(mol)

        embed_success = AllChem.EmbedMolecule(mol, ETversion=2, randomSeed=42)

        if embed_success == -1:
            logging.warning("Embedding failed")
            pass
        else:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        return mol

    @staticmethod
    def _embed_molecule_parallel(mol: Chem.Mol) -> Chem.Mol:
        """Helper function for parallel processing."""
        return Ligand_Conformer.mol_embbeding_3d(mol)

    def write_sdf(
        self, smi_filename: str, sdf_output_folder: str, num_workers: int = None
    ) -> None:
        """
        Embeds all molecules from a SMILES file in 3D space, and writes each
        molecule to a separate SDF file in a specified folder.

        Parameters
        ----------
        smi_filename : str
            The path to the SMILES file.
        sdf_output_folder : str
            The path to the folder where the SDF files will be saved.

        Returns
        -------
        None

        Notes
        -----
        The function first reads all SMILES strings from the SMILES file and embeds them
        in 3D space. Then it creates a new folder with the same name as the SMILES file,
        and writes each embedded molecule to a separate SDF file in the folder.
        The SDF files are named as "ligand_0.sdf", "ligand_1.sdf", etc.
        """
        mol_list = Ligand_Conformer.list_smi(smi_filename)
        # Determine number of workers
        if num_workers is None:
            num_workers = cpu_count()

        logging.info("Using %d workers for parallel processing", num_workers)

        # Parallel embedding
        with Pool(processes=num_workers) as pool:
            embedded_mols = pool.map(self._embed_molecule_parallel, mol_list)

        refine_mol_list = [mol for mol in embedded_mols if mol is not None]
        logging.info("Successfully embedded 3D Mol: %d", len(refine_mol_list))

        if not os.path.exists(sdf_output_folder):
            os.makedirs(sdf_output_folder)

        for i, mol in enumerate(refine_mol_list):
            sdf_file = os.path.join(sdf_output_folder, f"ligand_{i}.sdf")

            writer = Chem.SDWriter(sdf_file)
            writer.write(mol)
            writer.close()

            ligand_folder = os.path.join(sdf_output_folder, f"ligand_{i}")
            if not os.path.exists(ligand_folder):
                os.makedirs(ligand_folder)
            shutil.move(sdf_file, os.path.join(ligand_folder, f"ligand_{i}.sdf"))
