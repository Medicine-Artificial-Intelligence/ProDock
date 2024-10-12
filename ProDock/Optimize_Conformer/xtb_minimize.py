import os
import glob
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class XTBMinimize:
    @staticmethod
    def retrive_filename(sdf_path_folder: str):
        """
        Extract folder path and file names from the sdf path.
        Parameters:
        - sdf_path_folder (str): Please input whole path to sdf folder, not the relative path.

        """
        folder_path = [i for i in glob.glob(f"{sdf_path_folder}/*")]
        file_name_list = [f"{i}/{i[len(sdf_path_folder)+1:]}.sdf" for i in folder_path]

        return sorted(folder_path), sorted(file_name_list)

    @staticmethod
    def xtb_optimize(filename: str):
        """Performs energy minimization using .sdf file in each folder.

        Parameters:
        - filename_list (list): The path to the XYZ file to be optimized.

        """

        try:
            subprocess.run(["xtb", filename, "--opt", "--silent", "tight"], check=True)

        except subprocess.CalledProcessError as e:
            logging.error(f"An error occurred with xtb: {e}")
            raise
        except FileNotFoundError:
            logging.error(
                "xtb executable not found. Ensure that xtb is correctly installed and available on the system path."
            )
            raise
        except Exception as e:
            logging.error(f"An error occurred during file handling: {e}")
            raise

    def fit(self, file_path: str):
        """
        Executes the full optimization workflow.

        Parameters:
        - file_path (str): The full path, not the relative path to the sdf folder

        Raises:
        - Exception: If any step in the workflow fails.
        """

        folder_path, file_name_list = XTBMinimize.retrive_filename(file_path)
        # Optimize with xtb
        temp_path = os.getcwd()
        for i, path in enumerate(file_name_list):
            filename = os.path.join(temp_path, path)
            os.chdir(folder_path[i])
            XTBMinimize.xtb_optimize(filename)
        os.chdir(temp_path)
