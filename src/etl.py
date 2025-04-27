import os
import shutil
import pandas as pd
from typing import List, Dict
from dotenv import dotenv_values
from env_enum import ResnetSourceEnvKeys, TargetEnvKeys


class Pipeline:
    """
    A simple ETL pipeline that extracts files from source directories,
    transforms their names into labels, and saves both files and labels
    to target locations defined in a .env file.
    """

    def __init__(self, env_path: str = ".env"):
        self.env_config = dotenv_values(env_path)

    def extract(self) -> List[str]:
        """
        Extracts file paths from source directories defined in .env file
        :return: list of file paths
        """
        return [
            os.path.join(self.env_config[env_key.value], file_name)
            for env_key in ResnetSourceEnvKeys
            for file_name in os.listdir(self.env_config[env_key.value])
        ]

    def transform(self, file_paths: List[str]) -> Dict:
        """
        Transforms file prefixes into labels and stores file names and corresponding labels into a dictionary
        :param file_paths: extracted file paths from source directories defined in .env file
        :return: dictionary of file names and corresponding labels
        """
        prefixes = ("plane", "military")
        file_names = [os.path.basename(file_path) for file_path in file_paths]
        labels = [
            1 if any(prefix in file_name for prefix in prefixes) else 0
            for file_name in file_names
        ]
        return {"FileNames": file_names, "Labels": labels}

    def load(self, file_paths: List[str], labels_dict: Dict) -> None:
        """
        Copies files to the target directory and saves labels to CSV.
        :param file_paths: list of file paths from source directories defined in .env file
        :param labels_dict: dictionary of file names and corresponding labels {"FileNames": list(), "Labels": list()}
        :return: None
        """
        target_data_path = self.env_config[TargetEnvKeys.RESNET34_DATA_TARGET_DIR.value]
        target_labels_path = self.env_config[TargetEnvKeys.RESNET34_LABEL_TARGET_DIR.value]

        os.makedirs(target_data_path, exist_ok=True)
        os.makedirs(os.path.dirname(target_labels_path), exist_ok=True)

        pd.DataFrame(labels_dict).to_csv(target_labels_path, index=False)

        for file_path in file_paths:
            target_path = os.path.join(target_data_path, os.path.basename(file_path))
            try:
                shutil.copy2(file_path, target_path)
                print(f"Copied: {file_path} -> {target_path}")
            except (IOError, OSError) as e:
                print(f"Failed to copy {file_path} to {target_path}: {e}")

    def run(self) -> None:
        """
        Runs the full pipeline: extract -> transform -> load
        :return: None
        """
        file_paths = self.extract()
        labels_dict = self.transform(file_paths)
        self.load(file_paths, labels_dict)


if __name__ == "__main__":
    etl_pipe = Pipeline()
    etl_pipe.run()
