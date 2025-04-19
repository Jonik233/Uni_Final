import os
import shutil
import pandas as pd
from typing import List, Dict
from dotenv import dotenv_values
from env_enum import SourceEnvKeys, TargetEnvKeys


class Pipeline:

    ENV_PATH = ".env"

    def __init__(self):
        self.env_config = dotenv_values(self.ENV_PATH)

    def __extract(self) -> List[str]:
        return [
            os.path.join(self.env_config[env_key.value], file_name)
            for env_key in SourceEnvKeys
            for file_name in os.listdir(self.env_config[env_key.value])
        ]

    def __transform(self, file_paths: List[str]) -> Dict:
        prefixes = ("plane", "military")
        file_names = [os.path.basename(file_path) for file_path in file_paths]
        labels = [
            1 if any(prefix in file_name for prefix in prefixes) else 0
            for file_name in file_names
        ]
        return {"FileNames": file_names, "Labels": labels}

    def __load(self, file_paths: List[str], labels_dict: Dict) -> None:
        target_labels_path = self.env_config[TargetEnvKeys.LABEL_TARGET_DIR.value]
        target_data_path = self.env_config[TargetEnvKeys.DATA_TARGET_DIR.value]

        pd.DataFrame(labels_dict).to_csv(target_labels_path, index=False)

        for file_path in file_paths:
            target_path = os.path.join(target_data_path, os.path.basename(file_path))
            shutil.copy2(file_path, target_path)
            print(f"Copied: {file_path} -> {target_path}")

    def run(self) -> None:
        file_paths = self.__extract()
        labels_dict = self.__transform(file_paths)
        self.__load(file_paths, labels_dict)


if __name__ == "__main__":
    etl_pipe = Pipeline()
    etl_pipe.run()
