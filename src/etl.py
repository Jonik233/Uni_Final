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
        file_paths = list()
        for env_key in SourceEnvKeys:
            data_path = self.env_config[env_key.value]
            for file_name in os.listdir(data_path):
                file_path = os.path.join(data_path, file_name)
                file_paths.append(file_path)
        return file_paths

    def __transform(self, file_paths: List[str]) -> Dict:
        labels_dict = {"Paths": list(), "Labels": list()}
        for file_path in file_paths:
            file_name = file_path.split("\\")[-1]
            prefix = file_name.split("_")[0]
            label = 1 if prefix in ["plane", "military"] else 0
            labels_dict["Paths"].append(file_name)
            labels_dict["Labels"].append(label)
        return labels_dict

    def __load(self, file_paths: List[str], labels_dict: Dict) -> None:
        label_key = TargetEnvKeys.LABEL_TARGET_DIR.value
        target_labels_path = self.env_config[label_key]

        data_key = TargetEnvKeys.DATA_TARGET_DIR.value
        target_data_path = self.env_config[data_key]

        df_labels = pd.DataFrame(labels_dict)
        df_labels.to_csv(target_labels_path, index=False)

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_data_path, filename)
            shutil.copy2(file_path, target_path)
            print(f"Copied: {file_path} -> {target_path}")

    def run(self) -> None:
        file_paths = self.__extract()
        labels_dict = self.__transform(file_paths)
        self.__load(file_paths, labels_dict)


if __name__ == "__main__":
    etl_pipe = Pipeline()
    etl_pipe.run()
