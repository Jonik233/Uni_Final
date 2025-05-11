import os
import pytest
from etl import Pipeline
from dotenv import dotenv_values
from env_enum import ResnetSourceEnvKeys

def test_transform_labels():
    file_paths = [
        '../data/resnet_data/classified/Empty/empty_1.png',
        '../data/resnet_data/classified/Civilian/plane_4.png',
        '../data/resnet_data/classified/Military/military_40.png'
    ]
    pipeline = Pipeline()
    labels = pipeline.transform(file_paths)
    assert labels['FileNames'] == ['empty_1.png', 'plane_4.png', 'military_40.png']
    assert labels['Labels'] == [0, 1, 1]

def test_extract():
    pipeline = Pipeline()
    paths = pipeline.extract()

    # Expect all data files for resnet train
    env_config = dotenv_values("../.env")
    military_planes_dir = env_config[ResnetSourceEnvKeys.MILITARY_PLANES_DIR.value]
    civilian_planes_dir = env_config[ResnetSourceEnvKeys.CIVILIAN_PLANES_DIR.value]
    empty_planes_dir = env_config[ResnetSourceEnvKeys.EMPTY_PLANE_DIR.value]
    assert len(paths) == len(os.listdir(military_planes_dir)) + len(os.listdir(civilian_planes_dir)) + len(os.listdir(empty_planes_dir))


if __name__ == '__main__':
    pytest.main()