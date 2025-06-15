import torch
import torchvision
import torch.nn as nn
import numpy as np

import argparse
from PIL import Image
from typing import List
import matplotlib.pyplot as plt

from ultralytics import YOLO
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet
from preprocessing import ResnetPreprocess, VisPreprocess

from dotenv import dotenv_values
from env_enum import ModelEnvKeys

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def init_resnet34(weights_path: str, num_classes: int = 2) -> ResNet:
    """
    Initializes resnet34 model
    :param weights_path: path to the weights of the trained resnet34
    :param num_classes: number of classes to specify in a new head
    :return: ResNet34 model
    """

    # Model initialization
    model = resnet34(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Loading fine-tuned weights into the model
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)

    # Switching model to evaluation mode
    model.eval()
    return model


def init_yolo(weights_path: str) -> YOLO:
    """
    Initialized YOLOV8 model
    :param weights_path: path to the weights of the trained YOLOV8
    :return: YOLOV8 model
    """

    # Loading model with fine-tuned weights
    model = YOLO(weights_path)
    return model


# noinspection PyShadowingNames
@torch.no_grad()
def resnet_infer(model: ResNet, img_path: str, device: str = "cpu") -> int:
    """
    Performs inference with ResNet34 model
    :param model: initialized ResNet34 model
    :param img_path: path to an image to use inference on
    :param device: device on which inference will be run on
    :return: binary label
    """
    model = model.to(device=device)
    preprocess_fn = ResnetPreprocess()
    img = preprocess_fn(img_path, device)
    output = model(img)
    pred_label = torch.argmax(output, dim=1).item()
    return pred_label


# noinspection PyShadowingNames
@torch.no_grad()
def yolo_infer(model: YOLO, img_path: str, device: str = "cpu") -> List:
    """
    Performs inference with YOLOV8 model
    :param model: YOLOV8 model
    :param img_path: path to an image to use inference on
    :param device: device on which inference will be run on
    :return: labeled image result
    """
    model = model.to(device=device)
    output = model(img_path)
    return output


# noinspection PyShadowingNames
def infer(
    resnet_model: ResNet, yolo_model: YOLO, paths_batch: List[str], device: str = "cpu"
) -> List[torch.Tensor]:
    """
    Applies two-step inference: ResNet34 -> YOLOV8
    :param resnet_model: ResNet34 model
    :param yolo_model: YOLOV8 model
    :param paths_batch: list with paths to images to perform inference on
    :param device: device on which inference will be run on
    :return: list of labeled image results
    """
    labeled_imgs = list()
    preprocess_fn = VisPreprocess()

    for i, img_path in enumerate(paths_batch):
        is_plane = resnet_infer(resnet_model, img_path, device)
        if is_plane:
            result = yolo_infer(yolo_model, img_path, device).pop()
            result.save(f"pred_{i}.png")
            img = Image.open(f"pred_{i}.png")
            preprocessed_img = preprocess_fn(img)
            labeled_imgs.append(preprocessed_img)

    return labeled_imgs


if __name__ == "__main__":

    # Initializing argument parser
    parser = argparse.ArgumentParser(description="Inference function")
    parser.add_argument(
        "img_paths", type=str, nargs="+", help="One or more image file paths"
    )

    # Fetching img_paths from console
    args = parser.parse_args()
    img_paths = args.img_paths

    # Initializing config for environment variables
    env_config = dotenv_values("../.env")

    # Fetching path for resnet weights
    resnet_weights_env_key = ModelEnvKeys.RESNET34_WEIGHTS_PATH.value
    resnet_weights_path = env_config[resnet_weights_env_key]

    # Fetching path for yolo weights
    yolo_weights_env_key = ModelEnvKeys.YOLO_WEIGHTS_PATH.value
    yolo_weights_path = env_config[yolo_weights_env_key]

    # Initializing resnet and yolo models
    resnet_model = init_resnet34(resnet_weights_path)
    yolo_model = init_yolo(yolo_weights_path)

    # Setting a device to perform inference on
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Performing inferencing
    imgs = infer(resnet_model, yolo_model, img_paths)

    # Visualization of results
    imgs_grid = torchvision.utils.make_grid(imgs)
    plt.figure(figsize=(18, 18))
    plt.imshow(imgs_grid.permute(1, 2, 0))
    plt.show()
