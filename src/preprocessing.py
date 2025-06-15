import torch
from PIL import Image, ImageFile
from torchvision.io import read_image
from torchvision import transforms as T


class ConvertToRGB:
    """Converts input image from BGR to RGB"""

    def __call__(self, img: Image) -> Image:
        return img.convert("RGB")


class AddBatchDim:
    """Adds batch dimension to the input tensor"""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(0)


class VisPreprocess:
    """
    Applies preprocessing for ImageFile before visualization
    """

    def __call__(self, img: ImageFile):
        preprocess_fn = T.Compose([T.Resize(424), T.CenterCrop(400), T.ToTensor()])
        return preprocess_fn(img)


class ResnetPreprocess:
    """
    Applies preprocessing for ResNet34 model inputs
    """

    def __call__(self, img_path: str, device: str = "cpu"):
        # Reading img using given file path
        img = read_image(img_path)

        # Defining img transformations
        preprocess = T.Compose(
            [
                T.ToPILImage(),
                ConvertToRGB(),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                AddBatchDim(),
            ]
        )

        # Preprocessing img
        preprocessed_img = preprocess(img)

        # Loading preprocessed img on given device
        final_img = preprocessed_img.to(device=device)

        return final_img
