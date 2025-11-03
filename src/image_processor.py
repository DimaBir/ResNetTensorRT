import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 256
CROP_SIZE = 224


class ImageProcessor:
    def __init__(self, img_path: str, device: str | torch.device = "cuda") -> None:
        self.img_path = img_path
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.transform = self._create_transform()

    @staticmethod
    def _create_transform() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def process_image(self) -> torch.Tensor:
        img = Image.open(self.img_path)
        img_transformed = self.transform(img)
        return torch.unsqueeze(img_transformed, 0).to(self.device)
