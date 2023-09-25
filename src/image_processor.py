from torchvision import transforms
from PIL import Image
import torch

class ImageProcessor:
    def __init__(self, img_path, device="cuda"):
        self.img_path = img_path
        self.device = device

    def process_image(self):
        img = Image.open(self.img_path)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img_transformed = transform(img)
        img_batch = torch.unsqueeze(img_transformed, 0).to(self.device)
        return img_batch
