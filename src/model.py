from typing import Union

import pandas as pd
import torch
from torchvision import models

IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

MODEL_REGISTRY = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.IMAGENET1K_V1),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
}


class ModelLoader:
    def __init__(self, model_type: str = "resnet50", device: Union[str, torch.device] = "cuda") -> None:
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = self._load_model(model_type)
        self.categories: pd.DataFrame = self._load_categories()

    def _load_model(self, model_type: str) -> torch.nn.Module:
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
        
        model_fn, weights = MODEL_REGISTRY[model_type]
        return model_fn(weights=weights).to(self.device)

    @staticmethod
    def _load_categories() -> pd.DataFrame:
        return pd.read_csv(IMAGENET_CLASSES_URL, header=None)
