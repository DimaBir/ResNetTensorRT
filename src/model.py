import pandas as pd
from torchvision import models


class ModelLoader:
    def __init__(self, device: str = "cuda") -> None:
        """
        Initialize the ModelLoader object.

        :param device: The device to load the model on ("cpu" or "cuda").
        """
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(
            device
        )
        self.device = device
        self.categories: pd.DataFrame = pd.read_csv(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            header=None,
        )
