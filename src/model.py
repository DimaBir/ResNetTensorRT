import pandas as pd
import torch
from torchvision import models


class ModelLoader:
    def __init__(self, device: str = "cuda") -> None:
        """
        Initialize the ModelLoader object.

        :param device: The device to load the model on ("cpu" or "cuda").
        """
        self.model = models.resnet50(pretrained=True).to(device)
        self.device = device
        self.categories: pd.DataFrame = pd.read_csv(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            header=None,
        )

    def predict(self, img_batch: torch.Tensor) -> torch.Tensor:
        """
        Make a prediction on the provided image batch.

        :param img_batch: A batch of images to make predictions on.
        :return: A tensor representing the probabilities of the predictions.
        """
        # Set the model to evaluation mode and make a prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_batch)

        # Compute the softmax probabilities
        prob = torch.nn.functional.softmax(outputs[0], dim=0)

        return prob
