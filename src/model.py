import os
import pandas as pd
import torch
from torchvision import models


class ModelLoader:
    def __init__(self, device: str = "cuda") -> None:
        """
        Initialize the ModelLoader object.

        :param device: The device to load the model on ("cpu" or "cuda").
        """
        self.device = device
        self.model_path = "models/resnet50.pth"
        self.categories_url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        self.categories_path = "models/imagenet_classes.txt"

        # Check if model exists locally, if not, download and save
        if not os.path.exists(self.model_path):
            os.makedirs("models", exist_ok=True)
        '''if not os.path.exists(self.model_path):
            os.makedirs("models", exist_ok=True)
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            ).to(device)
            torch.save(self.model.state_dict(), self.model_path)
        else:
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            ).to(device)
            self.model.load_state_dict(torch.load(self.model_path))
        '''

        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        ).to(device)

        # Check if categories exist locally, if not, download and save
        if not os.path.exists(self.categories_path):
            self.categories = pd.read_csv(self.categories_url, header=None)
            self.categories.to_csv(self.categories_path, index=False)
        else:
            self.categories = pd.read_csv(self.categories_path, header=None)

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
