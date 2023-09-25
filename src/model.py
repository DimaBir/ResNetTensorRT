import torch
from torchvision import models
import pandas as pd

class ModelLoader:
    def __init__(self, device="cuda"):
        self.model = models.resnet50(pretrained=True).to(device)
        self.device = device
        self.categories = pd.read_csv('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt', header=None)

    def predict(self, img_batch):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_batch)
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        return prob
