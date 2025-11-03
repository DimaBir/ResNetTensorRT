import numpy as np
import torch

from src.inference_base import InferenceBase


class PyTorchInference(InferenceBase):
    def __init__(self, model_loader, device: str | torch.device = "cpu", debug_mode: bool = False):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        super().__init__(model_loader, debug_mode=debug_mode)
        self.model = self.load_model()

    def load_model(self) -> torch.nn.Module:
        return self.model_loader.model.to(self.device)

    def predict(self, input_data: torch.Tensor, is_benchmark: bool = False) -> np.ndarray | None:
        super().predict(input_data, is_benchmark=is_benchmark)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_data.to(self.device))

        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()

        return self.get_top_predictions(prob, is_benchmark)
