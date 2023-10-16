import torch
from src.inference_base import InferenceBase


class PyTorchInference(InferenceBase):
    def __init__(self, model_loader, device="cpu", debug_mode=False):
        self.device = device
        super().__init__(model_loader, debug_mode=debug_mode)
        self.model = self.load_model()

    def load_model(self):
        if self.device == "cuda":
            model = torch.load(self.model_path)
        else:
            model = self.model_loader.model
        model.to(self.device)
        model.eval()
        return model

    def predict(self, input_data, is_benchmark=False):
        super().predict(input_data, is_benchmark=is_benchmark)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_data.to(self.device))

        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()

        return self.get_top_predictions(prob, is_benchmark)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        super().benchmark(input_data, num_runs, warmup_runs)
