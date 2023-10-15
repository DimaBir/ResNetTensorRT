import logging
import torch
from src.inference_base import InferenceBase


class PyTorchCPUInference(InferenceBase):
    def __init__(self, model_loader):
        super().__init__(model_loader)

    def load_model(self):
        return self.model_loader.model.to(self.model_loader.device)

    def predict(self, input_data, is_benchmark=False):
        logging.info(f"Running prediction for PyTorch CPU model")
        self.model_loader.model.to(self.model_loader.device)
        self.model_loader.model.eval()
        with torch.no_grad():
            outputs = self.model_loader.model(input_data.to(self.model_loader.device))
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()
        return self.get_top_predictions(prob, is_benchmark)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        return super().benchmark(input_data, num_runs, warmup_runs)


class PyTorchCUDAInference(InferenceBase):
    def __init__(self, model_loader):
        super().__init__(model_loader)

    def load_model(self):
        model = torch.load(self.model_path)
        model.to("cuda")
        model.eval()
        return model

    def predict(self, input_data, topk: int):
        logging.info(f"Running prediction for PyTorch CUDA model")
        input_data = input_data.to("cuda")
        with torch.no_grad():
            outputs = self.model(input_data)
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()
        return self.get_top_predictions(prob, topk)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        super().benchmark(input_data, num_runs, warmup_runs)
