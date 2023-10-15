import logging
import torch
from inference_base import InferenceBase


class PyTorchCPUInference(InferenceBase):
    def load_model(self):
        model = torch.load(self.model_path, map_location="cpu")
        model.eval()
        return model

    def predict(self, input_data, topk: int):
        logging.info(f"Running prediction for PyTorch CPU model")
        with torch.no_grad():
            outputs = self.model(input_data)
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()
        return self.log_top_predictions(prob, topk)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        super().benchmark(input_data, num_runs, warmup_runs)


class PyTorchCUDAInference(InferenceBase):
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
        return self.log_top_predictions(prob, topk)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        super().benchmark(input_data, num_runs, warmup_runs)
