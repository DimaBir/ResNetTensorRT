import torch
import torch.nn.functional as F
import logging
import numpy as np
from src.inference_base import InferenceBase


class PyTorchInference(InferenceBase):
    def __init__(self, model_loader, device="cpu", debug_mode=False):
        """
        Initialize the PyTorchInference object.

        :param model_loader: Object responsible for loading the model and categories.
        :param device: The device to load the model on ("cpu" or "cuda").
        :param debug_mode: If True, print additional debug information.
        """
        self.device = device
        super().__init__(model_loader, debug_mode=debug_mode)
        self.model = self.load_model()

    def load_model(self):
        """
        Load the PyTorch model to the specified device.

        :return: Loaded PyTorch model.
        """
        return self.model_loader.model.to(self.device)

    def predict(self, input_data, is_benchmark=False):
        """
        Run prediction on the input data using the PyTorch model.

        :param input_data: Data to run the prediction on.
        :param is_benchmark: If True, the prediction is part of a benchmark run.
        :return: Top predictions based on the probabilities.
        """
        super().predict(input_data, is_benchmark=is_benchmark)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_data.to(self.device))

        # Compute the softmax probabilities
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()

        return self.get_top_predictions(prob, is_benchmark)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        """
        Benchmark the prediction performance using the PyTorch model.

        :param input_data: Data to run the benchmark on.
        :param num_runs: Number of runs for the benchmark.
        :param warmup_runs: Number of warmup runs before the benchmark.
        :return: Average inference time in milliseconds.
        """
        return super().benchmark(input_data, num_runs, warmup_runs)

    def get_top_predictions(self, output, is_benchmark=False):
        """
        Get the top predictions based on the model's output.

        :param output: Raw output (logits) from the model.
        :param is_benchmark: If True, the method is called during a benchmark run.
        :return: List of dictionaries with label and confidence.
        """
        if is_benchmark:
            return None

        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(output, dim=1)

        # Get the top indices and probabilities
        top_probs, top_indices = torch.topk(probabilities, self.topk)
        top_probs = top_probs[0].tolist()  # Convert to list
        top_indices = top_indices[0].tolist()

        # Prepare the list of predictions
        predictions = []
        for i in range(self.topk):
            probability = top_probs[i]
            class_label = self.categories[0][top_indices[i]]
            predictions.append({"label": class_label, "confidence": float(probability)})

            # Log the top predictions
            logging.info(f"#{i + 1}: {probability * 100:.2f}% {class_label}")

        return predictions
