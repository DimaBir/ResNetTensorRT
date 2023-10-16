import os
import shutil
import time
import logging
import numpy as np


class InferenceBase:
    def __init__(self, model_loader, onnx_path=None, ov_path=None, topk=5):
        self.model_loader = model_loader
        self.model = self.model_loader.model
        # self.model_path = model_loader.model_path

        self.onnx_path = onnx_path
        self.ov_path = ov_path

        self.categories = model_loader.categories
        self.model = self.load_model()
        self.topk = topk

    def load_model(self):
        raise NotImplementedError

    def predict(self, input_data, is_benchmark=False):
        raise NotImplementedError

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        # Warmup
        logging.info(f"Starting warmup for {self.__class__.__name__} inference...")
        for _ in range(warmup_runs):
            self.predict(input_data, is_benchmark=True)

        # Benchmark
        logging.info(f"Starting benchmark for {self.__class__.__name__} inference...")
        start_time = time.time()
        for _ in range(num_runs):
            self.predict(input_data, is_benchmark=True)
        avg_time = ((time.time() - start_time) / num_runs) * 1000  # To ms
        logging.info(f"Average inference time for {num_runs} runs: {avg_time:.4f} ms")
        print(
            f"Average inference time for {self.__class__.__name__} and {num_runs} runs: {avg_time:.4f} ms"
        )
        return avg_time

    def get_top_predictions(self, prob: np.ndarray, is_benchmark=False):
        if is_benchmark:
            return None

        top_indices = prob.argsort()[-self.topk :][::-1]
        top_probs = prob[top_indices]
        for i in range(self.topk):
            probability = top_probs[i]
            class_label = self.categories[0][int(top_indices[i])]
            logging.info(f"#{i + 1}: {int(probability * 100)}% {class_label}")
        return prob
