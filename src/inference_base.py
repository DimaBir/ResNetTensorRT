import os
import shutil
import time
import logging
from typing import List, Tuple, Union, Dict, Any
import openvino as ov
import torch
import onnxruntime as ort
import numpy as np


class InferenceBase:
    def __init__(self, model_path, image_processor, categories: List[str]):
        self.model_path = model_path
        self.image_processor = image_processor
        self.categories = categories
        self.local_model_dir = "models"
        self.local_model_path = os.path.join(
            self.local_model_dir, os.path.basename(self.model_path)
        )
        self.load_or_save_local_model()
        self.model = self.load_model()

    def load_or_save_local_model(self):
        if not os.path.exists(self.local_model_dir):
            os.makedirs(self.local_model_dir)
        if not os.path.exists(self.local_model_path):
            shutil.copy2(self.model_path, self.local_model_path)
        self.model_path = self.local_model_path

    def load_model(self):
        raise NotImplementedError

    def preprocess(self):
        input_data = self.image_processor.process_image()
        return input_data

    def predict(self, input_data, topk: int):
        raise NotImplementedError

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        # Warmup
        logging.info(f"Starting warmup for {self.__class__.__name__} inference...")
        for _ in range(warmup_runs):
            self.predict(input_data, topk=5)

        # Benchmark
        logging.info(f"Starting benchmark for {self.__class__.__name__} inference...")
        start_time = time.time()
        for _ in range(num_runs):
            self.predict(input_data, topk=5)
        avg_time = ((time.time() - start_time) / num_runs) * 1000  # To ms
        logging.info(f"Average inference time for {num_runs} runs: {avg_time:.4f} ms")
        return avg_time

    def get_top_predictions(self, prob: np.ndarray, topk: int):
        top_indices = prob.argsort()[-topk:][::-1]
        top_probs = prob[top_indices]
        for i in range(topk):
            probability = top_probs[i]
            class_label = self.categories[0][int(top_indices[i])]
            logging.info(f"#{i + 1}: {int(probability * 100)}% {class_label}")
        return prob
