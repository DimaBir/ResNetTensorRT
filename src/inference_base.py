import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch

from common.utils import DEFAULT_TOPK

DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_RUNS = 100
DEFAULT_WARMUP_RUNS = 50
MS_PER_SECOND = 1000


class InferenceBase:
    def __init__(
        self,
        model_loader,
        onnx_path: Optional[str] = None,
        ov_path: Optional[str] = None,
        topk: int = DEFAULT_TOPK,
        debug_mode: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.model_loader = model_loader
        self.onnx_path = onnx_path
        self.ov_path = ov_path
        self.categories = model_loader.categories
        self.model = self.load_model()
        self.topk = topk
        self.debug_mode = debug_mode
        self.batch_size = batch_size

    def load_model(self):
        raise NotImplementedError

    def predict(self, input_data, is_benchmark: bool = False):
        if not is_benchmark:
            logging.info(f"Running prediction for {self.__class__.__name__} model")
            if self.debug_mode:
                print(f"Running prediction for {self.__class__.__name__} model")

    def _prepare_batch(self, input_data: torch.Tensor) -> torch.Tensor:
        if len(input_data.shape) == 4:
            input_data = input_data.squeeze(0)
        return torch.stack([input_data] * self.batch_size)

    def _warmup(self, input_batch: torch.Tensor, warmup_runs: int):
        logging.info(f"Starting warmup for {self.__class__.__name__} inference...")
        for _ in range(warmup_runs):
            for img in input_batch:
                self.predict(img.unsqueeze(0), is_benchmark=True)

    def _run_benchmark(self, input_batch: torch.Tensor, num_runs: int) -> Tuple[float, int]:
        logging.info(f"Starting benchmark for {self.__class__.__name__} inference...")
        start_time = time.time()
        for _ in range(num_runs):
            for img in input_batch:
                self.predict(img.unsqueeze(0), is_benchmark=True)
        
        elapsed_time = time.time() - start_time
        total_samples = self.batch_size * num_runs
        return elapsed_time, total_samples

    def _calculate_metrics(self, elapsed_time: float, total_samples: int) -> Tuple[float, float]:
        avg_time = (elapsed_time / total_samples) * MS_PER_SECOND
        throughput = total_samples / elapsed_time
        
        logging.info(f"Average inference time: {avg_time:.4f} ms")
        logging.info(f"Throughput: {throughput:.2f} samples/sec")
        
        if self.debug_mode:
            print(f"Average inference time for {self.__class__.__name__}: {avg_time:.4f} ms")
            print(f"Throughput for {self.__class__.__name__}: {throughput:.2f} samples/sec")
        
        return avg_time, throughput

    def benchmark(
        self, 
        input_data: torch.Tensor, 
        num_runs: int = DEFAULT_NUM_RUNS, 
        warmup_runs: int = DEFAULT_WARMUP_RUNS
    ) -> Tuple[float, float]:
        input_batch = self._prepare_batch(input_data)
        self._warmup(input_batch, warmup_runs)
        elapsed_time, total_samples = self._run_benchmark(input_batch, num_runs)
        return self._calculate_metrics(elapsed_time, total_samples)

    def get_top_predictions(self, prob: np.ndarray, is_benchmark: bool = False) -> Optional[np.ndarray]:
        if is_benchmark:
            return None

        top_indices = prob.argsort()[-self.topk:][::-1]
        top_probs = prob[top_indices]

        for i in range(self.topk):
            probability = top_probs[i]
            class_label = self.categories[0][int(top_indices[i])]
            logging.info(f"#{i + 1}: {int(probability * 100)}% {class_label}")
            if self.debug_mode:
                print(f"#{i + 1}: {int(probability * 100)}% {class_label}")
        
        return prob
