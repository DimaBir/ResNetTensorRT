import time
import logging
import numpy as np
import torch
from multiprocessing import Pool


class InferenceBase:
    def __init__(
        self,
        model_loader,
        onnx_path=None,
        ov_path=None,
        topk=5,
        debug_mode=False,
        batch_size=8,
    ):
        """
        Base class for inference.

        :param model_loader: Object responsible for loading the model and categories.
        :param onnx_path: Path to the ONNX model (if applicable).
        :param ov_path: Path to the OpenVINO model (if applicable).
        :param topk: Number of top predictions to return.
        :param debug_mode: If True, print additional debug information.
        :param batch_size: How many input images to stack for benchmark
        """
        self.model_loader = model_loader
        self.onnx_path = onnx_path
        self.ov_path = ov_path
        self.categories = model_loader.categories
        self.model = self.load_model()
        self.topk = topk
        self.debug_mode = debug_mode
        self.batch_size = batch_size

    def load_model(self):
        """
        Load the model. This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def predict(self, input_data, is_benchmark=False):
        """
        Run prediction on the input data.

        :param input_data: Data to run the prediction on.
        :param is_benchmark: If True, the prediction is part of a benchmark run.
        """
        if not is_benchmark:
            logging.info(f"Running prediction for {self.__class__.__name__} model")
            if self.debug_mode:
                print(f"Running prediction for {self.__class__.__name__} model")

    def _benchmark_single_process(self, args):
        """
        Single process benchmarking function.
        """
        input_batch, num_runs = args
        start_time = time.time()
        for _ in range(num_runs):
            self.predict(input_batch, is_benchmark=True)
        return time.time() - start_time

    def benchmark(self, input_data, num_runs=100, warmup_runs=50, num_processes=4):
        """
        Parallelized benchmark method.
        """
        # Prepare batch data once
        input_batch = torch.stack([input_data] * self.batch_size)

        # Warmup (not parallelized to avoid CUDA context issues)
        for _ in range(warmup_runs):
            self.predict(input_batch, is_benchmark=True)

        # Parallel Benchmark
        with Pool(processes=num_processes) as pool:
            process_args = [(input_batch, num_runs // num_processes) for _ in range(num_processes)]
            times = pool.map(self._benchmark_single_process, process_args)

        total_time = sum(times)
        avg_time = (total_time / num_runs) * 1000  # Convert to ms

        # Log only final results
        if self.debug_mode:
            logging.info(f"Average inference time for {self.__class__.__name__}: {avg_time:.4f} ms")

        # Calculate throughput
        total_samples = self.batch_size * num_runs
        throughput = total_samples / total_time

        if self.debug_mode:
            logging.info(f"Throughput for {self.__class__.__name__}: {throughput:.2f} samples/sec")

        return avg_time, throughput

    def get_top_predictions(self, logits: np.ndarray, is_benchmark=False):
        raise NotImplementedError
