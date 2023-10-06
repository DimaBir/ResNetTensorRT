import time
from typing import Tuple

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import logging
import onnxruntime as ort
import openvino as ov
import openvino.inference_engine as ie
from openvino import utils

# Configure logging
logging.basicConfig(filename="model.log", level=logging.INFO)


class Benchmark(ABC):
    """
    Abstract class representing a benchmark.
    """

    def __init__(self, nruns: int = 100, nwarmup: int = 50):
        self.nruns = nruns
        self.nwarmup = nwarmup

    @abstractmethod
    def run(self) -> None:
        """
        Abstract method to run the benchmark.
        """
        pass


class PyTorchBenchmark:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        input_shape: Tuple[int, int, int, int] = (32, 3, 224, 224),
        dtype: torch.dtype = torch.float32,
        nwarmup: int = 50,
        nruns: int = 100,
    ) -> None:
        """
        Initialize the Benchmark object.

        :param model: The model to be benchmarked.
        :param device: The device to run the benchmark on ("cpu" or "cuda").
        :param input_shape: The shape of the input data.
        :param dtype: The data type to be used in the benchmark (typically torch.float32 or torch.float16).
        :param nwarmup: The number of warmup runs before timing.
        :param nruns: The number of runs for timing.
        """
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self.dtype = dtype
        self.nwarmup = nwarmup
        self.nruns = nruns

        cudnn.benchmark = True  # Enable cuDNN benchmarking optimization

    def run(self) -> None:
        """
        Run the benchmark with the given model, input shape, and other parameters.
        Log the average batch time and print the input shape and output feature size.
        """
        # Prepare input data
        input_data = torch.randn(self.input_shape).to(self.device).to(self.dtype)

        # Warm up
        print("Warm up ...")
        with torch.no_grad():
            for _ in range(self.nwarmup):
                features = self.model(input_data)
        torch.cuda.synchronize()

        # Start timing
        print("Start timing ...")
        timings = []
        with torch.no_grad():
            for i in range(1, self.nruns + 1):
                start_time = time.time()
                features = self.model(input_data)
                torch.cuda.synchronize()
                end_time = time.time()
                timings.append(end_time - start_time)

                if i % 10 == 0:
                    print(
                        f"Iteration {i}/{self.nruns}, ave batch time {np.mean(timings) * 1000:.2f} ms"
                    )

        # Print and log results
        print(f"Input shape: {input_data.size()}")
        print(f"Output features size: {features.size()}")
        logging.info(f"Average batch time: {np.mean(timings) * 1000:.2f} ms")


class ONNXBenchmark(Benchmark):
    """
    A class used to benchmark the performance of an ONNX model.
    """

    def __init__(
        self,
        ort_session: ort.InferenceSession,
        input_shape: tuple,
        nruns: int = 100,
        nwarmup: int = 50,
    ):
        super().__init__(nruns)
        self.ort_session = ort_session
        self.input_shape = input_shape
        self.nwarmup = nwarmup
        self.nruns = nruns

    def run(self) -> None:
        print("Warming up ...")
        # Adjusting the batch size in the input shape to match the expected input size of the model.
        input_shape = (1,) + self.input_shape[1:]
        input_data = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(self.nwarmup):  # Warm-up runs
            _ = self.ort_session.run(None, {"input": input_data})

        print("Starting benchmark ...")
        timings = []

        for _ in range(self.nruns):
            start_time = time.time()
            _ = self.ort_session.run(None, {"input": input_data})
            end_time = time.time()
            timings.append(end_time - start_time)

        avg_time = np.mean(timings) * 1000
        logging.info(f"Average ONNX inference time: {avg_time:.2f} ms")


class OVBenchmark(Benchmark):
    def __init__(self, model: ov.frontend.FrontEnd, input_shape: Tuple[int, int, int, int]):
        """
        Initialize the OVBenchmark with the OpenVINO model and the input shape.

        :param model: ov.frontend.FrontEnd
            The OpenVINO model.
        :param input_shape: Tuple[int, int, int, int]
            The shape of the model input.
        """
        self.ov_model = model
        self.core = ov.Core()
        self.compiled_model = None
        self.input_shape = input_shape
        self.warmup_runs = 50
        self.num_runs = 100
        self.dummy_input = np.random.randn(*input_shape).astype(np.float32)

    def warmup(self):
        """
        Compile the OpenVINO model for optimal execution on available hardware.
        """
        self.compiled_model = self.core.compile_model(self.ov_model, "AUTO")

    def inference(self, input_data) -> dict:
        """
        Perform inference on the input data using the compiled OpenVINO model.

        :param input_data: np.ndarray
            The input data for the model.
        :return: dict
            The model's output as a dictionary.
        """
        outputs = self.compiled_model.infer(inputs={"input": input_data})
        return outputs

    def run(self):
        """
        Run the benchmark on the OpenVINO model. It first warms up by compiling the model and then measures
        the average inference time over a set number of runs.
        """
        # Warm-up runs
        logging.info("Warming up ...")
        for _ in range(self.warmup_runs):
            self.warmup()

        # Benchmarking
        total_time = 0
        for _ in range(self.num_runs):
            start_time = time.time()
            _ = self.inference(self.dummy_input)
            total_time += time.time() - start_time

        avg_time = total_time / self.num_runs
        logging.info(f"Average inference time: {avg_time * 1000:.2f} ms")
