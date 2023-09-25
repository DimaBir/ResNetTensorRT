import time
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import logging

# Configure logging
logging.basicConfig(filename="model.log", level=logging.INFO)


class Benchmark:
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
