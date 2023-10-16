import torch
import logging
from src.inference_base import InferenceBase

# Check for CUDA and TensorRT availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    try:
        import torch_tensorrt as trt
    except ImportError:
        logging.warning("torch-tensorrt is not installed. Running on CPU mode only.")
        CUDA_AVAILABLE = False


class TensorRTInference(InferenceBase):
    def __init__(self, model_loader, device, precision=torch.float32, debug_mode=False):
        """
        Initialize the TensorRTInference object.

        :param model_loader: Object responsible for loading the model and categories.
        :param precision: Precision mode for TensorRT (default is torch.float32).
        """
        self.precision = precision
        self.device = device
        super().__init__(model_loader, debug_mode=debug_mode)
        if CUDA_AVAILABLE:
            self.load_model()

    def load_model(self):
        """
        Load and convert the PyTorch model to TensorRT format.
        """
        # Load the PyTorch model
        self.model = self.model_loader.model.to(self.device).eval()

        # Convert the model to the desired precision
        if self.precision == torch.float16:
            self.model = self.model.half()
        elif self.precision == torch.float32:
            self.model = self.model.float()

        # Convert the input tensor for tracing to the desired precision
        tracing_input = torch.randn((1, 3, 224, 224)).to(self.device).to(self.precision)

        self.model = torch.jit.trace(self.model, [tracing_input])

        # Convert the PyTorch model to TensorRT
        self.model = trt.ts.compile(
            self.model, inputs=[trt.Input((1, 3, 224, 224), dtype=self.precision)]
        )

    def predict(self, input_data, is_benchmark=False):
        """
        Run prediction on the input data using the TensorRT model.

        :param input_data: Data to run the prediction on.
        :param is_benchmark: If True, the prediction is part of a benchmark run.
        :return: Top predictions based on the probabilities.
        """
        super().predict(input_data, is_benchmark=is_benchmark)

        with torch.no_grad():
            outputs = self.model(input_data.to(self.device).to(dtype=self.precision))

        # Compute the softmax probabilities
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()

        return self.get_top_predictions(prob, is_benchmark)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        """
        Benchmark the prediction performance using the TensorRT model.

        :param input_data: Data to run the benchmark on.
        :param num_runs: Number of runs for the benchmark.
        :param warmup_runs: Number of warmup runs before the benchmark.
        :return: Average inference time in milliseconds.
        """
        return super().benchmark(input_data, num_runs, warmup_runs)
