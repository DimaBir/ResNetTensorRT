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
    def __init__(self, model_loader, precision=torch.float32):
        """
        Initialize the TensorRTInference object.

        :param model_loader: Object responsible for loading the model and categories.
        :param precision: Precision mode for TensorRT (default is torch.float32).
        """
        self.precision = precision
        super().__init__(model_loader)
        if CUDA_AVAILABLE:
            self.load_model()

    def load_model(self):
        """
        Load and convert the PyTorch model to TensorRT format.
        """
        # Load the PyTorch model
        self.model = self.model_loader.model.to(self.device).eval()

        # Convert the PyTorch model to TensorRT
        self.model = trt.ts.compile(
            self.model, inputs=[trt.Input((1, 3, 224, 224), dtype=self.precision)]
        )

    def predict(self, img_batch, topk: int):
        """
        Run prediction on the input data using the TensorRT model.

        :param img_batch: Data to run the prediction on.
        :param topk: Number of top predictions to return.
        :return: Top predictions based on the probabilities.
        """
        logging.info(
            f"Running prediction for TensorRT (CUDA) model with {self.precision} precision"
        )
        with torch.no_grad():
            outputs = self.model(img_batch.to(self.device))

        # Compute the softmax probabilities
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()

        return self.log_top_predictions(prob, topk)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        """
        Benchmark the prediction performance using the TensorRT model.

        :param input_data: Data to run the benchmark on.
        :param num_runs: Number of runs for the benchmark.
        :param warmup_runs: Number of warmup runs before the benchmark.
        :return: Average inference time in milliseconds.
        """
        return super().benchmark(input_data, num_runs, warmup_runs)
