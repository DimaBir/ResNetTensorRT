import logging
import torch

from src.onnx_inference import ONNXInference
from src.ov_inference import OVInference
from src.pytorch_inference import PyTorchCPUInference, PyTorchCUDAInference

from src.tensorrt_inference import TensorRTInference

CUDA_AVAILABLE = False
if torch.cuda.is_available():
    try:
        import torch_tensorrt

        CUDA_AVAILABLE = True
    except ImportError:
        print("torch-tensorrt is not installed. Running on CPU mode only.")

from benchmark.benchmark_utils import plot_benchmark_results
from common.utils import parse_arguments
from src.image_processor import ImageProcessor
from prediction.prediction_models import *
from src.model import ModelLoader
import warnings

# Filter out the specific warning from torchvision
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

# Configure logging
logging.basicConfig(filename="model.log", level=logging.INFO)


def main():
    """
    Main function to run inference, benchmarks, and predictions on the model
    using provided image and optional parameters.
    """
    args = parse_arguments()

    # Model and Image Initialization
    benchmark_results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loader = ModelLoader(device=device)
    img_processor = ImageProcessor(img_path=args.image_path, device=device)
    img_batch = img_processor.process_image()

    # ONNX
    if args.mode in ["onnx", "all"]:
        onnx_inference = ONNXInference(model_loader, args.onnx_path, img_processor)

        benchmark_results["ONNX (CPU)"] = onnx_inference.benchmark()
        onnx_inference.predict(img_batch)

    # OpenVINO
    if args.mode in ["ov", "all"]:
        ov_inference = OVInference(model_loader, args.ov_path, img_processor)

        benchmark_results["OpenVINO (CPU)"] = ov_inference.benchmark()
        ov_inference.predict(img_batch)

    # PyTorch CPU
    if args.mode in ["pytorch_cpu", "all"]:
        pytorch_cpu_inference = PyTorchCPUInference(model_loader, img_processor)

        benchmark_results["PyTorch (CPU)"] = pytorch_cpu_inference.benchmark()
        pytorch_cpu_inference.predict(img_batch)

    # PyTorch CUDA
    if args.mode in ["pytorch_cuda", "all"] and device == "cuda":
        pytorch_cuda_inference = PyTorchCUDAInference(model_loader, img_processor)

        benchmark_results["PyTorch (CUDA)"] = pytorch_cuda_inference.benchmark()
        pytorch_cuda_inference.predict(img_batch)

    # TensorRT
    if args.mode in ["tensorrt", "all"] and device == "cuda":
        precisions = [torch.float16, torch.float32]
        for precision in precisions:
            tensorrt_inference = TensorRTInference(
                model_loader, img_processor, precision=precision
            )

            benchmark_results[f"TRT_{precision}"] = tensorrt_inference.benchmark()
            tensorrt_inference.predict(img_batch)

    # Plot graph combining all results
    if args.mode == "all":
        plot_benchmark_results(benchmark_results)


if __name__ == "__main__":
    main()
