import logging
import torch

from src.onnx_inference import ONNXInference
from src.ov_inference import OVInference
from src.pytorch_inference import PyTorchInference

from src.tensorrt_inference import TensorRTInference

CUDA_AVAILABLE = False
if torch.cuda.is_available():
    try:
        import torch_tensorrt

        CUDA_AVAILABLE = True
    except ImportError:
        print("torch-tensorrt is not installed. Running on CPU mode only.")

from common.utils import parse_arguments, plot_benchmark_results
from src.image_processor import ImageProcessor
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
        onnx_inference = ONNXInference(model_loader, args.onnx_path)

        benchmark_results["ONNX (CPU)"] = onnx_inference.benchmark(img_batch)
        onnx_inference.predict(img_batch)

    # OpenVINO
    if args.mode in ["ov", "all"]:
        ov_inference = OVInference(model_loader, args.ov_path)

        benchmark_results["OpenVINO (CPU)"] = ov_inference.benchmark(img_batch)
        ov_inference.predict(img_batch)

    # PyTorch CPU
    if args.mode in ["cpu", "all"]:
        pytorch_cpu_inference = PyTorchInference(model_loader, device="cpu")

        benchmark_results["PyTorch (CPU)"] = pytorch_cpu_inference.benchmark(img_batch)
        pytorch_cpu_inference.predict(img_batch)

    # PyTorch CUDA
    if args.mode in ["cuda", "all"] and device == "cuda":
        pytorch_cuda_inference = PyTorchInference(model_loader, device=device)

        benchmark_results["PyTorch (CUDA)"] = pytorch_cuda_inference.benchmark(
            img_batch
        )
        pytorch_cuda_inference.predict(img_batch)

    # TensorRT
    if args.mode in ["tensorrt", "all"] and device == "cuda":
        precisions = [torch.float16, torch.float32]
        for precision in precisions:
            tensorrt_inference = TensorRTInference(model_loader, precision=precision)

            benchmark_results[f"TRT_{precision}"] = tensorrt_inference.benchmark(
                img_batch
            )
            tensorrt_inference.predict(img_batch)

    # Plot graph combining all results
    if args.mode == "all":
        plot_benchmark_results(benchmark_results)


if __name__ == "__main__":
    main()
