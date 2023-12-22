import logging
import torch
import warnings
from src.onnx_inference import ONNXInference
from src.ov_inference import OVInference
from src.pytorch_inference import PyTorchInference
from src.tensorrt_inference import TensorRTInference
from common.utils import parse_arguments, plot_benchmark_results, check_cuda_availability, run_inference_for_mode
from src.image_processor import ImageProcessor
from src.model import ModelLoader

# Configure logging
logging.basicConfig(filename="inference.log", level=logging.INFO)

# Filter out the specific warning from torchvision
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")


def main():
    args = parse_arguments()
    cuda_available = check_cuda_availability()
    device = torch.device("cuda" if cuda_available else "cpu")

    # Initialize the model loader and image processor
    model_loader = ModelLoader(device=device)
    img_processor = ImageProcessor(img_path=args.image_path, device=device)

    # Process the input image into a batch for inference
    img_batch = img_processor.process_image()
    benchmark_results = {}

    # Enable debug logging if specified in the arguments
    if args.DEBUG:
        logging.debug("Debug mode is enabled")

    # Define the available inference modes and their corresponding classes and paths
    inference_modes = {
        "onnx": (ONNXInference, args.onnx_path),
        "ov": (OVInference, args.ov_path),
        "cpu": (PyTorchInference, None, "cpu"),
        "cuda": (PyTorchInference, None) if cuda_available else None,
        "tensorrt": (TensorRTInference, None) if cuda_available else None
    }

    # Iterate through the specified inference modes and run inference and benchmarking
    for mode, (inference_class, path, *extra_args) in inference_modes.items():
        if mode in args.mode or args.mode == "all":
            run_inference_for_mode(
                inference_class(model_loader, path, *extra_args, debug_mode=args.DEBUG),
                model_loader,
                img_batch,
                device,
                args.DEBUG,
                benchmark_results
            )

    # Only for 'all' modes, plot a graph to compare the benchmark results
    if args.mode == "all":
        plot_benchmark_results(benchmark_results)


if __name__ == "__main__":
    main()
