import logging
import os.path
import torch

CUDA_AVAILABLE = False
if torch.cuda.is_available():
    try:
        import torch_tensorrt
        CUDA_AVAILABLE = True
    except ImportError:
        print("torch-tensorrt is not installed. Running on CPU mode only.")

from benchmark.benchmark_models import benchmark_onnx_model, benchmark_ov_model
from benchmark.benchmark_utils import run_all_benchmarks, plot_benchmark_results
from common.utils import (
    parse_arguments,
    init_onnx_model,
    init_ov_model,
    init_cuda_model,
    export_onnx_model,
)
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
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loader = ModelLoader(device=device)
    img_processor = ImageProcessor(img_path=args.image_path, device=device)
    img_batch = img_processor.process_image()

    # ONNX
    if args.mode in ["onnx", "all"]:
        ort_session = init_onnx_model(args.onnx_path, model_loader, device)
        if args.mode != "all":
            benchmark_onnx_model(ort_session)
            predict_onnx_model(
                ort_session, img_batch, args.topk, model_loader.categories
            )

    # OpenVINO
    if args.mode in ["ov", "all"]:
        # Check if ONNX model wasn't exported previously
        if not os.path.isfile(args.onnx_path):
            export_onnx_model(
                onnx_path=args.onnx_path, model_loader=model_loader, device=device
            )

        ov_model = init_ov_model(args.onnx_path)
        if args.mode != "all":
            ov_benchmark = benchmark_ov_model(ov_model)
            predict_ov_model(
                ov_benchmark.compiled_model,
                img_batch,
                args.topk,
                model_loader.categories,
            )

    # CUDA
    if args.mode in ["cuda", "all"]:
        # CUDA configurations
        cuda_configs = [
            {"device": "cpu", "precision": torch.float32, "is_trt": False},
            {"device": "cuda", "precision": torch.float32, "is_trt": False},
            {"device": "cuda", "precision": torch.float32, "is_trt": True},
            {"device": "cuda", "precision": torch.float16, "is_trt": True},
        ]

        for config in cuda_configs:
            device = config["device"]
            precision = config["precision"]
            is_trt = config["is_trt"]

            # check if CUDA is available
            if device.lower() == "cuda" and not CUDA_AVAILABLE:
                continue

            model = init_cuda_model(model_loader, device, precision)

            # If the configuration is not for TensorRT, store the model under a PyTorch key
            if not is_trt:
                models[f"PyTorch_{device}"] = model
                model = model.to(device)
                img_batch = img_batch.to(device)
            else:
                print("Compiling TensorRT model")
                batch_size = 1 if args.mode == "cuda" else 32
                model = torch_tensorrt.compile(
                    model,
                    inputs=[torch_tensorrt.Input((batch_size, 3, 224, 224), dtype=precision)],
                    enabled_precisions={precision},
                    truncate_long_and_double=True,
                    require_full_compilation=True,
                )
                # If it is for TensorRT, determine the mode (FP32 or FP16) and store under a TensorRT key
                mode = "fp32" if precision == torch.float32 else "fp16"
                models[f"trt_{mode}"] = model

            if args.mode != "all":
                predict_cuda_model(
                    model, img_batch, args.topk, model_loader.categories, precision
                )

    # Aggregate Benchmark (if mode is "all")
    if args.mode == "all":
        models["onnx"] = ort_session
        models["ov"] = ov_model

        results = run_all_benchmarks(models, img_batch)
        plot_benchmark_results(results)


if __name__ == "__main__":
    main()
