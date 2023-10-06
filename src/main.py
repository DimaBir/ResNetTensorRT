import logging

from benchmark.benchmark_models import benchmark_onnx_model, benchmark_ov_model
from benchmark.benchmark_utils import run_all_benchmarks, plot_benchmark_results
from common.utils import (
    parse_arguments,
    init_onnx_model,
    init_ov_model,
    init_cuda_model,
)
from image_processor import ImageProcessor
from prediction.prediction_models import *
from model import ModelLoader

# Configure logging
logging.basicConfig(filename="model.log", level=logging.INFO)


def main() -> None:
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
        benchmark_onnx_model(ort_session)
        predict_onnx_model(ort_session, img_batch, args.topk, model_loader.categories)

    # OpenVINO
    if args.mode in ["ov", "all"]:
        ov_model = init_ov_model(args.onnx_path)
        benchmark_ov_model(ov_model)
        predict_ov_model(ov_model, img_batch, args.topk, model_loader.categories)

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

            model = init_cuda_model(model_loader, device, precision)

            # If the configuration is not for TensorRT, store the model under a PyTorch key
            if not is_trt:
                models[f"PyTorch_{device}"] = model
            else:
                # If it is for TensorRT, determine the mode (FP32 or FP16) and store under a TensorRT key
                mode = "fp32" if precision == torch.float32 else "fp16"
                models[f"trt_{mode}"] = model

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
