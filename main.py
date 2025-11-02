import logging
import warnings
from typing import Dict, Tuple

import torch

from common.utils import parse_arguments, plot_benchmark_results
from src.image_processor import ImageProcessor
from src.model import ModelLoader
from src.onnx_inference import ONNXInference
from src.ov_inference import OVInference
from src.pytorch_inference import PyTorchInference
from src.tensorrt_inference import TensorRTInference

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
logging.basicConfig(filename="inference.log", level=logging.INFO)

CUDA_AVAILABLE = False
if torch.cuda.is_available():
    try:
        import torch_tensorrt
        CUDA_AVAILABLE = True
    except ImportError:
        print("torch-tensorrt not installed. Running in CPU mode only.")


def _run_onnx_inference(args, model_loader, img_batch) -> Dict[str, Tuple[float, float]]:
    onnx_inference = ONNXInference(model_loader, args.onnx_path, debug_mode=args.DEBUG)
    benchmark_result = onnx_inference.benchmark(img_batch)
    onnx_inference.predict(img_batch)
    return {"ONNX (CPU)": benchmark_result}


def _run_openvino_inference(args, model_loader, img_batch) -> Dict[str, Tuple[float, float]]:
    ov_inference = OVInference(model_loader, args.ov_path, debug_mode=args.DEBUG)
    benchmark_result = ov_inference.benchmark(img_batch)
    ov_inference.predict(img_batch)
    return {"OpenVINO (CPU)": benchmark_result}


def _run_pytorch_cpu_inference(args, model_loader, img_batch) -> Dict[str, Tuple[float, float]]:
    pytorch_cpu_inference = PyTorchInference(model_loader, device="cpu", debug_mode=args.DEBUG)
    benchmark_result = pytorch_cpu_inference.benchmark(img_batch)
    pytorch_cpu_inference.predict(img_batch)
    return {"PyTorch (CPU)": benchmark_result}


def _run_pytorch_cuda_inference(args, model_loader, device, img_batch) -> Dict[str, Tuple[float, float]]:
    print("Running CUDA inference...")
    pytorch_cuda_inference = PyTorchInference(model_loader, device=device, debug_mode=args.DEBUG)
    benchmark_result = pytorch_cuda_inference.benchmark(img_batch)
    pytorch_cuda_inference.predict(img_batch)
    return {"PyTorch (CUDA)": benchmark_result}


def _run_tensorrt_inference(args, model_loader, device, img_batch) -> Dict[str, Tuple[float, float]]:
    results = {}
    precisions = [torch.float16, torch.float32]
    
    for precision in precisions:
        tensorrt_inference = TensorRTInference(
            model_loader, device=device, precision=precision, debug_mode=args.DEBUG
        )
        benchmark_result = tensorrt_inference.benchmark(img_batch)
        tensorrt_inference.predict(img_batch)
        results[f"TRT_{precision}"] = benchmark_result
    
    return results


def main():
    args = parse_arguments()

    if args.DEBUG:
        print("Debug mode enabled")

    benchmark_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_loader = ModelLoader(device=device)
    img_processor = ImageProcessor(img_path=args.image_path, device=device)
    img_batch = img_processor.process_image()

    if args.mode in ["onnx", "all"]:
        benchmark_results.update(_run_onnx_inference(args, model_loader, img_batch))

    if args.mode in ["ov", "all"]:
        benchmark_results.update(_run_openvino_inference(args, model_loader, img_batch))

    if args.mode in ["cpu", "all"]:
        benchmark_results.update(_run_pytorch_cpu_inference(args, model_loader, img_batch))

    if torch.cuda.is_available():
        if args.mode in ["cuda", "all"]:
            benchmark_results.update(_run_pytorch_cuda_inference(args, model_loader, device, img_batch))

        if args.mode in ["tensorrt", "all"]:
            benchmark_results.update(_run_tensorrt_inference(args, model_loader, device, img_batch))

    if args.mode == "all":
        plot_benchmark_results(benchmark_results)


if __name__ == "__main__":
    main()
