import argparse
import logging
import openvino as ov
import torch
import torch_tensorrt
from typing import List, Tuple, Union, Dict, Any
import onnxruntime as ort
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model import ModelLoader
from image_processor import ImageProcessor
from benchmark import PyTorchBenchmark, ONNXBenchmark, OVBenchmark
from onnx_exporter import ONNXExporter
from ov_exporter import OVExporter

# Configure logging
logging.basicConfig(filename="model.log", level=logging.INFO)


def run_benchmark(
    model: torch.nn.Module,
    device: str,
    dtype: torch.dtype,
    ort_session: ort.InferenceSession = None,
    onnx: bool = False,
) -> None:
    """
    Run and log the benchmark for the given model, device, and dtype.

    :param onnx:
    :param ort_session:
    :param model: The model to be benchmarked.
    :param device: The device to run the benchmark on ("cpu" or "cuda").
    :param dtype: The data type to be used in the benchmark (typically torch.float32 or torch.float16).
    """
    if onnx:
        logging.info(f"Running Benchmark for ONNX")
        benchmark = ONNXBenchmark(ort_session, input_shape=(32, 3, 224, 224))
    else:
        logging.info(f"Running Benchmark for {device.upper()}")
        benchmark = PyTorchBenchmark(model, device=device, dtype=dtype)
    benchmark.run()


def make_prediction(
    model: Union[torch.nn.Module, ort.InferenceSession],
    img_batch: Union[torch.Tensor, np.ndarray],
    topk: int,
    categories: List[str],
    precision: torch.dtype = None,
) -> None:
    """
    Make and print predictions for the given model, img_batch, topk, and categories.

    :param model: The model (or ONNX Runtime InferenceSession) to make predictions with.
    :param img_batch: The batch of images to make predictions on.
    :param topk: The number of top predictions to show.
    :param categories: The list of categories to label the predictions.
    :param precision: The data type to be used for the predictions (typically torch.float32 or torch.float16) for PyTorch models.
    """
    is_onnx_model = isinstance(model, ort.InferenceSession)
    is_ov_model = isinstance(model, ov.CompiledModel)

    if is_onnx_model:
        # Get the input name for the ONNX model.
        input_name = model.get_inputs()[0].name

        # Run the model with the properly named input.
        ort_inputs = {input_name: img_batch}
        ort_outs = model.run(None, ort_inputs)

        # Assuming the model returns a list with one array of class probabilities.
        if len(ort_outs) > 0:
            prob = ort_outs[0]

            # Checking if prob has more than one dimension and selecting the right one.
            if prob.ndim > 1:
                prob = prob[0]

            # Apply Softmax to get probabilities
            prob = np.exp(prob) / np.sum(np.exp(prob))
    elif is_ov_model:
        # For OV, the input name is usually the first input
        input_name = next(iter(model.inputs))
        outputs = model.infer({input_name: img_batch})

        # Assuming the model returns a dictionary with one key for class probabilities
        prob_key = next(iter(outputs))
        prob = outputs[prob_key]

        # Apply Softmax to get probabilities
        prob = np.exp(prob) / np.sum(np.exp(prob))

    else:  # PyTorch Model
        img_batch = img_batch.clone().to(precision)

        model.eval()
        with torch.no_grad():
            outputs = model(img_batch.to(precision))
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()

    top_indices = prob.argsort()[-topk:][::-1]
    top_probs = prob[top_indices]

    for i in range(topk):
        probability = top_probs[i]
        if is_onnx_model:
            # Accessing the DataFrame by row number using .iloc[]
            class_label = categories.iloc[top_indices[i]].item()
        else:
            class_label = categories[0][int(top_indices[i])]
        logging.info(f"#{i + 1}: {int(probability * 100)}% {class_label}")


def run_all_benchmarks(
    models: Dict[str, Any], img_batch: np.ndarray
) -> Dict[str, float]:
    """
    Run benchmarks for all models and return a dictionary of average inference times.

    :param models: Dictionary of models. Key is model type ("onnx", "ov", "pytorch", "trt_fp32", "trt_fp16"), value is the model.
    :param img_batch: The batch of images to run the benchmark on.
    :return: Dictionary of average inference times. Key is model type, value is average inference time.
    """
    results = {}

    # ONNX benchmark
    onnx_benchmark = ONNXBenchmark(models["onnx"], img_batch.shape)
    avg_time_onnx = onnx_benchmark.run()
    results["onnx"] = avg_time_onnx

    # OpenVINO benchmark
    ov_benchmark = OVBenchmark(models["ov"], img_batch.shape)
    avg_time_ov = ov_benchmark.run()
    results["ov"] = avg_time_ov

    # PyTorch benchmark
    pytorch_benchmark = PyTorchBenchmark(models["pytorch"], img_batch.shape)
    avg_time_pytorch = pytorch_benchmark.run()
    results["pytorch"] = avg_time_pytorch

    # TensorRT benchmarks
    for mode in ["fp32", "fp16"]:
        trt_benchmark = PyTorchBenchmark(models[f"trt_{mode}"], img_batch.shape)
        avg_time_trt = trt_benchmark.run()
        results[f"trt_{mode}"] = avg_time_trt

    return results


def plot_benchmark_results(results: Dict[str, float]):
    """
    Plot the benchmark results using Seaborn.

    :param results: Dictionary of average inference times. Key is model type, value is average inference time.
    """
    # Convert dictionary to two lists for plotting
    models = list(results.keys())
    times = list(results.values())

    # Plot
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=times, y=models, palette="rocket")
    ax.set(xlabel="Average Inference Time (ms)", ylabel="Model Type")
    plt.show()


def main() -> None:
    """
    Main function to run inference, benchmarks, and predictions on the model
    using provided image and optional parameters.
    """
    # Initialize ArgumentParser with description
    parser = argparse.ArgumentParser(description="PyTorch Inference")
    parser.add_argument(
        "--image_path",
        type=str,
        default="./inference/cat3.jpg",
        help="Path to the image to predict",
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="Number of top predictions to show"
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="./inference/model.onnx",
        help="Path where model in ONNX format will be exported",
    )
    parser.add_argument(
        "--mode",
        choices=["onnx", "ov", "cuda", "all"],
        required=True,
        help="Mode for exporting and running the model. Choices are: onnx, ov, cuda or all.",
    )

    args = parser.parse_args()

    models = {}

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and image processor
    model_loader = ModelLoader(device=device)
    img_processor = ImageProcessor(img_path=args.image_path, device=device)
    img_batch = img_processor.process_image()

    if args.mode == "onnx" or args.mode == "all":
        onnx_path = args.onnx_path

        # Export the model to ONNX format using ONNXExporter
        onnx_exporter = ONNXExporter(model_loader.model, device, onnx_path)
        onnx_exporter.export_model()

        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

        models["onnx"] = ort_session

        # Run benchmark
        run_benchmark(None, None, None, ort_session, onnx=True)

        # Make prediction
        print(f"Making prediction with {ort.get_device()} for ONNX model")
        make_prediction(
            ort_session,
            img_batch.cpu().numpy(),
            topk=args.topk,
            categories=model_loader.categories,
        )
    if args.mode == "ov" or args.mode == "all":
        # Export the ONNX model to OpenVINO
        ov_exporter = OVExporter(args.onnx_path)
        ov_model = ov_exporter.export_model()

        models["ov"] = ov_model

        # Benchmark the OpenVINO model
        ov_benchmark = OVBenchmark(ov_model, input_shape=(1, 3, 224, 224))
        ov_benchmark.run()

        # Run inference using the OpenVINO model
        img_batch_ov = (
            img_processor.process_image().cpu().numpy()
        )  # Assuming batch size of 1
        print(f"Making prediction with OpenVINO model")
        make_prediction(
            ov_benchmark.compiled_model,
            img_batch_ov,
            topk=args.topk,
            categories=model_loader.categories,
        )
    if args.mode == "cuda" or args.mode == "all":
        # Define configurations for which to run benchmarks and make predictions
        configs = [
            ("cpu", torch.float32),
            ("cuda", torch.float32),
            ("cuda", torch.float16),
        ]

        for device, precision in configs:
            cuda_model = model_loader.model.to(device)
            models["pytorch"] = model_loader.model

            if device == "cuda":
                print(f"Tracing {device} model")
                cuda_model = torch.jit.trace(
                    cuda_model, [torch.randn((1, 3, 224, 224)).to(device)]
                )

            if (
                device == "cuda"
                and precision == torch.float32
                or precision == torch.float16
            ):
                print("Compiling TensorRT model")
                trt_model = torch_tensorrt.compile(
                    cuda_model,
                    inputs=[torch_tensorrt.Input((32, 3, 224, 224), dtype=precision)],
                    enabled_precisions={precision},
                    truncate_long_and_double=True,
                )
                if precision == torch.float32:
                    models["trt_fp32"] = trt_model
                else:
                    models["trt_fp16"] = trt_model

            print(f"Making prediction with {device} model in {precision} precision")
            make_prediction(
                trt_model,
                img_batch.to(device),
                args.topk,
                model_loader.categories,
                precision,
            )

            print(f"Running Benchmark for {device} model in {precision} precision")
            run_benchmark(trt_model, device, precision)
    if args.mode == "all":
        # Run all benchmarks
        results = run_all_benchmarks(models, img_batch)

        # Plot results
        plot_benchmark_results(results)


if __name__ == "__main__":
    main()
