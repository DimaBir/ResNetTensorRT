import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import torch
import onnxruntime as ort

from src.benchmark_class import PyTorchBenchmark, ONNXBenchmark, OVBenchmark


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
        logging.info(f"Running Benchmark for {device.upper()} and precision {dtype}")
        benchmark = PyTorchBenchmark(model, device=device, dtype=dtype)
    benchmark.run()


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
    logging.info(f"Running benchmark inference for ONNX model")
    onnx_benchmark = ONNXBenchmark(models["onnx"], img_batch.shape)
    avg_time_onnx = onnx_benchmark.run()
    results["ONNX"] = avg_time_onnx

    # OpenVINO benchmark
    logging.info(f"Running benchmark inference for OpenVINO model")
    ov_benchmark = OVBenchmark(models["ov"], img_batch.shape)
    avg_time_ov = ov_benchmark.run()
    results["OpenVINO"] = avg_time_ov

    # PyTorch + TRT benchmark
    configs = [
        ("cpu", torch.float32, False),
        ("cuda", torch.float32, False),
        ("cuda", torch.float32, True),
        ("cuda", torch.float16, True),
    ]
    for device, precision, is_trt in configs:
        model_to_use = models[f"PyTorch_{device}"].to(device)

        if not is_trt:
            pytorch_benchmark = PyTorchBenchmark(
                model_to_use, device=device, dtype=precision
            )
            logging.info(f"Running benchmark inference for PyTorch_{device} model")
            avg_time_pytorch = pytorch_benchmark.run()
            results[f"PyTorch_{device}"] = avg_time_pytorch

        else:
            # TensorRT benchmarks
            if precision == torch.float32 or precision == torch.float16:
                mode = "fp32" if precision == torch.float32 else "fp16"
                logging.info(f"Running benchmark inference for TRT_{mode} model")
                trt_benchmark = PyTorchBenchmark(
                    models[f"trt_{mode}"], device=device, dtype=precision
                )
                avg_time_trt = trt_benchmark.run()
                results[f"TRT_{mode}"] = avg_time_trt

    return results


def plot_benchmark_results(results: Dict[str, float]):
    """
    Plot the benchmark results using Seaborn.

    :param results: Dictionary of average inference times. Key is model type, value is average inference time.
    """
    # Convert dictionary to two lists for plotting
    models = list(results.keys())
    times = list(results.values())

    # Create a DataFrame for plotting
    data = pd.DataFrame({"Model": models, "Time": times})

    # Sort the DataFrame by Time
    data = data.sort_values("Time", ascending=True)

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=data["Time"], y=data["Model"], hue=data["Model"], palette="rocket", legend=False)

    # Adding the actual values on the bars
    for index, value in enumerate(data["Time"]):
        ax.text(value, index, f"{value:.2f} ms", color="black", ha="left", va="center")

    plt.xlabel("Average Inference Time (ms)")
    plt.ylabel("Model Type")
    plt.title("ResNet50 - Inference Benchmark Results")

    # Save the plot to a file
    plt.savefig("./inference/plot.png", bbox_inches="tight")
    plt.show()
