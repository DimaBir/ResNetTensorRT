import torch
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Type
from src.model import ModelLoader
from src.inference_base import InferenceBase


def plot_benchmark_results(results: Dict[str, Tuple[float, float]]):
    """
    Plot the benchmark results using Seaborn.

    :param results: Dictionary where the key is the model type and the value is a tuple (average inference time, throughput).
    """
    plot_path = "./inference/plot.png"

    # Extract data from the results
    models = list(results.keys())
    times = [value[0] for value in results.values()]
    throughputs = [value[1] for value in results.values()]

    # Create DataFrames for plotting
    time_data = pd.DataFrame({"Model": models, "Time": times})
    throughput_data = pd.DataFrame({"Model": models, "Throughput": throughputs})

    # Sort the DataFrames
    time_data = time_data.sort_values("Time", ascending=True)
    throughput_data = throughput_data.sort_values("Throughput", ascending=False)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot inference times
    sns.barplot(
        x=time_data["Time"],
        y=time_data["Model"],
        hue=time_data["Model"],
        palette="rocket",
        ax=ax1,
        legend=False,
    )
    ax1.set_xlabel("Average Inference Time (ms)")
    ax1.set_ylabel("Model Type")
    ax1.set_title("ResNet50 - Inference Benchmark Results")
    for index, value in enumerate(time_data["Time"]):
        ax1.text(value, index, f"{value:.2f} ms", color="black", ha="left", va="center")

    # Plot throughputs
    sns.barplot(
        x=throughput_data["Throughput"],
        y=throughput_data["Model"],
        hue=throughput_data["Model"],
        palette="viridis",
        ax=ax2,
        legend=False,
    )
    ax2.set_xlabel("Throughput (samples/sec)")
    ax2.set_ylabel("")
    ax2.set_title("ResNet50 - Throughput Benchmark Results")
    for index, value in enumerate(throughput_data["Throughput"]):
        ax2.text(value, index, f"{value:.2f}", color="black", ha="left", va="center")

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()

    print(f"Plot saved to {plot_path}")


def parse_arguments():
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
        default="./models/model.onnx",
        help="Path where model in ONNX format will be exported",
    )

    parser.add_argument(
        "--ov_path",
        type=str,
        default="./models/model.ov",
        help="Path where model in OpenVINO format will be exported",
    )

    parser.add_argument(
        "--mode",
        choices=["onnx", "ov", "cpu", "cuda", "tensorrt", "all"],
        default="all",
        help="Mode for exporting and running the model. Choices are: onnx, ov, cuda, tensorrt or all.",
    )

    parser.add_argument(
        "-D",
        "--DEBUG",
        action="store_true",
        help="Enable or disable debug capabilities.",
    )

    return parser.parse_args()


def check_cuda_availability() -> bool:
    """
    Check the availability of CUDA and TensorRT on the system.

    Determines if CUDA is available and if the 'torch_tensorrt' package is
    installed. Logs a warning if 'torch_tensorrt' is not installed.

    :return: True if CUDA is available and 'torch_tensorrt' is installed, False otherwise.
    """
    cuda_available = False
    if torch.cuda.is_available():
        try:
            import torch_tensorrt
            cuda_available = True
        except ImportError:
            logging.warning("torch-tensorrt is not installed. Running on CPU mode only.")
    return cuda_available


def run_inference_for_mode(
    inference_type: Type[InferenceBase],
    model_loader: ModelLoader,
    img_batch: torch.Tensor,
    device: torch.device,
    debug_mode: bool,
    benchmark_results: Dict[str, float]
) -> None:
    """
    Run the inference process for a given inference type and accumulate benchmark results.

    Initializes the specified inference type, performs benchmarking, and runs predictions
    on the provided image batch. The benchmark results are updated with the new results.

    :param inference_type: The class of the inference type to be used.
    :param model_loader: The model loader used to load the inference model.
    :param img_batch: The batch of images to run the inference on.
    :param device: The device (CPU or CUDA) to run the inference on.
    :param debug_mode: If True, additional debug information will be printed.
    :param benchmark_results: A dictionary to store the benchmark results.
    :return: None.
    """
    inference = inference_type(model_loader, device=device, debug_mode=debug_mode)
    benchmark_results[inference.name()] = inference.benchmark(img_batch)
    inference.predict(img_batch)