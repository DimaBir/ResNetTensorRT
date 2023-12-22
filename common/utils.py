import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple


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
