import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict


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
    ax = sns.barplot(
        x=data["Time"],
        y=data["Model"],
        hue=data["Model"],
        palette="rocket",
        legend=False,
    )

    # Adding the actual values on the bars
    for index, value in enumerate(data["Time"]):
        ax.text(value, index, f"{value:.2f} ms", color="black", ha="left", va="center")

    plt.xlabel("Average Inference Time (ms)")
    plt.ylabel("Model Type")
    plt.title("ResNet50 - Inference Benchmark Results")

    # Save the plot to a file
    plt.savefig("./inference/plot.png", bbox_inches="tight")
    plt.show()


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

    return parser.parse_args()
