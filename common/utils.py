import argparse
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PLOT_OUTPUT_PATH = "./inference/plot.png"
DEFAULT_IMAGE_PATH = "./inference/cat3.jpg"
DEFAULT_ONNX_PATH = "./models/model.onnx"
DEFAULT_OV_PATH = "./models/model.ov"
DEFAULT_TOPK = 5
INFERENCE_MODES = ["onnx", "ov", "cpu", "cuda", "tensorrt", "all"]


def _create_sorted_dataframe(data: Dict[str, float], column_name: str, ascending: bool) -> pd.DataFrame:
    df = pd.DataFrame(list(data.items()), columns=["Model", column_name])
    return df.sort_values(column_name, ascending=ascending)


def _plot_bar_chart(ax, data: pd.DataFrame, x_col: str, y_col: str, 
                    xlabel: str, ylabel: str, title: str, palette: str, value_format: str):
    sns.barplot(x=data[x_col], y=data[y_col], hue=data[y_col], palette=palette, 
                ax=ax, legend=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    for index, value in enumerate(data[x_col]):
        ax.text(value, index, value_format.format(value), color="black", ha="left", va="center")


def plot_benchmark_results(results: Dict[str, Tuple[float, float]]):
    models = list(results.keys())
    times = {model: results[model][0] for model in models}
    throughputs = {model: results[model][1] for model in models}

    time_data = _create_sorted_dataframe(times, "Time", ascending=True)
    throughput_data = _create_sorted_dataframe(throughputs, "Throughput", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    _plot_bar_chart(ax1, time_data, "Time", "Model", 
                    "Average Inference Time (ms)", "Model Type",
                    "ResNet50 - Inference Benchmark Results", "rocket", "{:.2f} ms")

    _plot_bar_chart(ax2, throughput_data, "Throughput", "Model",
                    "Throughput (samples/sec)", "",
                    "ResNet50 - Throughput Benchmark Results", "viridis", "{:.2f}")

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, bbox_inches="tight")
    plt.show()

    print(f"Plot saved to {PLOT_OUTPUT_PATH}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Inference")

    parser.add_argument(
        "--image_path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the image to predict",
    )

    parser.add_argument(
        "--topk", 
        type=int, 
        default=DEFAULT_TOPK, 
        help="Number of top predictions to show"
    )

    parser.add_argument(
        "--onnx_path",
        type=str,
        default=DEFAULT_ONNX_PATH,
        help="Path where model in ONNX format will be exported",
    )

    parser.add_argument(
        "--ov_path",
        type=str,
        default=DEFAULT_OV_PATH,
        help="Path where model in OpenVINO format will be exported",
    )

    parser.add_argument(
        "--mode",
        choices=INFERENCE_MODES,
        default="all",
        help="Mode for exporting and running the model",
    )

    parser.add_argument(
        "-D",
        "--DEBUG",
        action="store_true",
        help="Enable debug mode",
    )

    return parser.parse_args()
