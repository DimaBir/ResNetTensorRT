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
