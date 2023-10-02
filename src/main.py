import argparse
import logging
import onnx
import torch
import torch_tensorrt
from typing import List, Tuple
import onnxruntime as ort
import numpy as np

from model import ModelLoader
from image_processor import ImageProcessor
from benchmark import Benchmark
from onnx_exporter import ONNXExporter

# Configure logging
logging.basicConfig(filename='model.log', level=logging.INFO)


def run_benchmark(model: torch.nn.Module, device: str, dtype: torch.dtype) -> None:
    """
    Run and log the benchmark for the given model, device, and dtype.

    :param model: The model to be benchmarked.
    :param device: The device to run the benchmark on ("cpu" or "cuda").
    :param dtype: The data type to be used in the benchmark (typically torch.float32 or torch.float16).
    """
    logging.info(f"Running Benchmark for {device.upper()}")
    benchmark = Benchmark(model, device=device, dtype=dtype)
    benchmark.run()


def make_prediction(
    model: torch.nn.Module,
    img_batch: torch.Tensor,
    topk: int,
    categories: List[str],
    precision: torch.dtype,
) -> None:
    """
    Make and print predictions for the given model, img_batch, topk, and categories.

    :param model: The model to make predictions with.
    :param img_batch: The batch of images to make predictions on.
    :param topk: The number of top predictions to show.
    :param categories: The list of categories to label the predictions.
    :param precision: The data type to be used for the predictions (typically torch.float32 or torch.float16).
    """
    # Clone img_batch to avoid in-place modifications
    img_batch = img_batch.clone().to(precision)

    model.eval()
    with torch.no_grad():
        outputs = model(img_batch.to(precision))
    prob = torch.nn.functional.softmax(outputs[0], dim=0)

    probs, classes = torch.topk(prob, topk)
    for i in range(topk):
        probability = probs[i].item()
        class_label = categories[0][int(classes[i])]
        logging.info(f"#{i + 1}: {int(probability * 100)}% {class_label}")


def make_prediction_onnx(
        ort_session: ort.InferenceSession,
        img_batch: np.ndarray,
        topk: int,
        categories: List[str],
) -> None:
    """
    Make and print predictions for the given ONNX model, img_batch, topk, and categories.

    :param ort_session: The ONNX Runtime InferenceSession to make predictions with.
    :param img_batch: The batch of images to make predictions on.
    :param topk: The number of top predictions to show.
    :param categories: The list of categories to label the predictions.
    """
    # Run ONNX inference
    outputs = ort_session.run(None, {"input": img_batch})

    # Assuming the model returns a list with one array of class probabilities
    if isinstance(outputs, list) and len(outputs) > 0:
        prob = outputs[0]
        print(f"\n\n\n\nPROB: {prob}")

        # Checking if prob has more than one dimension and selecting the right one
        if prob.ndim > 1:
            prob = prob[0]

        top_indices = prob.argsort()[-topk:][::-1]
        top_probs = prob[top_indices]

        for i in range(topk):
            probability = top_probs[i]
            class_label = categories[top_indices[i]]
            logging.info(f"#{i + 1}: {int(probability * 100)}% {class_label}")
    else:
        logging.error("Invalid model output")


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
        default="./inference/cat.png",
        help="Path to the image to predict",
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="Number of top predictions to show"
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="If we want export model to ONNX format"
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="./inference/model.onnx",
        help="Path where model in ONNX format will be exported",
    )

    args = parser.parse_args()

    # Setup device
    device = "cpu"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and image processor
    model_loader = ModelLoader(device=device)
    img_processor = ImageProcessor(img_path=args.image_path, device=device)
    img_batch = img_processor.process_image()

    if args.onnx:
        onnx_path = args.onnx_path

        # Export the model to ONNX format using ONNXExporter
        onnx_exporter = ONNXExporter(model_loader.model, device, onnx_path)
        onnx_exporter.export_model()

        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # Make prediction
        make_prediction_onnx(ort_session, img_batch.numpy(), topk=args.topk, categories=model_loader.categories)
        print("FINISHED ONNX")
        exit(0)


    # Make and log predictions for CPU
    print("Making prediction with CPU model")
    make_prediction(
        model_loader.model.to("cpu"), img_batch.to("cpu"), args.topk, model_loader.categories, torch.float32
    )

    # Run benchmarks for CPU and CUDA
    run_benchmark(model_loader.model.to("cpu"), "cpu", torch.float32)
    run_benchmark(model_loader.model.to("cuda"), "cuda", torch.float32)

    # Trace CUDA model
    print("Tracing CUDA model")
    traced_model = torch.jit.trace(
        model_loader.model, [torch.randn((1, 3, 224, 224)).to("cuda")]
    )

    # Compile, run benchmarks and make predictions with TensorRT models
    for precision in [torch.float32, torch.float16]:
        logging.info(
            f"Running Inference Benchmark for TensorRT with precision: {precision}"
        )
        trt_model = torch_tensorrt.compile(
            traced_model,
            inputs=[torch_tensorrt.Input((32, 3, 224, 224), dtype=precision)],
            enabled_precisions={precision},
            truncate_long_and_double=True,
        )
        run_benchmark(trt_model, "cuda", precision)
        print("Making prediction with TensorRT model")
        make_prediction(
            trt_model, img_batch, args.topk, model_loader.categories, precision
        )


if __name__ == "__main__":
    main()
