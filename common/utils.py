import argparse


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
