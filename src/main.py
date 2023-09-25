from onnx_to_tensorrt import convert_onnx_to_tensorrt
from infer import run_inference, load_cifar10_images


def main():
    # Convert ONNX model to TensorRT
    convert_onnx_to_tensorrt("resnet50.onnx", "resnet50_fp16.engine", "FP16")
    convert_onnx_to_tensorrt("resnet50.onnx", "resnet50_fp32.engine", "FP32")

    # Load sample images and labels from CIFAR10
    images, labels = load_cifar10_images()

    # Run inference and compare results
    run_inference(images, labels, "resnet50.onnx", "resnet50_fp16.engine", "resnet50_fp32.engine")


if __name__ == "__main__":
    main()
