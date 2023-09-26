# ResNet-50 Inference with TensorRT
## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Steps to Run](#steps-to-run)
4. [Example Command](#example-command)
5. [Results](#results)
   - [Example of Results](#example-of-results)
   - [Explanation of Results](#explanation-of-results)
6. [Author](#author)
7. [References](#references)
8. [Notes](#notes)

## Overview
This project demonstrates how to perform inference with a PyTorch model and optimize it using NVIDIA TensorRT. The script loads a pre-trained ResNet-50 model from torchvision, performs inference on a user-provided image, and prints the top-K predicted classes. Additionally, the script benchmarks the model's performance in the following configurations: CPU, CUDA, TensorRT-FP32, and TensorRT-FP16, providing insights into the speedup gained through optimization.

## Requirements
- Docker
- NVIDIA GPU (for CUDA and TensorRT benchmarks and optimizations)
- Python 3.x
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide) (for running the Docker container with GPU support)

## Steps to Run

```sh
# 1. Build the Docker Image
docker build -t awesome-tesnorrt .

# 2. Run the Docker Container
docker run --gpus all --rm -it awesome-tesnorrt

# 3. Run the Script inside the Container
python main.py --image_path /path-to-image/image.jpg --topk 2
```

### Arguments
- `--image_path`: Specifies the path to the image you want to predict.
- `--topk`: (Optional) Specifies the number of top predictions to show. Defaults to 5 if not provided.

## Example Command
```sh
python src/main.py --image_path ./inference/cat3.jpg --topk 3 --show_image
```

This command will run predictions on the image at the specified path, show the top 3 predictions, and display the image. If you do not want to display the image, omit the `--show_image` flag. For the default 5 top predictions, omit the `--topk` argument or set it to 5.

## Results

The results of the predictions and benchmarks are saved to `model.log`. This log file contains information about the predicted class for the input image and the average batch time for the different configurations during the benchmark.

### Example of Results
Here is an example of the contents of `model.log` after running predictions and benchmarks on this image:

<img src="./inference/cat3.jpg" width="20%">

```
My prediction: %33 tabby
My prediction: %26 Egyptian cat
Running Benchmark for CPU
Average batch time: 942.47 ms
Running Benchmark for CUDA
Average batch time: 41.02 ms
Compiling and Running Inference Benchmark for TensorRT with precision: torch.float32
Average batch time: 19.20 ms
Compiling and Running Inference Benchmark for TensorRT with precision: torch.float16
Average batch time: 7.25 ms
```

### Explanation of Results
- First k lines show the topk predictions. For example, `My prediction: %33 tabby` displays the highest confidence prediction made by the model for the input image, confidence level (`%33`), and the predicted class (`tabby`).
- The following lines provide information about the average batch time for running the model in different configurations:
  - `Running Benchmark for CPU` and `Average batch time: 942.47 ms` indicate the average batch time when running the model on the CPU.
  - `Running Benchmark for CUDA` and `Average batch time: 41.02 ms` indicate the average batch time when running the model on CUDA.
  - `Compiling and Running Inference Benchmark for TensorRT with precision: torch.float32` and `Average batch time: 19.20 ms` show the average batch time when running the model with TensorRT using `float32` precision.
  - `Compiling and Running Inference Benchmark for TensorRT with precision: torch.float16` and `Average batch time: 7.25 ms` indicate the average batch time when running the model with TensorRT using `float16` precision.

## Author
[DimaBir](https://github.com/DimaBir)

## References
- [ResNetTensorRT Project](https://github.com/DimaBir/ResNetTensorRT/tree/main)

## Notes
- The project uses a Docker container built on top of the NVIDIA TensorRT image to ensure that all dependencies, including CUDA and TensorRT, are correctly installed and configured.
- Please ensure you have the NVIDIA Container Toolkit installed to run the container with GPU support.
