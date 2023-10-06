
<img src="./inference/logo.png" width="60%">

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
    - [Steps to Run](#steps-to-run)
    - [Example Command](#example-command)
5. [RESULTS](#results) ![Static Badge](https://img.shields.io/badge/update-yellow)
    - [Results explanation](#results-explanation)
    - [Example Input](#example-input)
6. [Benchmark Implementation Details](#benchmark-implementation-details) ![New](https://img.shields.io/badge/-New-red)
    - [PyTorch CPU & CUDA](#pytorch-cpu--cuda)
    - [TensorRT FP32 & FP16](#tensorrt-fp32--fp16)
    - [ONNX](#onnx)
    - [OpenVINO](#openvino)
7. [Used methodologies](#used-methodologies) ![New](https://img.shields.io/badge/-New-red)
    - [TensorRT Optimization](#tensorrt-optimization)
    - [ONNX Exporter](#onnx-exporter)
    - [OV Exporter](#ov-exporter)
10. [Author](#author)
11. [References](#references)


<img src="./inference/plot.png" width="70%">

## Overview
This project demonstrates how to perform inference with a PyTorch model and optimize it using ONNX, OpenVINO, NVIDIA TensorRT. The script loads a pre-trained ResNet-50 model from torchvision, performs inference on a user-provided image, and prints the top-K predicted classes. Additionally, the script benchmarks the model's performance in the following configurations: CPU, CUDA, TensorRT-FP32, and TensorRT-FP16, providing insights into the speedup gained through optimization.

## Requirements
- This repo cloned
- Docker
- NVIDIA GPU (for CUDA and TensorRT benchmarks and optimizations)
- Python 3.x
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide) (for running the Docker container with GPU support)

### Steps to Run

```sh
# 1. Build the Docker Image
docker build -t awesome-tensorrt

# 2. Run the Docker Container
docker run --gpus all --rm -it awesome-tensorrt

# 3. Run the Script inside the Container
python src/main.py
```

### Arguments
- `--image_path`: (Optional) Specifies the path to the image you want to predict.
- `--topk`: (Optional) Specifies the number of top predictions to show. Defaults to 5 if not provided.
- `--mode`: Specifies the mode for exporting and running the model. Choices are: `onnx`, `ov`, `all`.

### Example Command
```sh
python src/main.py --topk 3 --mode=all
```

This command will run predictions on the default image (`./inference/cat3.jpg`), show the top 3 predictions and run all models (PyTorch CPU, CUDA, ONNX, OV, TRT-FP16, TRT-FP32). At the end results plot will be saved to `./inference/plot.png`

## RESULTS
### Inference Benchmark Results
<img src="./inference/plot.png" width="70%">

### Results explanation
  - `PyTorch_cpu: 973.52 ms` indicate the average batch time when running `PyTorch` model on `CPU` device.
  - `PyTorch_cuda: 41.11 ms` indicate the average batch time when running `PyTorch` model on `CUDA` device.
  - `TRT_fp32: 19.10 ms` shows the average batch time when running the model with `TensorRT` using `float32` precision.
  - `TRT_fp16: 7.22 ms` indicate the average batch time when running the model with `TensorRT` using `float16` precision.
  - `ONNX: 15.38 ms` indicate the average batch inference time when running the `PyTorch` converted to the `ONNX` model on the `CPU` device.
  - `OpenVINO: 14.04 ms` indicate the average batch inference time when running the `ONNX` model converted to `OpenVINO` on the `CPU` device.

### Example Input
Here is an example of the input image to run predictions and benchmarks on:

<img src="./inference/cat3.jpg" width="20%">

## Benchmark Implementation Details
Here you can see flow for each model and benchmark.

### PyTorch CPU & CUDA
In the provided code, we perform inference using the native PyTorch framework on both CPU and GPU (CUDA) configurations. This serves as a baseline to compare the performance improvements gained from other optimization techniques.

#### Flow:
1. The ResNet-50 model is loaded from torchvision and, if available, transferred to the GPU.
2. Inference is performed on the provided image using the specified model.
3. Benchmark results, including average inference time, are logged for both the CPU and CUDA setups.

### TensorRT FP32 & FP16
TensorRT offers significant performance improvements by optimizing the neural network model. In this code, we utilize TensorRT's capabilities to run benchmarks in both FP32 (single precision) and FP16 (half precision) modes.

#### Flow:
1. Load the ResNet-50 model.
2. Convert the PyTorch model to TensorRT format with the specified precision.
3. Perform inference on the provided image.
4. Log the benchmark results for the specified TensorRT precision mode.

### ONNX
The code includes an exporter that converts the PyTorch ResNet-50 model to ONNX format, allowing it to be inferred using ONNX Runtime. This provides a flexible, cross-platform solution for deploying the model.

#### Flow:
1. The ResNet-50 model is loaded.
2. Using the ONNX exporter utility, the PyTorch model is converted to ONNX format.
3. ONNX Runtime session is created.
4. Inference is performed on the provided image using the ONNX model.
5. Benchmark results are logged for the ONNX model.

### OpenVINO
OpenVINO is a toolkit from Intel that optimizes deep learning model inference for Intel CPUs, GPUs, and other hardware. In the code, we convert the ONNX model to OpenVINO's format and then run benchmarks using the OpenVINO runtime.

#### Flow:
1. The ONNX model (created in the previous step) is loaded.
2. Convert the ONNX model to OpenVINO's IR format.
3. Create an inference engine using OpenVINO's runtime.
4. Perform inference on the provided image using the OpenVINO model.
5. Benchmark results, including average inference time, are logged for the OpenVINO model.

## Used methodologies
### TensorRT Optimization
TensorRT is a high-performance deep learning inference optimizer and runtime library developed by NVIDIA. It is designed for optimizing and deploying trained neural network models on production environments. This project supports TensorRT optimizations in both FP32 (single precision) and FP16 (half precision) modes, offering different trade-offs between inference speed and model accuracy.

#### Features
- **Performance Boost**: TensorRT can significantly accelerate the inference of neural network models, making it suitable for deployment in resource-constrained environments.
- **Precision Modes**: Supports FP32 for maximum accuracy and FP16 for faster performance with a minor trade-off in accuracy.
- **Layer Fusion**: TensorRT fuses layers and tensors in the neural network to reduce memory access overhead and improve execution speed.
- **Dynamic Tensor Memory**: Efficiently handles varying batch sizes without re-optimization.

#### Usage
To employ TensorRT optimizations in the project, use the `--mode all` argument when running the main script.
This will initiate all models including PyTorch models that will be compiled to TRT model with `FP16` and `FP32` precision modes. Then, in one of the steps, will run inference on the specified image using the TensorRT-optimized model.
Example:
```sh
python src/main.py --mode all
```
#### Requirements
Ensure you have the TensorRT library and the torch_tensorrt package installed in your environment. Also, for FP16 optimizations, it's recommended to have a GPU that supports half-precision arithmetic (like NVIDIA GPUs with Tensor Cores).

### ONNX Exporter
ONNX Model Exporter (`ONNXExporter`) utility is incorporated within this project to enable the conversion of the native PyTorch model into the ONNX format.
Using the ONNX format, inference and benchmarking can be performed with the ONNX Runtime, which offers platform-agnostic optimizations and is widely supported across numerous platforms and devices.

#### Features
- **Standardized Format**: ONNX provides an open-source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types.
- **Interoperability**: Models in ONNX format can be used across a variety of frameworks, tools, runtimes, and compilers.
- **Optimizations**: The ONNX Runtime provides performance optimizations for both cloud and edge devices.

#### Usage
To leverage the `ONNXExporter` and conduct inference using the ONNX Runtime, utilize the `--mode onnx` argument when executing the main script.
This will initiate the conversion process and then run inference on the specified image using the ONNX model.
Example:
```sh
python src/main.py --mode onnx
```

#### Requirements
Ensure the ONNX library is installed in your environment to use the ONNXExporter. Additionally, if you want to run inference using the ONNX model, make sure you have the ONNX Runtime installed.

### OV Exporter
OpenVINO Model Exporter utility (`OVExporter`) has been integrated into this project to facilitate the conversion of the ONNX model to the OpenVINO format.
This enables inference and benchmarking using OpenVINO, a framework optimized for Intel hardware, providing substantial speed improvements especially on CPUs.

#### Features
- **Model Optimization**: Converts the ONNX model to OpenVINO's Intermediate Representation (IR) format. This optimized format allows for faster inference times on Intel hardware.
- **Versatility**: OpenVINO can target a variety of Intel hardware devices such as CPUs, integrated GPUs, FPGAs, and VPUs.
- **Ease of Use**: The `OVExporter` provides a seamless transition from ONNX to OpenVINO, abstracting the conversion details and providing a straightforward interface.

#### Usage
To utilize `OVExporter` and perform inference using OpenVINO, use the `--mode ov` argument when running the main script.
This will trigger the conversion process and subsequently run inference on the provided image using the optimized OpenVINO model.
Example:
```sh
python src/main.py --mode ov
```

#### Requirements
Ensure you have the OpenVINO Toolkit installed and the necessary dependencies set up to use OpenVINO's model optimizer and inference engine.


## ONNX Exporter
The ONNX Exporter utility is integrated into this project to allow the conversion of the PyTorch model to ONNX format, enabling inference and benchmarking using ONNX Runtime. The ONNX model can provide hardware-agnostic optimizations and is widely supported across various platforms and devices.

## Author
[DimaBir](https://github.com/DimaBir)

## References
- [ResNetTensorRT Project](https://github.com/DimaBir/ResNetTensorRT/tree/main)
