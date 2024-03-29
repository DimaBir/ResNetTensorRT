
<img src="./inference/logo2.png" width="100%">

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
    - [Steps to Run](#steps-to-run)
    - [Example Command](#example-command)
3. [Results](#results)
   - [CPU Results](#cpu-results) ![Static Badge](https://img.shields.io/badge/update-orange)
   - [GPU (CUDA) Results](#gpu-cuda-results) ![Static Badge](https://img.shields.io/badge/update-orange)
   - [CPU Results M1 Pro](#cpu-results-m1-pro) ![New](https://img.shields.io/badge/-New-842E5B)
6. [Benchmark Implementation Details](#benchmark-implementation-details) ![New](https://img.shields.io/badge/-New-842E5B)
    - [PyTorch CPU & CUDA](#pytorch-cpu--cuda)
    - [TensorRT FP32 & FP16](#tensorrt-fp32--fp16)
    - [ONNX](#onnx)
    - [OpenVINO](#openvino)
7. [Author](#author)
8. [References](#references)


<img src="./inference/plot_trt.png" width="100%">

## Overview
This project showcases inference with PyTorch CNN models, such as ResNet50, EfficientNet, and MobileNet, and their optimization using ONNX, OpenVINO, and NVIDIA TensorRT. The script infers a user-specified image and displays top-K predictions. Benchmarking covers configurations like PyTorch CPU, ONNX CPU, OpenVINO CPU, PyTorch CUDA, TensorRT-FP32, and TensorRT-FP16.

The project is Dockerized for easy deployment:
1. **CPU-only Deployment** - Suitable for non-GPU systems (supports `PyTorch CPU`, `ONNX CPU`, and `OpenVINO CPU` models only).
2. **GPU Deployment** - Optimized for NVIDIA GPUs (supports all models: `PyTorch CPU`, `ONNX CPU`, `OpenVINO CPU`, `PyTorch CUDA`, `TensorRT-FP32`, and `TensorRT-FP16`).

Please look at the [Steps to Run](#steps-to-run) section for Docker instructions.


## Requirements
- Clone this repo:
  ```bash
   git clone https://github.com/DimaBir/ResNetTensorRT.git
   ```
- Python 3.x
- Verify that your system has Docker support
- NVIDIA GPU (for CUDA and TensorRT benchmarks and optimizations)
- NVIDIA drivers installed on the host machine.

## Steps to Run
### Building the Docker Image

1. **CPU-only Deployment**:
   ```bash
   docker build -t cpu_img .
   ```
   **Running**:
   ```bash
   docker run -it --rm cpu_img /bin/bash
   ```
   
2. **GPU (CUDA) Deployment**:
   ```bash
   docker build --build-arg ENVIRONMENT=gpu --build-arg BASE_IMAGE=nvcr.io/nvidia/tensorrt:23.08-py3 -t gpu_img .
   ```

   **Running**:
   ```bash
   docker run --gpus all -it --rm gpu_img
   ```

### Run the Script inside the Container
```sh
python main.py [--mode all]
```

### Arguments
- `--image_path`: (Optional) Specifies the path to the image you want to predict.
- `--topk`: (Optional) Specifies the number of top predictions to show. Defaults to 5 if not provided.
- `--mode`: (Optional) Specifies the model's mode for exporting and running. Choices are: `onnx`, `ov`, `cpu`, `cuda`, `tensorrt`, and `all`.  If not provided, it defaults to `all`.

### Example Command
```sh
python main.py --topk 3 --mode=all --image_path="./inference/cat3.jpg"
```

This command will run predictions on the chosen image (`./inference/cat3.jpg`), show the top 3 predictions, and run all available models. Note: plot created only for `--mode=all` and results plotted and saved to `./inference/plot.png`

## Results
### Example Input
Here is an example of the input image to run predictions and benchmarks on:

<img src="./inference/cat3.jpg" width="20%">


### Plot details:
1. **Average Inference Time**: This plot showcases the average time taken for inference across different model types and optimization techniques. The y-axis represents the model type (e.g., PyTorch CPU, TensorRT FP16, etc.), and the x-axis represents the average inference time in milliseconds. The shorter the bar, the faster the inference time.

2. **Throughput**: This plot compares the throughput achieved by different model types. Throughput is measured in terms of the number of images processed per second. The y-axis represents the model type, and the x-axis represents the throughput. A higher bar indicates better throughput, meaning the model can process more images in a given time frame.

These plots offer a comprehensive view of the performance improvements achieved by various inference optimization techniques, especially when leveraging TensorRT with different precision types like FP16 and FP32.

### CPU Results
<img src="./inference/plot_laptop.png" width="70%">

#### Prediction results
```
#1: 15% Egyptian cat
#2: 14% tiger cat
#3: 9% tabby
#4: 2% doormat
#5: 2% lynx
```
#### PC Setup Linux 
- CPU: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
- RAM: 16 GB
- GPU: None

### GPU (CUDA) Results
#### Inference Benchmark Results
<img src="./inference/plot_trt.png" width="100%">

#### Results explanation
  - `PyTorch_cpu: 31.93 ms` indicates the average batch time when running the `PyTorch` model on `CPU` device.
  - `PyTorch_cuda: 5.70 ms` indicates the average batch time when running the `PyTorch` model on the `CUDA` device.
  - `TRT_fp32: 1.69 ms` shows the average batch time when running the model with `TensorRT` using `float32` precision.
  - `TRT_fp16: 0.75 ms` indicates the average batch time when running the model with `TensorRT` using `float16` precision.
  - ![New](https://img.shields.io/badge/-New-842E5B)`ONNX: 16.25 ms` indicates the average batch inference time when running the `PyTorch` converted to the `ONNX` model on the `CPU` device.
  - ![New](https://img.shields.io/badge/-New-842E5B)`OpenVINO: 15.00 ms` indicates the average batch inference time when running the `ONNX` model converted to `OpenVINO` on the `CPU` device.

### Prediction results
```
#1: 15% Egyptian cat
#2: 14% tiger cat
#3: 9% tabby
#4: 2% doormat
#5: 2% lynx
```

### PC Setup 
- CPU: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz
- RAM: 32 GB
- GPU: GeForce RTX 3070 (CUDA)

### CPU Results M1 Pro
<img src="./inference/plot_mac.png" width="70%">

#### Prediction results
```
#1: 15% Egyptian cat
#2: 14% tiger cat
#3: 9% tabby
#4: 2% doormat
#5: 2% lynx
```
#### M1 PRO Setup 
- CPU: M1 Pro Chip
- RAM: 16 GB
- GPU: None

## Benchmark Implementation Details
Here you can see the flow for each model and benchmark.

### PyTorch CPU & CUDA
In the provided code, we perform inference using the native PyTorch framework on both CPU and GPU (CUDA) configurations. This is a baseline to compare the performance improvements gained from other optimization techniques.

#### Flow:
1. The ResNet-50 model is loaded from torchvision and, if available, transferred to the GPU.
2. Inference is performed on the provided image using the specified model.
3. Benchmark results, including average inference time, are logged for the CPU and CUDA setups.

### TensorRT FP32 & FP16
TensorRT offers significant performance improvements by optimizing the neural network model. This code uses TensorRT's capabilities to run benchmarks in FP32 (single precision) and FP16 (half precision) modes.

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
OpenVINO is a toolkit from Intel that optimizes deep learning model inference for Intel CPUs, GPUs, and other hardware. We convert the ONNX model to OpenVINO's format in the code and then run benchmarks using the OpenVINO runtime.

#### Flow:
1. The ONNX model (created in the previous step) is loaded.
2. Convert the ONNX model to OpenVINO's IR format.
3. Create an inference engine using OpenVINO's runtime.
4. Perform inference on the provided image using the OpenVINO model.
5. Benchmark results, including average inference time, are logged for the OpenVINO model.

## Author
[DimaBir](https://github.com/DimaBir)
  
## References
- **PyTorch**: [Official Documentation](https://pytorch.org/docs/stable/index.html)
- **Torch-TensorRT**: is a compiler for PyTorch/TorchScript, targeting NVIDIA GPUs via NVIDIA’s TensorRT Deep Learning Optimizer and Runtime. [Torch-TensorRT Documentation](https://pytorch.org/TensorRT/)
- **torch.onnx**: PyTorch's built-in ONNX exporter. [Documentation](https://pytorch.org/docs/stable/onnx.html)
- **OpenVINO**: Intel's toolkit for computer vision applications includes a model optimizer to convert trained models into a format suitable for optimal execution on end-point target devices. [Official Documentation](https://docs.openvinotoolkit.org/latest/index.html)
- **OpenVINO - Converting ONNX to OV**: [Convert Model From ONNX](https://docs.openvino.ai/2023.1/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html)
