import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cifar10_images():
    # Load a few images from CIFAR10
    transform = transforms.Compose([
        transforms.Resize(224),  # Resizing to fit ResNet input size
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
    images, labels = next(iter(dataloader))
    return images.to(device), labels.to(device)  # Move images and labels to the GPU


def run_inference(images, labels, onnx_file_path, fp16_engine_file_path, fp32_engine_file_path):
    # Define the transformations for the images
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Run inference with PyTorch model
    pytorch_results = []
    for image in images:
        result = infer_with_pytorch_model(image, onnx_file_path)
        pytorch_results.append(result)

    # Run inference with TensorRT FP16 engine
    fp16_results = []
    for image in images:
        result = infer_with_tensorrt_engine(image, fp16_engine_file_path)
        fp16_results.append(result)

    # Run inference with TensorRT FP32 engine
    fp32_results = []
    for image in images:
        result = infer_with_tensorrt_engine(image, fp32_engine_file_path)
        fp32_results.append(result)

    # Compare and print results
    for i, label in enumerate(labels):
        print(f"Image {i + 1} - Ground Truth: {label.item()}")
        print(
            f"PyTorch Prediction: {pytorch_results[i]}, FP16 Prediction: {fp16_results[i]}, FP32 Prediction: {fp32_results[i]}")
        print("-------------------------------------------------")


def infer_with_pytorch_model(image, model_path):
    # Load the ONNX model
    model = torch.jit.load(model_path).to(device)  # Move model to the GPU
    model.eval()

    # Run inference
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
        _, predicted = outputs.max(1)
    return predicted.item()


def infer_with_tensorrt_engine(image, engine_file_path):
    # Initialize CUDA
    cuda.init()

    # Load the TensorRT engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # Convert the image to the format expected by TensorRT
    image = image.cpu().numpy().astype(np.float32)  # Move image to the CPU
    image = np.transpose(image, (1, 2, 0))

    # Allocate buffers and run inference
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': cuda_mem})
        else:
            outputs.append({'host': host_mem, 'device': cuda_mem})
    np.copyto(inputs[0]['host'], image.ravel())
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    engine(context, bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    # Get the predicted label
    predicted_label = np.argmax(outputs[0]['host'])
    return predicted_label
