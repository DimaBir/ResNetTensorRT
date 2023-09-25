# Use an official TensorRT base image
FROM nvcr.io/nvidia/tensorrt:23.08-py3

# Install system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    git

# Install Python packages
RUN pip3 install torch torchvision torch-tensorrt pandas Pillow numpy

# Set the working directory
WORKDIR /workspace

# Copy local project files to /workspace in the image
COPY . /workspace
