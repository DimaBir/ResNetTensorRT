# Use an official TensorRT base image
FROM nvcr.io/nvidia/tensorrt:23.08-py3

# Install system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    git

# Copy the requirements.txt file into the container
COPY requirements.txt /workspace/requirements.txt

RUN pip3 install -q "openvino==2023.1.0.dev20230811"

# Install Python packages
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Install torch-tensorrt from the special location
RUN pip3 install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases

# Set the working directory
WORKDIR /workspace

# Copy local project files to /workspace in the image
COPY . /workspace