# Argument to determine environment: cpu or gpu (default is cpu)
ARG ENVIRONMENT=cpu

# Conditionally set the base image based on the environment
FROM python:3.8-slim as base_cpu
FROM nvcr.io/nvidia/tensorrt:23.08-py3 as base_gpu

FROM base_${ENVIRONMENT}

# Install required system packages conditionally
RUN apt-get update && apt-get install -y python3-pip git && \
    if [ "$ENVIRONMENT" = "gpu" ] ; then apt-get install -y libjpeg-dev libpng-dev ; fi

# Copy the requirements file based on the environment into the container
COPY requirements.txt /workspace/requirements.txt

# Install Python packages
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Only install torch-tensorrt for GPU environment
RUN if [ "$ENVIRONMENT" = "gpu" ] ; then pip3 install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases ; fi

# Set the working directory
WORKDIR /workspace

# Copy local project files to /workspace in the image
COPY . /workspace
