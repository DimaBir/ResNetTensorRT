# Argument for base image. Default is a neutral Python image.
ARG BASE_IMAGE=python:3.8-slim

# Use the base image specified by the BASE_IMAGE argument
FROM $BASE_IMAGE

# The rest of the Dockerfile remains the same...

# Install system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    libjpeg-dev \
    libpng-dev

# Argument to determine environment: cpu or gpu (default is cpu)
ARG ENVIRONMENT=cpu

# Copy the requirements file based on the environment into the container
COPY requirements_${ENVIRONMENT}.txt /workspace/requirements.txt

# Install Python packages
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Only install torch-tensorrt for GPU environment
RUN if [ "$ENVIRONMENT" = "gpu" ] ; then pip3 install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases ; fi

# Set the working directory
WORKDIR /workspace

# Copy local project files to /workspace in the image
COPY . /workspace
