# Argument for base image. Default is a neutral Python image.
ARG BASE_IMAGE=python:3.12-slim

# Use the base image specified by the BASE_IMAGE argument
FROM $BASE_IMAGE

# Argument to determine environment: cpu or gpu (default is cpu)
ARG ENVIRONMENT=cpu

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
