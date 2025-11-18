# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric dependencies
RUN pip3 install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for outputs
RUN mkdir -p models results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python3", "run_full_experiments.py", "--models", "gcn", "gat", "graphsage", "xgboost", "--epochs", "200", "--configs", "all"]
