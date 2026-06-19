# Use lightweight Python image as the base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install critical dependencies
# - ffmpeg: required to transcode the video for the streamlit web player
# - libglib2.0-0, libsm6, libxext6: required for OpenCV to work
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy ONLY the pyproject.toml file to the container first
COPY pyproject.toml .

# Force the CPU-only version of PyTorch for Cloud Deployment
# --no-sync prevents uv from downloading packages twice
RUN uv add torch torchvision --index https://download.pytorch.org/whl/cpu --no-sync

# Install the dependencies and generate the .venv
RUN uv sync

# Copy ONLY source code, ignoring everything else in the folder
COPY app.py .
COPY src/ ./src/

# This forces torchvision to download the 160MB file into the Linux image
RUN uv run python -c "from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights; fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)"

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]