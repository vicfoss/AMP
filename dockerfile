# Use the NVIDIA L4T ML image compatible with JetPack 5.1.4
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Set the working directory
WORKDIR /app

# Copy all application files (Python scripts, model, and media files)
COPY . /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    protobuf-compiler \
    libprotoc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install additional Python dependencies
RUN pip3 install \
    numpy==1.26.1 \
    ultralytics \
    imageio[ffmpeg] \
    pillow \
    matplotlib \
    easyocr \
    pandas \
    seaborn \
    scipy \
    opencv-python \
    opencv-python-headless \
    scikit-image

# Set the entry point for your application
#CMD ["python3", "ALPR.py"]
CMD ["python3", "multi_stream.py"]