FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git ffmpeg libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone YOLOv5 repo and install its requirements
RUN git clone https://github.com/ultralytics/yolov5 && \
    pip install -r yolov5/requirements.txt

# Copy the rest of the files
COPY . .

# Run the Python detection script
CMD ["python", "main.py"]
