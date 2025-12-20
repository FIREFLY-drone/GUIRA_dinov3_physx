FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY fusion_head.py .

# Create models directory
RUN mkdir -p models

# Environment variables
ENV YOLO_MODEL_PATH=models/yolo_fire.pt
ENV YOLO_CONF_THRESHOLD=0.25
ENV YOLO_IOU_THRESHOLD=0.45
ENV USE_EMBEDDING_FUSION=false
ENV PORT=8100

# Expose port
EXPOSE 8100

# Run the application
CMD ["python", "app.py"]
