# Use NVIDIA PyTorch base image for GPU support
FROM nvcr.io/nvidia/pytorch:23.11-py3

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
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DINO_MODEL_ID=facebook/dinov2-base
ENV USE_MINIO=true
ENV MINIO_ENDPOINT=http://localhost:9000
ENV EMBED_BUCKET=embeds

# Expose port for service
EXPOSE 8000

# Run the embedding service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]