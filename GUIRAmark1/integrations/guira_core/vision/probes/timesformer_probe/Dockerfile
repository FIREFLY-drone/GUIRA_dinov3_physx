FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create models directory
RUN mkdir -p models

# Environment variables
ENV TIMESFORMER_MODEL_PATH=models/timesformer_smoke.pt
ENV SEQUENCE_LENGTH=16
ENV FRAME_SIZE=224
ENV SMOKE_CONF_THRESHOLD=0.5
ENV USE_EMBEDDING_FUSION=false
ENV PORT=8101

# Expose port
EXPOSE 8101

# Run the application
CMD ["python", "app.py"]
