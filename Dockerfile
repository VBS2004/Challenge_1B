# Use Python 3.11.8 slim image as base
FROM python:3.11.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all files and folders in current directory to /app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data and output (if not already in COPY)
RUN mkdir -p /app/data/pdfs /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

ENTRYPOINT ["python", "1B.py"]
CMD ["--help"]

