# Optimized production Dockerfile
FROM python:3.9.18-slim-bullseye

# Set up environment
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install only essential dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Upgrade pip and install Python packages (using legacy resolver)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip cache purge && \
    pip install \
    --no-cache-dir \
    --use-deprecated=legacy-resolver \
    --default-timeout=300 \
    --retries 10 \
    -r requirements.txt

# Copy application files
COPY app.py . 
COPY engagement_model_89.tflite .

# Set up non-root user
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Configure health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:7860/ || exit 1

# Expose port and run the application (using 7860 for Hugging Face Spaces)
EXPOSE 7860
CMD ["python", "app.py", "--port", "7860"]
