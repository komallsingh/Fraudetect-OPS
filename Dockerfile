
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Application configuration
ENV PYTHONUNBUFFERED=1
ENV MODEL_SOURCE=mlflow
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# FastAPI port
EXPOSE 8000

# Start API server
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
