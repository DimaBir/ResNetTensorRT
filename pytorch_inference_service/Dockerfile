# Dockerfile for pytorch_inference_service
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
