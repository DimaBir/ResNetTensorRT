FROM python:3.8-slim
WORKDIR /app
COPY onnx_exporter_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY onnx_exporter_service/app /app
EXPOSE 5002
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5002"]
