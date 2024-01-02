FROM python:3.8-slim
WORKDIR /app
COPY pytorch_inference_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pytorch_inference_service/app /app
EXPOSE 5000
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
