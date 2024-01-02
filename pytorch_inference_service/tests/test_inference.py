import pytest
from app import create_app
import requests
import os

CORE_LOGIC_SERVICE_URL = "http://core_logic_service:5000"


@pytest.fixture(scope="session")
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
        # Additional configuration for testing
    })
    return app


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def preload_model():
    # Assuming Core Logic Service is running and accessible
    model_load_url = f"{CORE_LOGIC_SERVICE_URL}/load_model"
    response = requests.post(model_load_url, json={"model_type": "resnet50"})
    assert response.status_code == 200


@pytest.fixture(scope="session")
def process_test_image():
    image_process_url = f"{CORE_LOGIC_SERVICE_URL}/process_image"
    test_image_path = os.path.join(os.path.dirname(__file__), 'images', 'test_image.jpg')
    with open(test_image_path, 'rb') as img:
        files = {'image': img}
        response = requests.post(image_process_url, files=files)
    assert response.status_code == 200
    return response.json()['image_path']


def test_predict(client, preload_model, process_test_image):
    image_path = process_test_image
    response = client.post('/predict', json={"image_path": image_path})
    assert response.status_code == 200
    # Add more assertions based on the expected prediction output
    # For example:
    # prediction_result = response.json()
    # assert "predicted_label" in prediction_result


def test_benchmark(client, preload_model, process_test_image):
    image_path = process_test_image
    response = client.post('/benchmark', json={"image_path": image_path, "num_runs": 10})
    assert response.status_code == 200
    # Add more assertions based on the expected benchmark output
    # For example:
    # benchmark_result = response.json()
    # assert "average_time" in benchmark_result

