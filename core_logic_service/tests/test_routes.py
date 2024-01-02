import pytest
from ..app import create_app
from io import BytesIO
import shutil
import os


@pytest.fixture(scope="session")
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
        "REDIS_HOST": "localhost",
        "REDIS_PORT": 6379,
        "TMP_DIR": "test_tmp"
    })
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup the testing directory once we are finished."""
    def remove_test_dir():
        test_dir = os.path.join(request.config.rootdir, app.config['TMP_DIR'])
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
    request.addfinalizer(remove_test_dir)


def test_load_model(client):
    response = client.post('/load_model', json={"model_type": "resnet50", "device": "cpu"})
    assert response.status_code == 200
    assert "Model resnet50 loaded successfully on cpu" in response.get_data(as_text=True)


def test_process_image(client):
    with open('files/cat3.jpg', 'rb') as img:
        img_data = {'image': (BytesIO(img.read()), 'image.jpg')}
        response = client.post('/process_image', content_type='multipart/form-data', data=img_data)
    assert response.status_code == 200
    assert "Image processed successfully" in response.get_data(as_text=True)