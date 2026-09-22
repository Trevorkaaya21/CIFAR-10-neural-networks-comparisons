
from fastapi.testclient import TestClient
from unittest.mock import patch
from PIL import Image
import io

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "MobileNetV2"
    assert data["classes"] == 10


def test_invalid_file_type():
    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"This is not an image.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


@patch("api.main.model_service.predict")
def test_prediction_endpoint(mock_predict):

    mock_predict.return_value = {
        "predicted_class": "cat",
        "confidence": 0.87,
        "top_3": [
            {"class": "cat", "confidence": 0.87},
            {"class": "dog", "confidence": 0.08},
            {"class": "frog", "confidence": 0.05}
        ],
        "inference_time_ms": 90.0
    }

    image = Image.new(
        "RGB",
        (96, 96),
        color=(120, 150, 200)
    )

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image_bytes,
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"] == "cat"
    assert data["confidence"] == 0.87
    assert len(data["top_3"]) == 3
    assert data["inference_time_ms"] == 90.0

    mock_predict.assert_called_once()
