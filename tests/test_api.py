
from fastapi.testclient import TestClient
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


def test_prediction_endpoint():

    # Generate a valid RGB image in memory
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

    assert "predicted_class" in data
    assert "confidence" in data
    assert "top_3" in data
    assert "inference_time_ms" in data

    assert data["predicted_class"] in [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    assert 0 <= data["confidence"] <= 1
    assert len(data["top_3"]) == 3
