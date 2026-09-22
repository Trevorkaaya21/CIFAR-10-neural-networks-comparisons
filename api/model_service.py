
import io
import time
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


CLASS_NAMES = [
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

MODEL_PATH = "model/cifar10_mobilenetv2.keras"


class ModelService:

    def __init__(self):
        self.model = load_model(MODEL_PATH)

    def preprocess_image(self, image_bytes):
        """Convert an uploaded image into the format expected by the model."""

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Center crop and resize to MobileNetV2 input dimensions
        image = ImageOps.fit(
            image,
            (96, 96),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5)
        )

        # Convert pixels from 0-255 to 0-1
        image_array = np.array(image).astype("float32") / 255.0

        # Add batch dimension
        return np.expand_dims(image_array, axis=0)

    def predict(self, image_bytes):
        """Generate the top three CIFAR-10 predictions."""

        image_batch = self.preprocess_image(image_bytes)

        start_time = time.perf_counter()

        probabilities = self.model.predict(
            image_batch,
            verbose=0
        )[0]

        inference_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        top_indices = np.argsort(probabilities)[-3:][::-1]

        top_predictions = [
            {
                "class": CLASS_NAMES[index],
                "confidence": round(float(probabilities[index]), 4)
            }
            for index in top_indices
        ]

        return {
            "predicted_class": CLASS_NAMES[top_indices[0]],
            "confidence": round(
                float(probabilities[top_indices[0]]),
                4
            ),
            "top_3": top_predictions,
            "inference_time_ms": round(inference_time_ms, 2)
        }
