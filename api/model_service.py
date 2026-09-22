
import io
import json
import os
import time

import boto3
import numpy as np
from PIL import Image, ImageOps


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

ENDPOINT_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME",
    "cifar10-serverless-endpoint"
)

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-west-2"
)


class ModelService:
    def __init__(self, runtime=None):
        self.runtime = runtime

    def _get_runtime(self):
        if self.runtime is None:
            self.runtime = boto3.client(
                "sagemaker-runtime",
                region_name=AWS_REGION
            )
        return self.runtime

    def preprocess_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image = ImageOps.fit(
            image,
            (96, 96),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5)
        )

        image_array = np.array(image).astype("float32") / 255.0

        return np.expand_dims(image_array, axis=0)

    def predict(self, image_bytes):
        image_batch = self.preprocess_image(image_bytes)

        payload = json.dumps({
            "instances": image_batch.tolist()
        })

        start_time = time.perf_counter()

        response = self._get_runtime().invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        result = json.loads(
            response["Body"].read().decode("utf-8")
        )

        probabilities = np.array(result["predictions"][0])

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
            "confidence": round(float(probabilities[top_indices[0]]), 4),
            "top_3": top_predictions,
            "inference_time_ms": round(inference_time_ms, 2)
        }
