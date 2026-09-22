
from fastapi import FastAPI, UploadFile, File, HTTPException
from api.model_service import ModelService


app = FastAPI(
    title="CIFAR-10 Image Classification API",
    description="Image classification API powered by MobileNetV2",
    version="1.0.0"
)

# Load model once when application starts
model_service = ModelService()

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/avif"
}


@app.get("/")
def root():
    return {
        "message": "CIFAR-10 Image Classification API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "MobileNetV2",
        "classes": 10
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG, and AVIF images are supported."
        )

    try:
        image_bytes = await file.read()

        result = model_service.predict(image_bytes)

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
