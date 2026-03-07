"""
FastAPI application for AI Face Detector.

This module provides the REST API for detecting AI-generated faces.
"""

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from app.config import DEVICE, CORS_ORIGINS
from app.model_loader import load_model
from app.utils import transform_image, predict, validate_image_file


# Global model variable
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.

    Loads the model on startup.
    """
    global model

    # Startup
    print("=" * 70)
    print("🚀 AI FACE DETECTOR - STARTING API")
    print("=" * 70)

    try:
        model = load_model()
        print("✓ Model loaded successfully")
        print(f"✓ Device: {DEVICE}")
        print("✓ API ready to accept requests")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("⚠️  API will start but /detect endpoint will not work")

    yield

    # Shutdown
    print("\n👋 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="AI Face Detector",
    description="Detect AI-generated faces with high accuracy using Transfer Learning",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "status": "success",
        "message": "AI Face Detector API",
        "version": "0.1.0",
        "endpoints": {
            "POST /detect": "Upload an image to detect if it's AI-generated or real",
            "GET /health": "Check API health status",
            "GET /": "API information",
        },
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if model is not None else "not loaded"

    return {
        "status": "healthy",
        "model_status": model_status,
        "device": str(DEVICE),
        "version": "0.1.0",
    }


@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    """
    Detect if a face image is AI-generated or real.

    Args:
        file: Uploaded image file (max 5MB, JPG/PNG/WEBP)

    Returns:
        JSON response with detection result:
        {
            "status": "success",
            "result": "AI_GENERATED" | "REAL",
            "confidence": 0.98,
            "probabilities": {
                "AI_GENERATED": 0.02,
                "REAL": 0.98
            },
            "inference_time_ms": 45.2,
            "image_size": "512x512"
        }

    Raises:
        HTTPException: If file is invalid or model is not loaded
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs.",
        )

    # Read file content
    file_content = await file.read()

    # Validate file
    is_valid, error_msg = validate_image_file(file.filename, len(file_content))
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

    try:
        # Get image dimensions
        image = Image.open(io.BytesIO(file_content))
        width, height = image.size
        image_size = f"{width}x{height}"

        # Transform image
        image_tensor = transform_image(file_content)

        # Run inference
        prediction = predict(model, image_tensor, DEVICE)

        # Prepare response
        response = {
            "status": "success",
            "result": prediction["result"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
            "inference_time_ms": prediction["inference_time_ms"],
            "image_size": image_size,
        }

        return response

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during inference: {str(e)}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "detail": "An unexpected error occurred. Please try again.",
        },
    )


if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("🚀 AI FACE DETECTOR - STARTING SERVER")
    print("=" * 70)
    print("\n📡 Server will be available at:")
    print("  - http://localhost:8000")
    print("  - http://localhost:8000/docs (API documentation)")
    print("  - http://localhost:8000/health (health check)")
    print("\n" + "=" * 70)

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
