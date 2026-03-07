"""
Utility functions for image preprocessing and inference.
"""

import io
import time
from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms

from app.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES


def get_transform() -> transforms.Compose:
    """
    Get image preprocessing transform.

    This matches the training preprocessing pipeline.

    Returns:
        Transform pipeline for inference
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def transform_image(image_bytes: bytes) -> torch.Tensor:
    """
    Transform raw image bytes to model input tensor.

    Args:
        image_bytes: Raw image data as bytes

    Returns:
        Preprocessed tensor ready for model inference

    Raises:
        ValueError: If image data is invalid
    """
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Apply transforms
        transform = get_transform()
        image_tensor = transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")


def predict(
    model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device
) -> Dict:
    """
    Run inference on an image tensor.

    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on

    Returns:
        Dictionary with prediction results:
        {
            "result": "AI_GENERATED" | "REAL",
            "confidence": float (0-1),
            "probabilities": {"AI_GENERATED": float, "REAL": float}
        }
    """
    start_time = time.time()

    # Move to device and run inference
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()

    # Calculate inference time
    inference_time_ms = (time.time() - start_time) * 1000

    # Convert to label
    # Note: 0=AI, 1=REAL (based on training setup)
    if probability < 0.5:
        result = CLASS_NAMES[0]  # AI_GENERATED
        confidence = 1 - probability  # Confidence for AI class
    else:
        result = CLASS_NAMES[1]  # REAL
        confidence = probability  # Confidence for REAL class

    return {
        "result": result,
        "confidence": round(confidence, 4),
        "probabilities": {
            "AI_GENERATED": round(1 - probability, 4),
            "REAL": round(probability, 4),
        },
        "inference_time_ms": round(inference_time_ms, 2),
    }


def validate_image_file(
    filename: str, file_size: int, max_size: int = 5 * 1024 * 1024
) -> Tuple[bool, str]:
    """
    Validate uploaded image file.

    Args:
        filename: Name of the uploaded file
        file_size: Size of the file in bytes
        max_size: Maximum allowed file size (default 5MB)

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        return False, "Invalid file type. Please upload JPG, PNG, or WEBP image."

    # Check file size
    if file_size > max_size:
        return False, f"File too large. Maximum size is {max_size // (1024 * 1024)}MB."

    # Check if file is not empty
    if file_size == 0:
        return False, "File is empty."

    return True, ""
