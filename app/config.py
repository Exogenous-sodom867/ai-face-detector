"""
Configuration settings for AI Face Detector.
"""

import torch
from pathlib import Path

# Model settings
MODEL_PATH = Path("model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224  # MobileNetV2 input size

# Class labels (0=AI, 1=REAL)
CLASS_NAMES = ["AI_GENERATED", "REAL"]

# ImageNet normalization (for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# API settings
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# CORS settings (allow frontend)
CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]
