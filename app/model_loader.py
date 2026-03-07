"""
Model loading utilities for AI Face Detector.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from typing import Optional

from app.config import DEVICE, MODEL_PATH


def create_model() -> nn.Module:
    """
    Create MobileNetV2 model for binary classification.

    This recreates the exact model architecture used during training.

    Returns:
        MobileNetV2 model with custom classifier head
    """
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)

    # Replace classifier with binary classification head
    # Note: We don't freeze layers here since we're loading trained weights
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 1),  # Binary classification
    )

    return model


def load_model(model_path: Optional[Path] = None) -> nn.Module:
    """
    Load trained model weights.

    Args:
        model_path: Path to model weights file. Defaults to config.MODEL_PATH

    Returns:
        Model with loaded weights in evaluation mode

    Raises:
        FileNotFoundError: If model weights file doesn't exist
        RuntimeError: If weights file is corrupted
    """
    if model_path is None:
        model_path = MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            f"Please train the model first or download the pretrained weights."
        )

    # Create model architecture
    model = create_model()

    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"✓ Model loaded successfully from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

    # Set to evaluation mode and move to device
    model.eval()
    model = model.to(DEVICE)

    print(f"✓ Model running on {DEVICE}")

    return model
