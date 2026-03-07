#!/usr/bin/env python3
"""
AI Face Detector - Cloud-Ready Training Script

This script trains a MobileNetV2 model to detect AI-generated faces using Transfer Learning.
Designed for Google Colab and Kaggle Notebooks with GPU support.

Dataset: GRAVEX-200K (200K images: 100K real, 100K AI-generated)
Model: MobileNetV2 (pre-trained on ImageNet)
Task: Binary classification (Real vs AI-Generated)

Usage in Colab:
    1. Upload this script to Colab
    2. Upload/download dataset to content/
    3. Run: !python train_model.py --data_path /content/data --epochs 15

Usage in Kaggle:
    1. Add GRAVEX-200K dataset to notebook
    2. Run: python train_model.py --data_path /kaggle/input --epochs 15
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# ================================
# Configuration
# ================================


class Config:
    """Training configuration."""

    # Data paths
    DATA_PATH = Path(
        "/content/data"
    )  # Default for Colab, change to /kaggle/input for Kaggle
    AI_IMAGES_DIR = "my_real_vs_ai_dataset/my_real_vs_ai_dataset/ai_images"
    REAL_IMAGES_DIR = "my_real_vs_ai_dataset/my_real_vs_ai_dataset/real_images"

    # Model settings
    MODEL_NAME = "mobilenet_v2"
    NUM_CLASSES = 1  # Binary classification
    DROPOUT_RATE = 0.3

    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 5

    # Data augmentation
    IMAGE_SIZE = 224  # MobileNetV2 input size
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output files
    MODEL_SAVE_PATH = "model.pth"
    HISTORY_SAVE_PATH = "training_history.json"
    REPORT_SAVE_PATH = "evaluation_report.json"

    # ImageNet normalization (for pre-trained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


# ================================
# Dataset Class
# ================================


class FaceDataset(Dataset):
    """Custom dataset for face images."""

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (0=AI, 1=Real)
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and transform an image."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new(
                "RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), color="white"
            )

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


# ================================
# Data Loading and Preparation
# ================================


def get_data_transforms() -> Dict[str, transforms.Compose]:
    """Get training and validation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
        ]
    )

    return {"train": train_transform, "val": val_transform, "test": val_transform}


def load_dataset(data_path: Path) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and split the dataset into train, validation, and test sets.

    Args:
        data_path: Path to the dataset directory

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("=" * 70)
    print("📁 LOADING DATASET")
    print("=" * 70)

    # Paths to AI and real images
    ai_images_path = data_path / Config.AI_IMAGES_DIR
    real_images_path = data_path / Config.REAL_IMAGES_DIR

    # Check if paths exist
    if not ai_images_path.exists():
        raise FileNotFoundError(f"AI images directory not found: {ai_images_path}")
    if not real_images_path.exists():
        raise FileNotFoundError(f"Real images directory not found: {real_images_path}")

    # Get all image paths
    ai_images = list(ai_images_path.glob("*.jpg"))
    real_images = list(real_images_path.glob("*.jpg"))

    print(f"\n✓ Found {len(ai_images):,} AI-generated images")
    print(f"✓ Found {len(real_images):,} real images")

    # Create labels (0=AI, 1=Real)
    ai_labels = [0] * len(ai_images)
    real_labels = [1] * len(real_images)

    # Combine
    all_images = ai_images + real_images
    all_labels = ai_labels + real_labels

    print(f"\n✓ Total images: {len(all_images):,}")

    # Get transforms
    transforms_dict = get_data_transforms()

    # Create full dataset
    full_dataset = FaceDataset(
        all_images, all_labels, transform=transforms_dict["train"]
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * Config.TRAIN_SPLIT)
    val_size = int(total_size * Config.VAL_SPLIT)
    test_size = total_size - train_size - val_size

    print("\n✓ Split sizes:")
    print(f"  - Train: {train_size:,} ({Config.TRAIN_SPLIT * 100:.0f}%)")
    print(f"  - Val: {val_size:,} ({Config.VAL_SPLIT * 100:.0f}%)")
    print(f"  - Test: {test_size:,} ({Config.TEST_SPLIT * 100:.0f}%)")

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Update transforms for val/test
    val_dataset.dataset.transform = transforms_dict["val"]
    test_dataset.dataset.transform = transforms_dict["test"]

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if Config.DEVICE.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if Config.DEVICE.type == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if Config.DEVICE.type == "cuda" else False,
    )

    return train_loader, val_loader, test_loader


# ================================
# Model Creation
# ================================


def create_model() -> nn.Module:
    """
    Create MobileNetV2 model for binary classification.

    Returns:
        Modified MobileNetV2 model
    """
    print("=" * 70)
    print("🧠 CREATING MODEL")
    print("=" * 70)

    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)

    # Freeze feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the number of features in the classifier
    num_features = model.classifier[1].in_features

    # Replace classifier with custom binary classification head
    model.classifier = nn.Sequential(
        nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(num_features, Config.NUM_CLASSES)
    )

    print(f"\n✓ Model: {Config.MODEL_NAME}")
    print("✓ Pre-trained on ImageNet")
    print("✓ Feature extractor frozen")
    print(f"✓ Custom classifier: Linear({num_features} -> {Config.NUM_CLASSES})")
    print(f"✓ Dropout rate: {Config.DROPOUT_RATE}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    return model


# ================================
# Training Functions
# ================================


def train_epoch(
    model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module, dataloader: DataLoader, criterion, device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    Train the model with early stopping.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on

    Returns:
        Dictionary with training history
    """
    print("=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    print(f"\n✓ Device: {device}")
    print(f"✓ Epochs: {Config.NUM_EPOCHS}")
    print(f"✓ Batch size: {Config.BATCH_SIZE}")
    print(f"✓ Learning rate: {Config.LEARNING_RATE}")
    print(f"✓ Early stopping patience: {Config.EARLY_STOPPING_PATIENCE}")

    # Move model to device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
    }

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)

        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"✓ Model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")

        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
            break

    training_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n✓ Training time: {training_time / 60:.2f} minutes")
    print(f"✓ Best val loss: {best_val_loss:.4f}")
    print(f"✓ Model saved to: {Config.MODEL_SAVE_PATH}")

    return history


# ================================
# Evaluation
# ================================


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Dict:
    """
    Evaluate the model on the test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 70)
    print("📊 EVALUATING ON TEST SET")
    print("=" * 70)

    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Convert to numpy arrays
    import numpy as np

    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities).flatten()

    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probabilities)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()

    # Print results
    print(f"\n✓ Test Accuracy: {accuracy * 100:.2f}%")
    print(f"✓ Precision: {precision * 100:.2f}%")
    print(f"✓ Recall: {recall * 100:.2f}%")
    print(f"✓ F1 Score: {f1:.4f}")
    print(f"✓ AUC-ROC: {auc:.4f}")

    print("\nConfusion Matrix:")
    print(f"  True Negatives: {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives: {tp:,}")

    # Create report
    report = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    return report


# ================================
# Visualization
# ================================


def plot_training_history(
    history: Dict, save_path: str = "training_curves.png"
) -> None:
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(history["val_acc"], label="Val Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    # Learning Rate
    axes[2].plot(
        history["learning_rate"], label="Learning Rate", marker="o", color="green"
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Training curves saved to: {save_path}")

    # For Colab: display the plot
    try:
        from google.colab import files

        files.download(save_path)
    except:
        pass


# ================================
# Main Function
# ================================


def main():
    """Main training function."""
    print("\n" + "=" * 70)
    print("🤖 AI FACE DETECTOR - MODEL TRAINING")
    print("=" * 70)

    # Check device
    print(f"\n🔧 Device: {Config.DEVICE}")
    if Config.DEVICE.type == "cuda":
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("⚠️  No GPU detected, using CPU (training will be slow)")

    # Load data
    train_loader, val_loader, test_loader = load_dataset(Config.DATA_PATH)

    # Create model
    model = create_model()

    # Train model
    history = train_model(model, train_loader, val_loader, Config.DEVICE)

    # Load best model for evaluation
    print("\n" + "=" * 70)
    print("📈 LOADING BEST MODEL FOR EVALUATION")
    print("=" * 70)

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model = model.to(Config.DEVICE)

    # Evaluate on test set
    report = evaluate_model(model, test_loader, Config.DEVICE)

    # Save training history
    with open(Config.HISTORY_SAVE_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved to: {Config.HISTORY_SAVE_PATH}")

    # Save evaluation report
    with open(Config.REPORT_SAVE_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Evaluation report saved to: {Config.REPORT_SAVE_PATH}")

    # Plot training curves
    plot_training_history(history)

    # Print summary
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📁 Output files:")
    print(f"  1. {Config.MODEL_SAVE_PATH} - Trained model weights")
    print(f"  2. {Config.HISTORY_SAVE_PATH} - Training history")
    print(f"  3. {Config.REPORT_SAVE_PATH} - Evaluation report")
    print("  4. training_curves.png - Training curves visualization")

    print("\n📋 Model Performance:")
    print(f"  - Test Accuracy: {report['accuracy'] * 100:.2f}%")
    print(f"  - F1 Score: {report['f1_score']:.4f}")
    print(f"  - AUC-ROC: {report['auc_roc']:.4f}")

    # For Colab: download files
    try:
        from google.colab import files

        print("\n📥 Downloading files...")
        files.download(Config.MODEL_SAVE_PATH)
        files.download(Config.HISTORY_SAVE_PATH)
        files.download(Config.REPORT_SAVE_PATH)
        print("✓ Files ready for download")
    except:
        print("\n💡 For Colab: Use the file browser to download model.pth")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
