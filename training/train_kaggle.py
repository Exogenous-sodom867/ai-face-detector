#!/usr/bin/env python3
"""
AI Face Detector - Kaggle Training Script

This script is specifically designed for Kaggle Notebooks with the
"140k real and fake faces" dataset by xhlulu.

Dataset: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
Model: MobileNetV2 (pre-trained on ImageNet)
Task: Binary classification (Real vs AI-Generated)

Usage in Kaggle:
    1. Add dataset "140k real and fake faces" to your notebook
    2. Run this script
    3. Download model.pth when training completes
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# ================================
# Configuration
# ================================


class Config:
    """Training configuration for Kaggle."""

    # Dataset paths (will be auto-detected)
    DATA_PATH = Path("/kaggle/input")

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

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output files
    MODEL_SAVE_PATH = "/kaggle/working/model.pth"
    HISTORY_SAVE_PATH = "/kaggle/working/training_history.json"
    REPORT_SAVE_PATH = "/kaggle/working/evaluation_report.json"
    PLOT_SAVE_PATH = "/kaggle/working/training_curves.png"

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


# ================================
# Dataset Detection
# ================================


def find_dataset_path() -> Path:
    """
    Automatically find the dataset path in Kaggle.

    Returns:
        Path to the dataset directory
    """
    print("=" * 70)
    print("🔍 DETECTING DATASET")
    print("=" * 70)

    base_path = Config.DATA_PATH

    # Recursively search for the dataset
    def find_real_vs_fake_recursive(current_path: Path, depth: int = 0) -> Path:
        """Recursively search for real_vs_fake directory."""
        if depth > 5:  # Limit recursion depth
            return None

        try:
            contents = os.listdir(current_path)
        except:
            return None

        # Check if this directory contains train/real and train/fake
        train_real = current_path / "train" / "real"
        train_fake = current_path / "train" / "fake"

        if train_real.exists() and train_fake.exists():
            return current_path

        # Recursively search subdirectories
        for item in contents:
            item_path = current_path / item
            if item_path.is_dir() and not item.startswith("."):
                result = find_real_vs_fake_recursive(item_path, depth + 1)
                if result:
                    return result

        return None

    # Start recursive search
    print(f"\n🔎 Searching for dataset in {base_path}...")
    dataset_path = find_real_vs_fake_recursive(base_path)

    if dataset_path:
        print(f"\n✅ Found dataset at: {dataset_path}")

        # Verify structure
        train_real = dataset_path / "train" / "real"
        train_fake = dataset_path / "train" / "fake"

        real_count = len(list(train_real.glob("*.jpg")))
        fake_count = len(list(train_fake.glob("*.jpg")))

        print("✅ Dataset structure verified!")
        print(f"   - Train real: {real_count:,} images")
        print(f"   - Train fake: {fake_count:,} images")

        return dataset_path

    # If not found, raise error with guidance
    raise ValueError(
        "❌ Dataset bulunamadı!\n\n"
        "Lütfen şunları kontrol et:\n"
        "1. Dataset eklenmiş mi?: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces\n"
        "2. Notebook'un sağında 'Input' bölümünde dataset görünüyor mu?\n\n"
        "Beklenen yapı: real_vs_fake/train/real/ ve real_vs_fake/train/fake/"
    )


# ================================
# Dataset Class
# ================================


class FaceDataset(Dataset):
    """Custom dataset for face images."""

    def __init__(
        self,
        image_paths: list,
        labels: list,
        transform: transforms.Compose = None,
    ):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (0=AI/Fake, 1=Real)
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        """Load and transform an image."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {e}")
            # Return a blank image
            image = Image.new(
                "RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), color="white"
            )

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


# ================================
# Data Loading
# ================================


def load_data_splits(dataset_path: Path) -> tuple:
    """
    Load train, validation, and test splits from the dataset.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    print("\n" + "=" * 70)
    print("📁 LOADING DATA SPLITS")
    print("=" * 70)

    # Define paths
    train_real_path = dataset_path / "train" / "real"
    train_fake_path = dataset_path / "train" / "fake"
    val_real_path = dataset_path / "valid" / "real"
    val_fake_path = dataset_path / "valid" / "fake"
    test_real_path = dataset_path / "test" / "real"
    test_fake_path = dataset_path / "test" / "fake"

    # Verify paths exist
    for path in [
        train_real_path,
        train_fake_path,
        val_real_path,
        val_fake_path,
        test_real_path,
        test_fake_path,
    ]:
        if not path.exists():
            raise ValueError(f"❌ Path not found: {path}")

    # Get image paths
    train_real = list(train_real_path.glob("*.jpg"))
    train_fake = list(train_fake_path.glob("*.jpg"))
    val_real = list(val_real_path.glob("*.jpg"))
    val_fake = list(val_fake_path.glob("*.jpg"))
    test_real = list(test_real_path.glob("*.jpg"))
    test_fake = list(test_fake_path.glob("*.jpg"))

    print("\n✅ Dataset splits:")
    print(
        f"  Train: {len(train_real):,} real + {len(train_fake):,} fake = {len(train_real) + len(train_fake):,}"
    )
    print(
        f"  Val:   {len(val_real):,} real + {len(val_fake):,} fake = {len(val_real) + len(val_fake):,}"
    )
    print(
        f"  Test:  {len(test_real):,} real + {len(test_fake):,} fake = {len(test_real) + len(test_fake):,}"
    )

    # Create labels (0=Fake/AI, 1=Real)
    train_images = train_real + train_fake
    train_labels = [1] * len(train_real) + [0] * len(train_fake)

    val_images = val_real + val_fake
    val_labels = [1] * len(val_real) + [0] * len(val_fake)

    test_images = test_real + test_fake
    test_labels = [1] * len(test_real) + [0] * len(test_fake)

    # Create transforms
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

    # Create datasets
    train_dataset = FaceDataset(train_images, train_labels, train_transform)
    val_dataset = FaceDataset(val_images, val_labels, val_transform)
    test_dataset = FaceDataset(test_images, test_labels, val_transform)

    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset
) -> tuple:
    """Create data loaders for train, val, and test sets."""
    print("\n✅ Creating data loaders...")

    # IMPORTANT: num_workers=0 for Kaggle compatibility
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Kaggle doesn't support multiprocessing
        pin_memory=True if Config.DEVICE.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if Config.DEVICE.type == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if Config.DEVICE.type == "cuda" else False,
    )

    print(f"✅ Data loaders created (batch_size={Config.BATCH_SIZE})")

    return train_loader, val_loader, test_loader


# ================================
# Model Creation
# ================================


def create_model() -> nn.Module:
    """Create MobileNetV2 model for binary classification."""
    print("\n" + "=" * 70)
    print("🧠 CREATING MODEL")
    print("=" * 70)

    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Get number of features
    num_features = model.classifier[1].in_features

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(num_features, Config.NUM_CLASSES)
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✅ Model: {Config.MODEL_NAME}")
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")

    return model


# ================================
# Training Functions
# ================================


def train_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.1f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.1f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, device) -> Dict:
    """Train the model with early stopping."""
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    print(f"\n✅ Device: {device}")
    print(f"✅ Epochs: {Config.NUM_EPOCHS}")
    print(f"✅ Batch size: {Config.BATCH_SIZE}")
    print(f"✅ Learning rate: {Config.LEARNING_RATE}")

    # Move model to device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    # History
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"{'=' * 70}")

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
        history["lr"].append(current_lr)

        # Print summary
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"✅ Model saved! (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(
                f"⏳ No improvement ({patience_counter}/{Config.EARLY_STOPPING_PATIENCE})"
            )

        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("\n⚠️  Early stopping triggered!")
            break

    training_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n⏱️  Training time: {training_time / 60:.2f} minutes")
    print(f"🏆 Best val loss: {best_val_loss:.4f}")

    return history


# ================================
# Evaluation
# ================================


def evaluate_model(model, test_loader, device) -> Dict:
    """Evaluate model on test set."""
    print("\n" + "=" * 70)
    print("📊 EVALUATING ON TEST SET")
    print("=" * 70)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().flatten())

    # Calculate metrics
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n✅ Test Accuracy:  {accuracy * 100:.2f}%")
    print(f"✅ Precision:      {precision * 100:.2f}%")
    print(f"✅ Recall:         {recall * 100:.2f}%")
    print(f"✅ F1 Score:       {f1:.4f}")
    print(f"✅ AUC-ROC:        {auc:.4f}")

    print("\nConfusion Matrix:")
    print(f"  TN: {tn:6,} | FP: {fp:6,}")
    print(f"  FN: {fn:6,} | TP: {tp:6,}")

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


def plot_history(history: Dict):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train", marker="o")
    axes[0].plot(history["val_loss"], label="Val", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train", marker="o")
    axes[1].plot(history["val_acc"], label="Val", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    # Learning Rate
    axes[2].plot(history["lr"], marker="o", color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate")
    axes[2].grid(True)
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(Config.PLOT_SAVE_PATH, dpi=150, bbox_inches="tight")
    print(f"\n✅ Plot saved to: {Config.PLOT_SAVE_PATH}")


# ================================
# Main
# ================================


def main():
    """Main training function."""
    print("\n" + "=" * 70)
    print("🤖 AI FACE DETECTOR - KAGGLE TRAINING")
    print("=" * 70)

    # Check device
    print(f"\n🔧 Device: {Config.DEVICE}")
    if Config.DEVICE.type == "cuda":
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ Memory: {mem:.2f} GB")
    else:
        print("⚠️  No GPU - using CPU (will be slow)")

    # Find dataset
    dataset_path = find_dataset_path()

    # Load data
    train_ds, val_ds, test_ds = load_data_splits(dataset_path)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, val_ds, test_ds
    )

    # Create model
    model = create_model()

    # Train
    history = train_model(model, train_loader, val_loader, Config.DEVICE)

    # Load best model
    print("\n" + "=" * 70)
    print("📈 LOADING BEST MODEL")
    print("=" * 70)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    # Evaluate
    report = evaluate_model(model, test_loader, Config.DEVICE)

    # Save files
    with open(Config.HISTORY_SAVE_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ History saved: {Config.HISTORY_SAVE_PATH}")

    with open(Config.REPORT_SAVE_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report saved: {Config.REPORT_SAVE_PATH}")

    # Plot
    plot_history(history)

    # Summary
    print("\n" + "=" * 70)
    print("🎉 COMPLETE!")
    print("=" * 70)
    print("\n📁 Output files (in /kaggle/working/):")
    print("  1. model.pth - Trained model (DOWNLOAD THIS!)")
    print("  2. training_history.json")
    print("  3. evaluation_report.json")
    print("  4. training_curves.png")

    print("\n📊 Final Results:")
    print(f"  Accuracy:  {report['accuracy'] * 100:.2f}%")
    print(f"  F1 Score:  {report['f1_score']:.4f}")
    print(f"  AUC-ROC:   {report['auc_roc']:.4f}")

    print("\n" + "=" * 70)
    print("📥 NEXT STEPS:")
    print("=" * 70)
    print("1. Open the file browser (right side)")
    print("2. Go to /kaggle/working/")
    print("3. Download model.pth")
    print("4. Place model.pth in your project root")
    print("5. Run: uvicorn app.main:app --reload")
    print("=" * 70)


if __name__ == "__main__":
    main()
