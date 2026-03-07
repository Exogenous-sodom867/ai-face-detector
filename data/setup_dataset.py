#!/usr/bin/env python3
"""
Dataset Setup Script for AI Face Detector

This script downloads the GRAVEX-200K dataset from Kaggle using MCP integration.
The dataset contains 200K images (100K real, 100K AI-generated faces).

Dataset: muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal
Size: ~2GB (200,000 images, 256x256 resolution)
License: CC0 Public Domain
"""

import sys
from pathlib import Path
from typing import Dict
import csv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_directory_structure(base_path: Path = Path("data")) -> None:
    """Create the necessary directory structure for the dataset."""
    dirs = [
        base_path / "raw",
        base_path / "processed" / "train" / "ai_generated",
        base_path / "processed" / "train" / "real",
        base_path / "processed" / "val" / "ai_generated",
        base_path / "processed" / "val" / "real",
        base_path / "processed" / "test" / "ai_generated",
        base_path / "processed" / "test" / "real",
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def download_dataset_via_kaggle_mcp() -> Dict[str, any]:
    """
    Download dataset using Kaggle MCP.

    This function provides instructions for downloading the dataset via Kaggle MCP.
    Since MCP tools are called externally, this script provides the necessary information.

    Returns:
        Dict with dataset information
    """
    dataset_info = {
        "owner_slug": "muhammadbilal6305",
        "dataset_slug": "200k-real-vs-ai-visuals-by-mbilal",
        "title": "GRAVEX-200K: 200K Real vs AI-Generated Faces",
        "size_gb": 1.9,
        "num_images": 200000,
        "resolution": "256x256",
        "split": "70% train, 20% val, 10% test",
        "license": "CC0 Public Domain",
    }

    print("\n" + "=" * 70)
    print("📥 DATASET DOWNLOAD REQUIRED")
    print("=" * 70)
    print("\nTo download the GRAVEX-200K dataset, use Kaggle MCP:")
    print(f"\n  Dataset: {dataset_info['owner_slug']}/{dataset_info['dataset_slug']}")
    print(f"\n  Title: {dataset_info['title']}")
    print(f"  Size: {dataset_info['size_gb']} GB")
    print(f"  Images: {dataset_info['num_images']:,}")
    print(f"  Resolution: {dataset_info['resolution']}")
    print(f"  License: {dataset_info['license']}")

    print("\n" + "-" * 70)
    print("MCP COMMAND TO DOWNLOAD:")
    print("-" * 70)
    print("\nUse: mcp__kaggle__download_dataset")
    print(f"  ownerSlug: {dataset_info['owner_slug']}")
    print(f"  datasetSlug: {dataset_info['dataset_slug']}")

    print("\n" + "-" * 70)
    print("ALTERNATIVE: Kaggle CLI")
    print("-" * 70)
    print(
        f"\n  kaggle datasets download -d {dataset_info['owner_slug']}/{dataset_info['dataset_slug']}"
    )

    print("\n" + "-" * 70)
    print("EXPECTED DATASET STRUCTURE:")
    print("-" * 70)
    print("""
    my_real_vs_ai_dataset/
    └── my_real_vs_ai_dataset/
        ├── ai_images/          (100,000 AI-generated faces)
        │   ├── 00276TOPP4.jpg
        │   ├── 002KDWZBHU.jpg
        │   └── ...
        └── real_images/        (100,000 real faces)
            ├── 00001.jpg
            ├── 00002.jpg
            └── ...
    """)

    return dataset_info


def validate_dataset(base_path: Path = Path("data/raw")) -> Dict[str, any]:
    """
    Validate the downloaded dataset.

    Args:
        base_path: Path to the raw dataset directory

    Returns:
        Dict with validation results and statistics
    """
    print("\n" + "=" * 70)
    print("🔍 VALIDATING DATASET")
    print("=" * 70)

    # Expected paths
    ai_images_path = (
        base_path / "my_real_vs_ai_dataset" / "my_real_vs_ai_dataset" / "ai_images"
    )
    real_images_path = (
        base_path / "my_real_vs_ai_dataset" / "my_real_vs_ai_dataset" / "real_images"
    )

    validation_results = {
        "dataset_found": False,
        "ai_images_count": 0,
        "real_images_count": 0,
        "total_images": 0,
        "ai_images_valid": 0,
        "real_images_valid": 0,
        "errors": [],
    }

    # Check if dataset directories exist
    if not ai_images_path.exists():
        validation_results["errors"].append(
            f"AI images directory not found: {ai_images_path}"
        )
        return validation_results

    if not real_images_path.exists():
        validation_results["errors"].append(
            f"Real images directory not found: {real_images_path}"
        )
        return validation_results

    validation_results["dataset_found"] = True

    # Count and validate images
    print("\nValidating AI-generated images...")
    ai_files = list(ai_images_path.glob("*.jpg"))
    validation_results["ai_images_count"] = len(ai_files)

    print(f"  Found {len(ai_files):,} AI-generated images")

    print("\nValidating real images...")
    real_files = list(real_images_path.glob("*.jpg"))
    validation_results["real_images_count"] = len(real_files)

    print(f"  Found {len(real_files):,} real images")

    validation_results["total_images"] = (
        validation_results["ai_images_count"] + validation_results["real_images_count"]
    )

    # Sample validation (check first 100 images from each class)
    from PIL import Image

    print("\nValidating image integrity (sample check)...")
    for i, img_path in enumerate(ai_files[:100]):
        try:
            with Image.open(img_path) as img:
                img.verify()
            validation_results["ai_images_valid"] += 1
        except Exception as e:
            validation_results["errors"].append(
                f"Invalid AI image: {img_path.name} - {str(e)}"
            )

    for i, img_path in enumerate(real_files[:100]):
        try:
            with Image.open(img_path) as img:
                img.verify()
            validation_results["real_images_valid"] += 1
        except Exception as e:
            validation_results["errors"].append(
                f"Invalid real image: {img_path.name} - {str(e)}"
            )

    print(f"  ✓ AI images valid: {validation_results['ai_images_valid']}/100 sampled")
    print(
        f"  ✓ Real images valid: {validation_results['real_images_valid']}/100 sampled"
    )

    return validation_results


def generate_statistics_report(
    validation_results: Dict[str, any], output_path: Path = Path("data/stats.csv")
) -> None:
    """
    Generate a statistics CSV report from validation results.

    Args:
        validation_results: Dictionary with validation results
        output_path: Path to save the statistics CSV
    """
    print("\n" + "=" * 70)
    print("📊 GENERATING STATISTICS REPORT")
    print("=" * 70)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(["Metric", "Value"])

        # Dataset overview
        writer.writerow(["Dataset", "GRAVEX-200K"])
        writer.writerow(["Total Images", validation_results.get("total_images", 0)])
        writer.writerow(
            ["AI-Generated Images", validation_results.get("ai_images_count", 0)]
        )
        writer.writerow(["Real Images", validation_results.get("real_images_count", 0)])
        writer.writerow(
            [
                "Balance",
                "Balanced"
                if validation_results.get("ai_images_count")
                == validation_results.get("real_images_count")
                else "Imbalanced",
            ]
        )

        # Validation results
        writer.writerow(
            [
                "Dataset Found",
                "Yes" if validation_results.get("dataset_found") else "No",
            ]
        )
        writer.writerow(
            ["Sample Valid (AI)", f"{validation_results.get('ai_images_valid', 0)}/100"]
        )
        writer.writerow(
            [
                "Sample Valid (Real)",
                f"{validation_results.get('real_images_valid', 0)}/100",
            ]
        )

        # Expected split (based on dataset documentation)
        writer.writerow(["Expected Train Split", "70% (140K images)"])
        writer.writerow(["Expected Val Split", "20% (40K images)"])
        writer.writerow(["Expected Test Split", "10% (20K images)"])

    print(f"\n✓ Statistics report saved to: {output_path}")


def print_summary(validation_results: Dict[str, any]) -> None:
    """Print a summary of the dataset setup process."""
    print("\n" + "=" * 70)
    print("✅ DATASET SETUP SUMMARY")
    print("=" * 70)

    if validation_results["dataset_found"]:
        print("\n✓ Dataset validated successfully!")
        print(f"\n  📁 Total Images: {validation_results['total_images']:,}")
        print(f"  🤖 AI-Generated: {validation_results['ai_images_count']:,}")
        print(f"  👤 Real Faces: {validation_results['real_images_count']:,}")
        print("\n  Next Steps:")
        print("    1. Use training/train_model.py to train the model")
        print("    2. Upload training script to Google Colab or Kaggle Notebooks")
        print("    3. Download trained model.pth and place in project root")
    else:
        print("\n⚠️  Dataset not found!")
        print("\n  Please download the dataset first:")
        print("    - Use Kaggle MCP (see instructions above)")
        print(
            "    - Or use Kaggle CLI: kaggle datasets download -d muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal"
        )
        print("    - Extract to: data/raw/")

    if validation_results.get("errors"):
        print(f"\n  ⚠️  Errors found: {len(validation_results['errors'])}")
        for error in validation_results["errors"][:5]:  # Show first 5 errors
            print(f"    - {error}")

    print("\n" + "=" * 70)


def main():
    """Main function to set up the dataset."""
    print("\n" + "=" * 70)
    print("🚀 AI FACE DETECTOR - DATASET SETUP")
    print("=" * 70)

    # Step 1: Create directory structure
    print("\n[Step 1/4] Creating directory structure...")
    create_directory_structure()

    # Step 2: Provide download instructions
    print("\n[Step 2/4] Dataset download information...")
    dataset_info = download_dataset_via_kaggle_mcp()

    # Step 3: Validate dataset (if it exists)
    print("\n[Step 3/4] Validating dataset...")
    validation_results = validate_dataset()

    # Step 4: Generate statistics report
    print("\n[Step 4/4] Generating statistics report...")
    generate_statistics_report(validation_results)

    # Print summary
    print_summary(validation_results)

    print("\n✅ Setup script complete!\n")


if __name__ == "__main__":
    main()
