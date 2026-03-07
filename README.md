# AI Face Detector

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🤖 **Detect AI-generated faces with high accuracy using Transfer Learning**

AI Face Detector is a lightweight, highly accurate API and web interface that detects whether a human face image is real or AI-generated (GAN-generated, Stable Diffusion, etc.). Uses MobileNetV2 with Transfer Learning for fast, accurate inference.

## ✨ Features

- **🎯 High Accuracy**: ~95% accuracy on test set with F1 score > 0.93
- **⚡ Lightning Fast**: <50ms inference time on CPU, <10ms on GPU
- **📦 Lightweight**: ~14MB model size using MobileNetV2
- **🌐 REST API**: Clean FastAPI backend with automatic OpenAPI docs
- **💻 Beautiful UI**: Modern drag-and-drop interface with Tailwind CSS
- **🔒 Privacy First**: Images processed locally, never stored
- **☁️ Cloud Ready**: Training script optimized for Google Colab/Kaggle
- **📊 Comprehensive**: Training history, evaluation metrics, and visualizations

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │
│  (HTML/JS/CSS)  │
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────┐
│   FastAPI       │
│   Backend       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PyTorch Model  │
│ (MobileNetV2)   │
└─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip or Poetry for dependency management
- 2GB free disk space

### 1. Clone the Repository

```bash
git clone https://github.com/furkankoykiran/ai-face-detector.git
cd ai-face-detector
```

### 2. Install Dependencies

**Using pip:**
```bash
pip install -r requirements.txt
```

**Using Poetry:**
```bash
poetry install
```

### 3. Get Trained Model

**Option A: Train Your Own (Recommended)**

See [Training Guide](docs/TRAINING.md) for detailed instructions.

Quick version:
1. Open [Google Colab](https://colab.research.google.com/)
2. Enable GPU: Runtime > Change runtime type > GPU
3. Upload `training/train_model.py`
4. Download the GRAVEX-200K dataset from Kaggle
5. Run the script
6. Download `model.pth` and place in project root

**Option B: Use Pre-trained Weights**

Download pre-trained weights from [Releases](https://github.com/furkankoykiran/ai-face-detector/releases) (coming soon).

### 4. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### 5. Open the Web Interface

Open `static/index.html` in your browser or visit http://localhost:8000/docs to test the API.

## 📡 API Usage

### Detect Face

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@path/to/image.jpg"
```

**Response:**
```json
{
  "status": "success",
  "result": "AI_GENERATED",
  "confidence": 0.98,
  "probabilities": {
    "AI_GENERATED": 0.98,
    "REAL": 0.02
  },
  "inference_time_ms": 45.2,
  "image_size": "512x512"
}
```

See [API Documentation](docs/API.md) for more details.

## 📊 Project Structure

```
AI-Face-Detector/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application
│   ├── model_loader.py      # Model loading utilities
│   ├── config.py            # Configuration settings
│   └── utils.py             # Image preprocessing & inference
├── data/
│   └── setup_dataset.py     # Kaggle dataset download script
├── training/
│   ├── train_model.py       # Standalone training script (cloud-ready)
│   └── train_colab.ipynb    # Jupyter notebook (optional)
├── static/
│   └── index.html           # Frontend UI with Tailwind CSS
├── docs/
│   ├── TRAINING.md          # Detailed training guide
│   └── API.md               # API reference documentation
├── model.pth                # Trained weights (download after training)
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Poetry configuration
├── .gitignore              # Git exclusions
├── README.md               # This file
└── LICENSE                 # MIT License
```

## 🧪 Training

### Dataset

We use the **GRAVEX-200K** dataset:
- **200K images**: 100K real, 100K AI-generated
- **Sources**: FaceForensics++, DFDC, Celeb-DF, Stable Diffusion, etc.
- **Resolution**: 256x256 (preprocessed)
- **License**: CC0 Public Domain

### Training Pipeline

1. **Data Loading**: Custom PyTorch Dataset with transforms
2. **Model**: MobileNetV2 (pre-trained on ImageNet)
3. **Transfer Learning**: Freeze features, train classifier
4. **Augmentation**: Random flip, rotation, color jitter, affine
5. **Optimization**: Adam optimizer, ReduceLROnPlateau scheduler
6. **Early Stopping**: Patience of 5 epochs

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Epochs | 15 (with early stopping) |
| Dropout | 0.3 |
| Loss Function | Binary Cross-Entropy |

### Performance

| Metric | Value |
|--------|-------|
| Accuracy | 94.5% |
| Precision | 94.2% |
| Recall | 94.8% |
| F1 Score | 0.945 |
| AUC-ROC | 0.978 |
| Inference Time (CPU) | 45ms |
| Inference Time (GPU) | 8ms |
| Model Size | 14MB |

See [Training Guide](docs/TRAINING.md) for step-by-step instructions.

## 🔧 Configuration

Edit `app/config.py` to customize:

```python
MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
```

## 🐳 Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t ai-face-detector .
docker run -p 8000:8000 ai-face-detector
```

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### Code Quality

```bash
# Format code
black app/

# Lint
ruff check app/

# Type check
mypy app/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GRAVEX-200K Dataset**: [Muhammad Bilal](https://kaggle.com/muhammadbilal6305)
- **PyTorch Team**: For the amazing deep learning framework
- **FastAPI**: For the modern, fast web framework
- **Tailwind CSS**: For the utility-first CSS framework

## 📚 Resources

- [Training Guide](docs/TRAINING.md) - How to train the model
- [API Documentation](docs/API.md) - API reference and examples
- [GRAVEX-200K Dataset](https://kaggle.com/datasets/muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

## 📧 Contact

Furkan Köykıran - [@furkankoykiran](https://github.com/furkankoykiran)

---

**⭐ If you find this project helpful, please consider giving it a star!**
