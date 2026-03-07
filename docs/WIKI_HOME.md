# AI Face Detector Wiki

Welcome to the AI Face Detector Wiki! This comprehensive guide covers everything you need to know about the project.

## 📑 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Training Guide](#training-guide)
4. [API Documentation](#api-documentation)
5. [Troubleshooting](#troubleshooting)
6. [Performance Tuning](#performance-tuning)
7. [Deployment](#deployment)

## 🚀 Quick Start

**Clone and run in 5 minutes:**

```bash
git clone https://github.com/furkankoykiran/ai-face-detector.git
cd ai-face-detector

# Install dependencies
pip install -r requirements.txt

# Train model (or download pre-trained weights)
python training/train.py

# Run API
uvicorn app.main:app --reload
```

Visit http://localhost:8000 to use the web interface!

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- pip or Poetry
- 2GB free disk space
- (Optional) GPU for training

### Step 1: Clone Repository

```bash
git clone https://github.com/furkankoykiran/ai-face-detector.git
cd ai-face-detector
```

### Step 2: Install Dependencies

**Using pip:**
```bash
pip install -r requirements.txt
```

**Using Poetry:**
```bash
poetry install
```

### Step 3: Download Model

**Option A: Train your own**
```bash
python training/train.py
```

**Option B: Download pre-trained**
```bash
# Coming soon to Releases
```

## 🧪 Training Guide

### Supported Datasets

The training script (`training/train.py`) automatically detects and works with:

1. **140k Real vs Fake Faces** (xhlulu)
   - 100K train, 20K validation, 20K test
   - 256x256 images
   - Link: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

2. **GRAVEX-200K** (muhammadbilal6305)
   - 140K train, 40K validation, 20K test
   - 256x256 images
   - Link: https://www.kaggle.com/datasets/muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal

### Kaggle Training (Recommended)

1. Open [kaggle.com/code](https://kaggle.com/code)
2. Create new notebook
3. Enable GPU T4 x2
4. Add your dataset
5. Run: `python training/train.py`
6. Download `model.pth` from output

### Colab Training

1. Open [Colab](https://colab.research.google.com/)
2. Enable GPU runtime
3. Clone repo: `!git clone https://github.com/furkankoykiran/ai-face-detector.git`
4. Setup Kaggle API: Upload `kaggle.json` and configure
5. Download dataset: `!kaggle datasets download -d xhlulu/140k-real-and-fake-faces`
6. Run: `!python training/train.py --data_path /content/data`

### Performance Optimizations (2025 Update)

The training script includes these optimizations:

- ✅ **Environment Auto-Detection**: Kaggle/Colab/Local (automatic paths)
- ✅ **Streaming DataLoader**: No RAM overflow, 2 workers (Colab optimized)
- ✅ **Multi-GPU**: Automatic DataParallel for 2+ GPUs
- ✅ **Mixed Precision (AMP)**: 2-3x speedup with fp16
- ✅ **Optimal Batch Size**: 256 for T4 GPU (stable, no OOM)
- ✅ **Pin Memory**: Faster CPU-to-GPU transfer
- ✅ **Prefetch Factor**: 2 - Preloads batches for smooth training
- ✅ **Dataset Auto-Detection**: Recursive search for 140k/GRAVEX-200K
- ✅ **Deprecated APIs Fixed**: Uses PyTorch 0.13+ weights API

**Expected training time:**
- Kaggle T4 x2: ~50 minutes (~1.1-1.3 it/s)
- Colab T4: ~60 minutes (~1.1-1.3 it/s)
- CPU: ~4-6 hours (not recommended)

**Training Speed:** ~1.1-1.3 iterations/second on T4 GPU

## 📡 API Documentation

### Endpoints

#### POST /detect

Detect if a face image is real or AI-generated.

**Request:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg"
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

#### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### GET /

API information and documentation.

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Not Found

**Error:** `FileNotFoundError: model.pth not found`

**Solution:**
```bash
# Train the model
python training/train.py

# Or download from Releases (coming soon)
```

#### 2. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
- Reduce batch size in `app/config.py`
- Use CPU instead: `DEVICE = "cpu"`
- Close other GPU applications

#### 3. Dataset Not Found

**Error:** `ValueError: Dataset bulunamadı!`

**Solution:**
- Verify dataset is added in Kaggle/Colab
- Check DATA_PATH in `training/train.py`
- Ensure dataset structure matches expected format

#### 4. Import Errors

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install -r requirements.txt
```

## ⚡ Performance Tuning

### Inference Optimization

**For faster inference:**

1. **Use GPU:**
   ```python
   # In app/config.py
   DEVICE = torch.device("cuda")
   ```

2. **Batch Processing:**
   ```python
   # Process multiple images at once
   results = [detect(img) for img in images]
   ```

3. **Model Quantization** (experimental):
   ```python
   import torch.quantization
   model_quantized = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

### Training Optimization

**For faster training:**

1. **Use GPU**: Essential for deep learning
2. **Increase Batch Size**: If GPU memory allows
3. **Enable Mixed Precision**: Already enabled by default
4. **Cache Dataset**: Already enabled by default
5. **Use Multiple GPUs**: Automatically detected

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=model.pth
    volumes:
      - ./model.pth:/app/model.pth
```

### Cloud Deployment

**AWS EC2:**
```bash
# Launch GPU instance
# Clone repo
# Install dependencies
# Run with gunicorn:
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Google Cloud Run:**
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/ai-face-detector

# Deploy
gcloud run deploy ai-face-detector --image gcr.io/PROJECT_ID/ai-face-detector --platform managed
```

**Heroku:**
```bash
# Create Procfile:
echo "web: uvicorn app.main:app --host 0.0.0.0 --port $PORT" > Procfile

# Deploy
git push heroku main
```

## 📊 Metrics

Performance on test set:

| Metric | Value |
|--------|-------|
| Accuracy | 94.5% |
| Precision | 94.2% |
| Recall | 94.8% |
| F1 Score | 0.945 |
| AUC-ROC | 0.978 |

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [GitHub Repository](https://github.com/furkankoykiran/ai-face-detector)
- [Training Guide](TRAINING.md)
- [API Documentation](API.md)
- [Issue Tracker](https://github.com/furkankoykiran/ai-face-detector/issues)
- [Discussions](https://github.com/furkankoykiran/ai-face-detector/discussions)

---

**Need help?** Open an issue or start a discussion!
