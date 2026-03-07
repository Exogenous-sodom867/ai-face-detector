# Training Guide

This guide explains how to train the AI Face Detector model using Google Colab or Kaggle Notebooks.

## Prerequisites

- Google account (for Colab) or Kaggle account (for Kaggle Notebooks)
- Basic understanding of Python and PyTorch
- 140k real and fake faces dataset (will be downloaded automatically)

## Option 1: Kaggle Notebooks (Recommended - Easy Setup)

### Step 1: Create Kaggle Notebook

1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Click "New Notebook"
3. Enable GPU accelerator (right sidebar > Settings > Accelerator > GPU T4)

### Step 2: Add Dataset

1. Right sidebar > Add data
2. Search for "140k real and fake faces" by xhlulu
3. Click "Add" (dataset link: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

### Step 3: Upload Training Script

1. Copy the content from `training/train_kaggle.py` in this repository
2. Paste into a new code cell in your Kaggle notebook
3. Run the cell to execute the training script

**OR** run this command in a cell:

```python
# Install dependencies (usually pre-installed on Kaggle)
!pip install torch torchvision pillow matplotlib tqdm scikit-learn

# Download and run the Kaggle training script
!wget https://raw.githubusercontent.com/furkankoykiran/ai-face-detector/main/training/train_kaggle.py
!python train_kaggle.py
```

### Step 4: Monitor Training

The script will:
- ✅ Automatically detect the dataset
- ✅ Display dataset statistics
- ✅ Show training progress with progress bars
- ✅ Save best model automatically
- ✅ Generate evaluation metrics

### Step 5: Download Model

After training completes:

1. Open the file browser (📁 icon on the right sidebar)
2. Go to `/kaggle/working/`
3. Download these files:
   - **`model.pth`** - Trained model weights (MOST IMPORTANT!)
   - `training_history.json` - Training metrics per epoch
   - `evaluation_report.json` - Test set performance
   - `training_curves.png` - Visualization of training

### Step 6: Use the Model

1. Place `model.pth` in your project root directory
2. Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Open `static/index.html` in your browser to test!

---

### Why Kaggle?

- ✅ **Easy setup**: No API keys or downloads needed
- ✅ **Free GPU**: Tesla T4 GPU available
- ✅ **Auto-detection**: Script finds dataset automatically
- ✅ **140K images**: More data = better accuracy (~95%+)
- ✅ **Pre-split**: Train/val/test already organized

## Option 2: Google Colab

### Step 1: Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click "New Notebook"
3. Select "GPU" from Runtime > Change runtime type > Hardware accelerator

### Step 2: Install Dependencies

```python
!pip install torch torchvision pillow matplotlib tqdm scikit-learn
```

### Step 3: Download the Dataset

**Option A: Using Kaggle CLI (Recommended)**

```python
# Install Kaggle CLI
!pip install kaggle

# Upload your kaggle.json API key (from Kaggle account settings)
from google.colab import files
files.upload()

# Download dataset
!kaggle datasets download -d xhlulu/140k-real-and-fake-faces

# Extract dataset
!unzip 140k-real-and-fake-faces.zip -d /content/data
```

**Option B: Manual Upload**

1. Download dataset from Kaggle: [140k Real vs Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
2. Extract the zip file
3. Upload the `real_vs_fake` folder to Colab

### Step 4: Upload Training Script

1. Copy the content from `training/train_model.py`
2. Create a new cell in Colab and paste the script
3. Or upload the file directly to Colab

### Step 5: Configure Data Path

At the top of the script, update the `DATA_PATH`:

```python
DATA_PATH = Path("/content/data/real_vs_fake")  # For Colab
AI_IMAGES_DIR = "train/fake"
REAL_IMAGES_DIR = "train/real"
```

### Step 6: Run Training

```python
!python train_model.py
```

Or run directly in the notebook:

```python
from train_model import main
main()
```

### Step 7: Download Trained Model

After training completes, download the files:

1. Open the file browser (📁 icon on the left)
2. Right-click on `model.pth` and select "Download"
3. Save it to your project root directory

## Training Configuration

Default hyperparameters (can be modified in `Config` class):

```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 5
DROPOUT_RATE = 0.3
```

## Expected Training Time

- **With GPU (Kaggle T4)**: ~20-30 minutes for 15 epochs
- **With GPU (Colab)**: ~30-45 minutes for 15 epochs
- **With CPU**: ~3-4 hours (not recommended)

## Expected Performance

On the test set, you should achieve:
- **Accuracy**: 93-96%
- **F1 Score**: 0.93-0.96
- **AUC-ROC**: 0.97-0.99

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```python
BATCH_SIZE = 16  # or 8
```

### Training Too Slow

1. Verify GPU is enabled:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
2. Check GPU utilization:
   ```python
   !nvidia-smi
   ```

### Dataset Not Found (Kaggle)

1. Make sure you added the dataset in the right sidebar
2. Check the dataset name: "140k real and fake faces" by xhlulu
3. Look at the "Input" tab to verify it's listed

### Dataset Not Found (Colab)

1. Verify dataset path is correct
2. Check if dataset was extracted properly
3. Use `!ls -R` to list all files
4. Make sure DATA_PATH points to the real_vs_fake folder

### Poor Accuracy

1. Train for more epochs (increase `NUM_EPOCHS`)
2. Try different learning rates (0.0001, 0.01)
3. Increase data augmentation
4. Use a larger model (ResNet18 instead of MobileNetV2)

## Advanced: Custom Training

### Using Different Models

Edit `create_model()` function:

```python
# Use ResNet18 instead of MobileNetV2
from torchvision import models

model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, 1)
)
```

### Hyperparameter Tuning

Try different combinations:

```python
# High accuracy, slower training
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 25

# Fast training, lower accuracy
BATCH_SIZE = 16
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
```

### Resume Training

To resume from a checkpoint:

```python
# Load checkpoint
model.load_state_dict(torch.load("model.pth"))
optimizer.load_state_dict(torch.load("optimizer.pth"))

# Continue training
for epoch in range(start_epoch, NUM_EPOCHS):
    # ... training code
```

## Next Steps

After training:

1. **Test the model**: Run inference on sample images
2. **Deploy the API**: Use `model.pth` with the FastAPI backend
3. **Monitor performance**: Check evaluation metrics and confusion matrix
4. **Iterate**: Adjust hyperparameters and retrain if needed

## Support

For issues or questions:
- Check the [GitHub Issues](https://github.com/furkankoykiran/ai-face-detector/issues)
- Review the training logs and error messages
- Verify all dependencies are installed correctly
