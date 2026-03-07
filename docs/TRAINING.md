# Training Guide

This guide explains how to train the AI Face Detector model using Google Colab or Kaggle Notebooks.

## Prerequisites

- Google account (for Colab) or Kaggle account (for Kaggle Notebooks)
- Basic understanding of Python and PyTorch
- GRAVEX-200K dataset (will be downloaded automatically)

## Option 1: Google Colab (Recommended)

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
!kaggle datasets download -d muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal

# Extract dataset
!unzip 200k-real-vs-ai-visuals-by-mbilal.zip -d /content/data
```

**Option B: Manual Upload**

1. Download dataset from Kaggle: [GRAVEX-200K](https://kaggle.com/datasets/muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal)
2. Extract the zip file
3. Upload the `my_real_vs_ai_dataset` folder to Colab

### Step 4: Upload Training Script

1. Copy the content from `training/train_model.py`
2. Create a new cell in Colab and paste the script
3. Or upload the file directly to Colab

### Step 5: Configure Data Path

At the top of the script, update the `DATA_PATH`:

```python
DATA_PATH = Path("/content/data")  # For Colab
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

## Option 2: Kaggle Notebooks

### Step 1: Create Kaggle Notebook

1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Click "New Notebook"
3. Enable GPU accelerator (right sidebar > Settings > Accelerator > GPU)

### Step 2: Add Dataset

1. Right sidebar > Add data
2. Search for "GRAVEX-200K" or "200k-real-vs-ai-visuals-by-mbilal"
3. Click "Add"

### Step 3: Upload Training Script

1. Copy `training/train_model.py` content
2. Paste into a new code cell
3. Or use the "Add utility script" feature

### Step 4: Configure Data Path

Update the `DATA_PATH` at the top of the script:

```python
DATA_PATH = Path("/kaggle/input/200k-real-vs-ai-visuals-by-mbilal")  # For Kaggle
```

### Step 5: Run Training

```python
# Install dependencies (if needed)
!pip install torch torchvision pillow matplotlib tqdm scikit-learn

# Run training
!python train_model.py
```

### Step 6: Download Model

1. Go to the "Output" tab on the right sidebar
2. Download `model.pth`, `training_history.json`, and `evaluation_report.json`

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

- **With GPU (Colab/Kaggle)**: ~30-45 minutes for 15 epochs
- **With CPU**: ~3-4 hours (not recommended)

## Expected Performance

On the test set, you should achieve:
- **Accuracy**: 90-95%
- **F1 Score**: 0.90-0.95
- **AUC-ROC**: 0.95-0.98

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

### Dataset Not Found

1. Verify dataset path is correct
2. Check if dataset was extracted properly
3. Use `!ls -R` to list all files

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
