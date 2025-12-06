# MNIST Digit Classifier

A Convolutional Neural Network (CNN) built with PyTorch for classifying handwritten digits from the MNIST dataset. This project demonstrates fundamental deep learning concepts including CNN architecture, training loops, and model evaluation.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Future Improvements](#future-improvements)

---

## Overview

This project implements a Convolutional Neural Network to recognize handwritten digits (0-9) from the MNIST dataset. The model uses modern deep learning techniques including:

- **Convolutional layers** for feature extraction
- **Dropout regularization** to prevent overfitting
- **Adam optimizer** for efficient training
- **MPS (Metal Performance Shaders)** support for Apple Silicon acceleration

**Key Features:**
- Achieves ~95-97% accuracy on validation set
- Automatic hardware detection (MPS/CPU)
- Training visualization with loss and accuracy plots
- Clean, modular code structure

---

## Dataset

### MNIST Dataset
- **Source:** [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **Description:** Collection of 70,000 handwritten digit images
- **Classes:** 10 (digits 0-9)

| Split | Images | Purpose |
|-------|--------|---------|
| Training | 60,000 | Model training |
| Test | 10,000 | Model evaluation |

### Data Preprocessing
- **Image Size:** 28×28 pixels (grayscale)
- **Normalization:** Converted to tensors with `ToTensor()` transform
- **Range:** Pixel values scaled to [0, 1]
- **No augmentation:** Using raw MNIST images

---

## Model Architecture

The model is a Convolutional Neural Network implemented in `model.py` as `DigitClassifierModel`.

### Architecture Overview

```
Input (1×28×28) 
    ↓
[Convolutional Block]
    Conv2D (1 → 28 channels, 3×3 kernel)
    ReLU
    Conv2D (28 → 64 channels, 3×3 kernel)
    ReLU
    MaxPool2D (2×2)
    ↓
Flatten
    ↓
[Fully Connected Block]
    Linear (9216 → 128)
    ReLU + Dropout (0.3)
    Linear (128 → 64)
    ReLU + Dropout (0.3)
    Linear (64 → 10)
    ↓
Output (10 classes)
```

### Layer Details

| Layer Type | Input Shape | Output Shape | Parameters |
|------------|-------------|--------------|------------|
| Conv2D | (1, 28, 28) | (28, 26, 26) | 280 |
| ReLU | (28, 26, 26) | (28, 26, 26) | 0 |
| Conv2D | (28, 26, 26) | (64, 24, 24) | 16,192 |
| ReLU | (64, 24, 24) | (64, 24, 24) | 0 |
| MaxPool2D | (64, 24, 24) | (64, 12, 12) | 0 |
| Flatten | (64, 12, 12) | (9216) | 0 |
| Linear | (9216) | (128) | 1,179,776 |
| Dropout | (128) | (128) | 0 |
| Linear | (128) | (64) | 8,256 |
| Dropout | (64) | (64) | 0 |
| Linear | (64) | (10) | 650 |

**Total Parameters:** ~1.2M trainable parameters

### Design Rationale
- **Convolutional layers** extract spatial features from digit images
- **MaxPooling** reduces spatial dimensions while retaining important features
- **Dropout (30%)** prevents overfitting by randomly deactivating neurons during training
- **Multiple FC layers** allow the model to learn complex decision boundaries
- **ReLU activation** introduces non-linearity for better feature learning

---

## Training

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 10 | Number of complete passes through training data |
| Batch Size | 64 | Number of samples per gradient update |
| Learning Rate | 0.001 | Step size for optimizer |
| Optimizer | Adam | Adaptive moment estimation optimizer |
| Loss Function | CrossEntropyLoss | Multi-class classification loss |
| Device | MPS/CPU | Apple Silicon GPU or CPU fallback |

### Training Process

The training pipeline (`train.py`) includes:

1. **Data Loading:** Automatic MNIST dataset download and DataLoader creation
2. **Model Initialization:** DigitClassifierModel instantiation and device transfer
3. **Training Loop:** 
   - Forward pass through training data
   - Loss computation
   - Backpropagation
   - Optimizer step
4. **Validation:** Evaluation on test set after each epoch
5. **Visualization:** Loss and accuracy plots saved as PNG files
6. **Model Saving:** Final weights saved to `digit_classifier.pth`

### Training Features
- ✅ Automatic hardware detection (MPS for Apple Silicon, CPU fallback)
- ✅ Validation after each epoch
- ✅ Training/validation loss tracking
- ✅ Accuracy metrics computation
- ✅ Visualization plots generation

---

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Final Validation Accuracy** | **95-97%** |
| Training Time | ~2-5 minutes (depending on hardware) |
| Model Size | ~4.8 MB |

### Training Visualizations

The training process generates three visualization plots:

1. **training_loss.png** - Training loss over epochs
2. **validation_loss.png** - Validation loss over epochs  
3. **training_accuracy.png** - Validation accuracy progression

These plots help diagnose:
- Model convergence
- Overfitting/underfitting
- Training stability

### Model Strengths
- High accuracy on clean, centered digits
- Fast inference time
- Lightweight architecture suitable for deployment
- Robust to typical handwriting variations

---

## Installation

### Prerequisites
- Python 3.11+ 
- pip package manager
- (Optional) Apple Silicon Mac for MPS acceleration

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/mecca1991/digit-classifier.git
cd digit-classifier
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install torch torchvision matplotlib numpy
```

### Required Packages
- `torch` - PyTorch deep learning framework
- `torchvision` - Vision datasets and transforms
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical operations

---

## Usage

### Training the Model

Run the training script to train from scratch:

```bash
python train.py
```

**What happens:**
1. Downloads MNIST dataset (first run only) to `./data/`
2. Initializes the CNN model
3. Trains for 10 epochs
4. Generates training plots (PNG files)
5. Saves trained model to `digit_classifier.pth`

**Output:**
- `digit_classifier.pth` - Trained model weights
- `training_loss.png` - Training loss curve
- `validation_loss.png` - Validation loss curve
- `training_accuracy.png` - Accuracy progression

### Loading and Using the Trained Model

```python
import torch
from model import DigitClassifierModel
from torchvision import transforms
from PIL import Image

# Load the trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = DigitClassifierModel()
model.load_state_dict(torch.load("digit_classifier.pth"))
model.to(device)
model.eval()

# Prepare an image for inference
transform = transforms.ToTensor()
image = Image.open("your_digit_image.png").convert('L')  # Grayscale
image = image.resize((28, 28))
image_tensor = transform(image).unsqueeze(0).to(device)

# Make prediction
with torch.no_grad():
    output = model(image_tensor)
    prediction = output.argmax(dim=1).item()
    print(f"Predicted digit: {prediction}")
```

### Evaluating the Model

To evaluate on the test set:

```python
from train import get_data_loader, testing
from model import DigitClassifierModel
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# For computers with cuda
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, test_loader = get_data_loader()

model = DigitClassifierModel()
model.load_state_dict(torch.load("digit_classifier.pth"))
model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
test_loss, test_accuracy = testing(test_loader, model, loss_fn)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
```

---

## Project Structure

```
digit-classifier/
│
├── model.py                  # CNN model architecture definition
├── train.py                  # Training script with evaluation
├── digit_classifier.pth      # Saved model weights (after training)
│
├── training_loss.png         # Training loss visualization
├── training_accuracy.png     # Validation accuracy plot
├── validation_loss.png       # Validation loss plot
│
├── LICENSE                   # Apache 2.0 License
├── README.md                 # Project documentation (this file)
├── .gitignore               # Git ignore rules
│
├── data/                     # MNIST dataset (auto-downloaded)
│   └── MNIST/
│       ├── raw/
│       └── processed/
│
└── venv/                     # Virtual environment (not tracked)
```

---

## Requirements

### Core Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
```

### Hardware Requirements

**Minimum:**
- CPU: Any modern processor
- RAM: 4GB
- Storage: 500MB (for dataset + dependencies)

## Future Improvements

### Model Enhancements
- [ ] Implement data augmentation (rotation, scaling, noise)
- [ ] Experiment with deeper architectures (ResNet, VGG)
- [ ] Add batch normalization for faster convergence
- [ ] Try different optimizers (SGD with momentum, AdamW)
- [ ] Implement learning rate scheduling

### Code Quality
- [ ] Add unit tests for model and training functions
- [ ] Create separate config file for hyperparameters
- [ ] Add command-line arguments for training configuration
- [ ] Implement logging with TensorBoard
- [ ] Add model checkpointing during training

### Deployment
- [ ] Create REST API for model inference
- [ ] Build simple web interface for digit drawing
- [ ] Export model to ONNX format
- [ ] Optimize for mobile deployment (PyTorch Mobile)
- [ ] Add Docker containerization

### Documentation
- [ ] Add detailed API documentation
- [ ] Create Jupyter notebook tutorial
- [ ] Include example predictions with visualizations
- [ ] Add troubleshooting guide

---

## Acknowledgments

- **Dataset:** [MNIST Database](http://yann.lecun.com/exdb/mnist/) by Yann LeCun
- **Framework:** [PyTorch](https://pytorch.org/)
- **Inspiration:** Classic computer vision and deep learning nanodegree from Udacity
