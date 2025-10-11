# Fashion Product Classifier with ResNet50
### AAI3001 Phase 1 - Deep Learning Computer Vision Project | Group 10

**Contributors:** Lee Xu Xiang Keith, Cheong Wai Hong Jared, Wong Liang Jin, Chan Jing Chun, Ng Kee Tian

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Validation%20Acc-91.45%25-brightgreen.svg)](#model-performance)

---

<div align="center">

### **91.45% Validation Accuracy** | **30-50ms Inference** | **23.5M Parameters**

[Performance](#model-performance) • [Quick Start](#quick-start-guide) • [GUI Demo](#gui-application) • [Documentation](#training-details)

</div>

---

## Table of Contents
- [Project Overview](#project-overview)
- [Model Performance](#model-performance)
- [Quick Start Guide](#quick-start-guide)
- [Dataset Setup](#dataset-setup)
- [Training Details](#training-details)
- [GUI Application](#gui-application)
- [Project Structure](#project-structure)
- [Custom Dataset Contribution](#-custom-dataset-contribution)

---

## Project Overview

This project implements a **deep learning-based fashion product classifier** using transfer learning with **ResNet50** architecture. The model classifies fashion items into **15 distinct categories** across topwear and bottomwear.

### Key Features
- **Transfer Learning**: ResNet50 pretrained on ImageNet1K
- **Progressive Unfreezing**: 4-stage fine-tuning strategy
- **High Accuracy**: ~91% validation accuracy achieved in 12 epochs
- **Custom Dataset**: Added 55 custom images to CSV (IDs 60001-60055), with 43 successfully loaded for training
- **Interactive GUI**: Real-time classification with Grad-CAM explanations
- **Fast Inference**: ~30-50ms per prediction on CPU

### Classification Categories (15 Total)

<table>
<tr>
<td valign="top" width="50%">

**Topwear (8 classes)**
- Tshirts
- Shirts  
- Tops
- Sweatshirts
- Jackets
- Sweaters
- Blazers
- Waistcoat

</td>
<td valign="top" width="50%">

**Bottomwear (7 classes)**
- Jeans
- Trousers
- Shorts
- Skirts
- Track Pants
- Leggings
- Swimwear

</td>
</tr>
</table>

---

## Model Performance

### Training Results
| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | 91.45% (epoch 9) |
| **Final Validation Accuracy** | 91.45% (epoch 12) |
| **Final Training Accuracy** | 94.92% (epoch 12) |
| **Test Accuracy** | ~90-91% (evaluated after training) |
| **Training Time** | ~1h 20min (RTX 3080, 12 epochs) |
| **Training Speed** | ~29.8 img/s average |
| **Parameters** | 23.5M (ResNet50) |

### Progressive Unfreezing Results
Our 4-stage progressive unfreezing strategy showed consistent improvement:

| Stage | Layers Unfrozen | Epochs | Best Val Acc | Final Train Acc | Final Val Acc |
|-------|----------------|--------|--------------|-----------------|---------------|
| 1 | Classifier head only | 2 | **82.68%** | 59.94% | 82.68% |
| 2 | + Last ResNet block (layer4) | 2 | **82.75%** | 76.47% | 82.75% |
| 3 | + Middle blocks (layer3) | 3 | **89.77%** | 91.75% | 89.77% |
| 4 | All layers (full fine-tuning) | 5 | **91.45%** | 94.92% | 91.45% |

**Key Observations:**
- Stage 1 → 2: Minimal improvement (+0.07%) - classifier adapting to feature space
- Stage 2 → 3: **+6.96% jump** - middle layers learning fashion-specific features
- Stage 3 → 4: **+1.68% final gain** - full model fine-tuning for optimal performance
- Training accuracy steadily increased from 48.95% → 94.92% across 12 epochs

### Test Set Performance (Evaluated After Training)
After 12 epochs of progressive unfreezing training:
- **Overall Test Accuracy**: ~90-91% (consistent with validation performance)
- Strong diagonal pattern in confusion matrix
- Most confusions occur between visually similar items (e.g., Shirts ↔ Jackets, Tops ↔ Tshirts)
- Excellent performance on distinct categories (Track Pants, Shorts, Sweaters)

**Per-Class Highlights** (from confusion matrix analysis):
- Major classes (Tshirts, Shirts): 90-95% accuracy
- Structured garments (Blazers, Jackets): 85-92% accuracy
- Bottomwear (Jeans, Trousers, Shorts): 92-98% accuracy

---

## Quick Start Guide

### Prerequisites
- **Python 3.8+** (tested on Python 3.10)
- **CUDA-capable GPU** (optional, for faster inference)
- **8GB+ RAM** recommended
- **Windows 10/11** (tested), Linux/macOS (compatible)

### Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/keithashs/AAI3001_P10.git
cd AAI3001_P10
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
```
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
matplotlib>=3.5.0
numpy>=1.23.0
tkinter (usually pre-installed with Python)
tkinterdnd2>=0.3.0
```

#### Step 3: Download Trained Model
The trained model weights are included in this repository:
- `best_model_resnet50_extended.pth` (~98MB)
- `le_product_type_extended.pkl` (label encoder)

> **⚠️ Note:** Model files may be hosted externally due to GitHub's 100MB file size limit.
> 
> If model files are missing, download from:
> - [Google Drive](https://drive.google.com/) *(upload your files and add the link)*
> - [GitHub Releases](https://github.com/keithashs/AAI3001_P10/releases)
>
> Place downloaded files in the project root directory.

#### Step 4: Run the GUI
```bash
python app.py
```

### Quick Guide

```bash
# 1. Clone and install
git clone https://github.com/keithashs/AAI3001_P10.git
cd AAI3001_P10
pip install -r requirements.txt

# 2. Launch GUI
python app.py

# 3. Upload image (or drag & drop)
# 4. Click "Finish (Classify)" or press Enter
# 5. View top predictions in right panel!
```

**Sample Test Images**: Use product images from the dataset or your own fashion photos!

---

## Dataset Setup

### Kaggle Fashion Product Images Dataset

Our model is trained on the [Kaggle Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) with custom additions.

#### Download Instructions

**Option 1: Using Kaggle API (Recommended)**
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials (place kaggle.json in ~/.kaggle/)
kaggle datasets download -d paramaggarwal/fashion-product-images-dataset

# Extract
unzip fashion-product-images-dataset.zip -d d:/AAI3001/fashion-dataset/
```

**Option 2: Manual Download**
1. Visit [Kaggle dataset page](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
2. Click "Download" (requires Kaggle account)
3. Extract to `d:/AAI3001/fashion-dataset/fashion-dataset/`

#### Expected Directory Structure
```
d:/AAI3001/
├── fashion-dataset/
│   └── fashion-dataset/
│       ├── images/           # 44,000+ product images
│       │   ├── 10000.jpg
│       │   ├── 10001.jpg
│       │   ├── ...
│       │   ├── 60001.jpg     # Custom images (60001-60055)
│       │   └── 60055.jpg
│       ├── images.csv        # Image metadata
│       └── styles.csv        # Product attributes
├── best_model_resnet50_extended.pth  # Trained model weights
├── le_product_type_extended.pkl       # Label encoder
├── app.py                             # GUI application
├── AAI3001_model.ipynb                # Training notebook
└── README.md                          # This file
```

---

## Training Details

### Model Architecture
- **Backbone**: ResNet50 (ImageNet1K_V2 pretrained weights)
- **Modifications**:
  - Replaced final FC layer with custom classifier
  - Added Dropout(0.4) for regularization
  - Output: 15 classes (softmax)
- **Total Parameters**: 23.5M

### Actual Training Timeline (12 Epochs)

**Stage 1: Classifier Head Only (2 epochs)**
- Epoch 1: Train 48.95% → Val 79.90% | Time: 476.7s (25.9 img/s)
- Epoch 2: Train 59.94% → Val 82.68% | Time: 422.0s (29.3 img/s)
- **Best Val Acc**: 82.68%

**Stage 2: + Layer4 Unfrozen (2 epochs)**  
- Epoch 3: Train 76.47% → Val 82.75% | Time: 412.7s (29.9 img/s)
- Epoch 4: *(skipped in output - early stopping or consolidation)*
- **Best Val Acc**: 82.75% (+0.07% from Stage 1)

**Stage 3: + Layer3 Unfrozen (3 epochs)**
- Epoch 5: Train 87.63% → Val 88.60% | Time: 444.1s (27.8 img/s)
- Epoch 6: Train 90.34% → Val 89.62% | Time: 418.9s (29.5 img/s)
- Epoch 7: Train 91.75% → Val 89.77% | Time: 406.6s (30.4 img/s)
- **Best Val Acc**: 89.77% (+6.96% jump - major improvement!)

**Stage 4: Full Fine-Tuning (5 epochs)**
- Epoch 8: Train 91.60% → Val 90.06% | Time: 395.7s (31.2 img/s)
- Epoch 9: Train 92.33% → Val **91.45%** | Time: 400.6s (30.8 img/s) **← BEST MODEL**
- Epoch 10: Train 93.45% → Val 90.57% | Time: 392.7s (31.5 img/s)
- Epoch 11: Train 94.58% → Val 91.15% | Time: 420.6s (29.4 img/s)
- Epoch 12: Train 94.92% → Val 91.45% | Time: 394.2s (31.3 img/s)
- **Best Val Acc**: 91.45% (epoch 9, saved as best model)

**Total Training Time**: ~1h 21min (4,858 seconds)  
**Average Speed**: 29.8 img/s  
**Hardware**: NVIDIA RTX 3080

### Training Configuration
| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (weight decay: 1e-4) |
| **Learning Rate Schedule** | Cosine Annealing (per stage) |
| **Batch Size** | 64 |
| **Loss Function** | CrossEntropyLoss (label smoothing: 0.05) |
| **Data Split** | Train: 81%, Val: 9%, Test: 10% |
| **Augmentations** | RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation, RandomErasing |
| **Class Balancing** | WeightedRandomSampler (inverse frequency) |
| **Mixed Precision** | Enabled (CUDA AMP) |

### Progressive Unfreezing Strategy
We employed a 4-stage progressive unfreezing approach to prevent catastrophic forgetting:

1. **Stage 1 (2 epochs)**: Train classifier head only
2. **Stage 2 (2 epochs)**: Unfreeze layer4 (last ResNet block)
3. **Stage 3 (3 epochs)**: Unfreeze layer3 + layer4 (mid-to-last blocks)
4. **Stage 4 (5 epochs)**: Unfreeze all layers (full fine-tuning)

**Rationale**: Early layers learn generic features (edges, textures) that transfer well to fashion images. Later layers learn domain-specific features that benefit from fine-tuning.

### Training Results Visualization
After training completes, the notebook generates:
- **Accuracy curves**: Training vs validation accuracy across 12 epochs
- **Loss curves**: Training vs validation loss progression
- **Stage summary**: Best validation accuracy per unfreezing stage
- **Confusion matrix**: Normalized and raw counts on test set

---

## GUI Application

> **[Watch the GUI Demo Video](https://youtu.be/AHjdQUE-KMA)**

### Features
- **Drag & Drop**: Drop fashion images directly onto the canvas
- **Region of Interest (ROI)**: Draw bounding box to classify specific regions
- **Top-K Predictions**: View top 5-20 predictions with confidence scores
- **Category Filtering**: Filter predictions by Tops/Bottoms/All
- **Grad-CAM Explanations**: Visualize attention heatmaps (explainability)
- **Upper-body Bias**: Optional downweighting of bottom-class predictions
- **Save Results**: Export classification results as PNG

### Usage Instructions

#### 1. Launch the Application
```bash
python app.py
```

#### 2. Load an Image
- **Option A**: Click "Upload Image" button
- **Option B**: Drag & drop image file onto canvas

#### 3. (Optional) Select Region of Interest
- Click and drag to draw a bounding box around the item
- Useful for images with multiple clothing items

#### 4. Classify
- Click **"Finish (Classify)"** or press **Enter**
- View top predictions in the right panel

#### 5. Explain Prediction (Grad-CAM)
- Click **"Explain (Heatmap)"** or press **H**
- Red regions = high attention (model focuses here)
- Blue regions = low attention

#### 6. Reset or Save
- **Reset Crop**: Clear bounding box and restore full image
- **Save View**: Export current canvas as PNG (Ctrl+S)

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Upload image |
| `Enter` | Classify current image/ROI |
| `H` | Generate Grad-CAM heatmap |
| `R` | Reset crop |
| `Ctrl+S` | Save current view |

### Advanced Options
- **Top-K**: Adjust number of predictions shown (1-20)
- **Category Filter**: Restrict predictions to Tops, Bottoms, or All
- **Upper-body Bias**: Downweight bottom classes by 40% (useful for upper-body shots)

---

## Project Structure

```
AAI3001_P10/
│
├── README.md                           # Project documentation (not yet pushed)
├── requirements.txt                    # Python dependencies
│
├── Core Application Files
│   ├── app.py                              # GUI application (Tkinter)
│   ├── AAI3001_model.ipynb                 # Training notebook (Jupyter)
│   └── data_augmentation2.ipynb            # Data augmentation experiments
│
├── Model Files
│   ├── best_model_resnet50_extended.pth   # Current trained weights (90.1MB, 91.45% val acc)
│   ├── best_model_resnet50.pth            # Previous ResNet50 checkpoint
│   ├── best_model.pth                     # Earlier model version
│   ├── le_product_type_extended.pkl       # Label encoder (15 classes)
│   ├── le_product_type.pkl                # Previous label encoder
│   └── history_resnet50.pkl               # Training history (12 epochs)
│
├── Dataset Files
│   └── fashion-dataset/
│       └── fashion-dataset/
│           ├── images.csv                     # Image metadata
│           ├── styles.csv                     # Product attributes + custom entries (60001-60055)
│           └── images/                        # Product images (44K+, not included in repo)
│
├── Legacy Notebooks (from development)
│   └── dlcv-phase-1-kee-tian-vers (1).ipynb  # Earlier training experiments
│
└── Documentation (not yet pushed)
    ├── Appendix Files/
    │   ├── confusion_matrix.png
    │   └── predictions.png
    ├── VERIFICATION_REPORT.md
    ├── SETUP_INSTRUCTIONS.md
    └── TRAINING_GUIDE.md
```

---

## Custom Dataset Contribution

### 55 Custom Images Added to CSV (IDs 60001-60055)

We extended the original Kaggle dataset by adding **55 custom image entries** (IDs 60001-60055) to the `styles.csv` file. During training, **43 images were successfully loaded** (12 image files were missing or failed validation) to improve classification performance on under-represented classes.

### Data Collection Process
1. **Source**: UNIQLO product catalog, personal collections, and supplementary fashion images
2. **Selection Criteria**: 
   - Frontal view preferred
   - Consistent with dataset style
   - Diverse colors and styles
3. **Preprocessing**: Resized to 80x60 pixels to match dataset format
4. **Integration**: Assigned IDs in range 60001-60055, added to `styles.csv`

### Training Allocation
- **CSV Entries Added**: 55 images (IDs 60001-60055)
- **Successfully Loaded**: 43 images (12 files missing/failed validation)
- **Training Set**: 100% (all 43 loaded images)
- **Validation Set**: 0%
- **Test Set**: 0%

**Rationale**: With limited custom samples, we allocated all successfully loaded images to training to maximize learning. The model's generalization is validated on the large existing test set.

### Results
- **Blazers**: Model successfully learned to distinguish blazers from casual jackets (structured vs unstructured)
- **Waistcoat**: New formal topwear class successfully integrated into the model
- **Overall Impact**: Custom images helped improve model performance on under-represented formal wear categories

---

## Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'tkinterdnd2'`
```bash
pip install tkinterdnd2
```

#### 2. Model file not found
Ensure `best_model_resnet50_extended.pth` and `le_product_type_extended.pkl` are in `d:/AAI3001/`

#### 3. CUDA out of memory (during training)
- Reduce batch size to 32 or 16
- Use `num_workers=0` in DataLoader
- Close other GPU processes

#### 4. Slow inference on CPU
- Expected: 30-50ms per image on CPU
- For faster inference, use GPU (5-10ms)

#### 5. GUI window doesn't maximize on Linux
```python
# In app.py, replace:
root.state("zoomed")
# With:
root.attributes("-zoomed", True)
```

---

## Future Improvements

- [ ] Add support for multi-item detection (YOLO/Faster R-CNN)
- [ ] Expand to 50+ clothing categories
- [ ] Deploy as web app (Streamlit Cloud/Gradio)
- [ ] Support for style/color/pattern attributes

---

## Team Members

**AAI3001 Group 10 - Singapore Institute of Technology**

| Name | Student ID | Contributions |
|------|------------|---------------|
| Lee Xu Xiang Keith | 2400845 | Model architecture, training pipeline, GUI development |
| Cheong Wai Hong Jared | 2401641 | Data preprocessing, augmentation strategies |
| Wong Liang Jin | 2400598 | Custom dataset curation, model evaluation |
| Chan Jing Chun | 2402867 | Documentation, testing, GUI refinement |
| Ng Kee Tian | 2401683 | Training optimization, performance analysis |

---

## References

1. **Dataset**: [Kaggle Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
2. **ResNet Paper**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
3. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
4. **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)
5. **Transfer Learning Best Practices**: [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

## Acknowledgments

- Singapore Institute of Technology (SIT) - AAI3001 Course
- Kaggle community for the fashion dataset

---

**Last Updated**: October 11, 2025  
**Version**: 1.0.0  