# 👗 Fashion Intelligence Suite
### AAI3001 Deep Learning for Computer Vision | Group 10

**Singapore Institute of Technology**

**Contributors:** Lee Xu Xiang Keith, Cheong Wai Hong Jared, Wong Liang Jin, Chan Jing Chun, Ng Kee Tian

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue.svg)](https://ultralytics.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/liangjinwong/fashion-detection-system)

---

<div align="center">

## 🚀 [Live Demo on Hugging Face](https://huggingface.co/spaces/liangjinwong/fashion-detection-system)

### **Multi-Model AI System** | **31 Fashion Classes** | **Real-Time Detection**

[Features](#-key-features) • [Models](#-models--performance) • [Installation](#-installation) • [GUI Guide](#-gui-application) • [Training](#-training-details)

</div>

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Models & Performance](#-models--performance)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [GUI Application](#-gui-application)
- [Training Details](#-training-details)
- [Dataset Information](#-dataset-information)
- [Active Learning](#-active-learning-with-roboflow)
- [Web Deployment](#-web-deployment)
- [Project Structure](#-project-structure)

---

## 🎯 Project Overview

The **Fashion Intelligence Suite** is a comprehensive multi-model AI system for fashion item detection and classification. Built as a course project for AAI3001 (Deep Learning for Computer Vision), it demonstrates advanced computer vision techniques including:

- **Object Detection** using fine-tuned YOLOv8
- **Image Classification** using transfer learning with ResNet50  
- **Multi-stage Pipeline** for hierarchical fashion analysis
- **Active Learning** with Roboflow for continuous improvement
- **Real-time Inference** via webcam and image upload

### What Makes This Project Unique

1. **Three Specialized Models Working Together:**
   - Clothes Detector (YOLOv8) → Detects 13 clothing categories
   - Accessory Detector (YOLOv8) → Detects 11 accessory categories  
   - Shoe Classifier (ResNet50) → Classifies 7 shoe types when shoes are detected

2. **Logic-Based Filtering:**
   - Gravity checks (shoes should be in lower 60% of image)
   - Skin detection (filters out false positives on exposed skin)
   - Duplicate removal between overlapping detectors

3. **Style Classification:**
   - Rule-based outfit style analysis (Casual, Smart Casual, Business Formal, Sporty, Elegant, Street Style)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Multi-Model Detection** | Three specialized models for comprehensive fashion detection |
| 📸 **Dual Input Modes** | Image upload + Live webcam detection |
| 🔍 **Grad-CAM Visualization** | Explainable AI showing model attention areas |
| 👗 **Style Classification** | Automatic outfit style analysis |
| 🎨 **Smart Preprocessing** | Image enhancement for low-quality webcams |
| 🌐 **Web Deployment** | Hugging Face Spaces for easy access |
| 🔄 **Active Learning** | Continuous improvement via Roboflow |

---

## 🧠 Models & Performance

### Model Summary

| Model | Architecture | Task | Classes | Metric | Score |
|-------|-------------|------|---------|--------|-------|
| **Clothes Detector** | YOLOv8s | Object Detection | 13 | mAP50 | **0.80** |
| **Accessory Detector** | YOLOv8s | Object Detection | 11 | mAP50 | **0.75** |
| **Shoe Classifier** | ResNet50 | Classification | 7 | Accuracy | **82.5%** |
| **Fashion Classifier** (Phase 1) | ResNet50 | Classification | 15 | Accuracy | **91.45%** |

### Classes Detected

<table>
<tr>
<td valign="top" width="33%">

**👕 Clothing (13 classes)**
- Short Sleeve Top
- Long Sleeve Top
- Vest
- Sling
- Shorts
- Trousers
- Skirt
- Short Sleeve Dress
- Long Sleeve Dress
- Vest Dress
- Sling Dress
- Short Sleeve Outwear
- Long Sleeve Outwear

</td>
<td valign="top" width="33%">

**👜 Accessories (11 classes)**
- Jacket
- Coat
- Glasses
- Hat
- Tie
- Watch
- Belt
- Sock
- Shoe
- Bag
- Scarf

</td>
<td valign="top" width="33%">

**👟 Shoe Types (7 classes)**
- Casual Shoes
- Flats
- Flip Flops
- Formal Shoes
- Heels
- Sandals
- Sports Shoes

</td>
</tr>
</table>

### Training Curves (Phase 2 - Clothes Detector)

After Active Learning with Roboflow v6 dataset:

| Epoch | mAP50 | Precision | Recall |
|-------|-------|-----------|--------|
| 1 | 0.30 | 0.77 | 0.32 |
| 10 | 0.76 | 0.90 | 0.55 |
| 20 | 0.77 | 0.68 | 0.71 |
| **30** | **0.80** | 0.83 | 0.72 |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FASHION INTELLIGENCE SUITE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   PHASE 1    │    │   PHASE 2    │    │   PHASE 3    │       │
│  │  ResNet50    │    │   YOLOv8s    │    │   YOLOv8s    │       │
│  │  Classifier  │    │   Clothes    │    │ Accessories  │       │
│  │  (15 types)  │    │  (13 types)  │    │  (11 types)  │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         │                   │    ┌──────────────┤                │
│         │                   │    │   If "shoe"  │                │
│         │                   │    ▼              │                │
│         │                   │  ┌──────────────┐ │                │
│         │                   │  │  ResNet50    │ │                │
│         │                   │  │   Shoe       │ │                │
│         │                   │  │  Classifier  │ │                │
│         │                   │  │  (7 types)   │ │                │
│         │                   │  └──────┬───────┘ │                │
│         │                   │         │         │                │
│         ▼                   ▼         ▼         ▼                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              LOGIC-BASED POST-PROCESSING              │       │
│  │  • Gravity checks (shoes in lower 60%)                │       │
│  │  • Skin detection (filter false positives)            │       │
│  │  • Duplicate removal (overlapping boxes)              │       │
│  │  • Style classification (outfit analysis)             │       │
│  └──────────────────────────────────────────────────────┘       │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                    OUTPUT LAYER                       │       │
│  │  • Annotated image with bounding boxes               │       │
│  │  • Class labels with confidence scores               │       │
│  │  • Style classification results                       │       │
│  │  • Grad-CAM heatmap visualization                    │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💻 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Webcam (for live detection)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/keithashs/AAI3001_P10.git
cd AAI3001_P10

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
pillow>=9.0.0
numpy>=1.23.0
matplotlib>=3.5.0
scikit-learn>=1.2.0
tqdm>=4.65.0
gradio>=3.50.0  # For web deployment
```

---

## 🖥 GUI Application

### Running the Desktop App

```bash
# Full-featured GUI with all models
python app_with_preprocessing.py

# Or the original Phase 1 GUI
python app.py
```

### GUI Features

#### Tab 1: Phase 1 Classification
- **Drag & Drop** or click to upload images
- **Crop Tool** to select specific regions
- **Grad-CAM** visualization for model explainability
- **Style Classification** for outfit analysis

#### Tab 2: Phase 2/3 Detection
- **Webcam Mode** for real-time detection
- **Image Mode** for static image analysis
- **Confidence Slider** (0.1 - 1.0) to adjust detection sensitivity
- **Enhancement Mode** (Off/Light/Strong) for low-quality webcams

### GUI Screenshots

| Phase 1 Classification | Phase 2/3 Detection |
|------------------------|---------------------|
| Drag & drop image → Crop → Classify | Webcam or image → Multi-model detection |
| Grad-CAM heatmap visualization | Bounding boxes with class labels |
| Style classification results | Shoe type specialization |

---

## 📚 Training Details

### Phase 1: Fashion Classifier (ResNet50)

**Training Strategy:** Progressive Unfreezing

| Stage | Layers Unfrozen | Epochs | Best Val Acc |
|-------|----------------|--------|--------------|
| 1 | Classifier head only | 2 | 82.68% |
| 2 | + layer4 | 2 | 82.75% |
| 3 | + layer3 | 4 | 87.21% |
| 4 | Full model | 8 | **91.45%** |

**Hyperparameters:**
- Optimizer: AdamW (weight_decay=0.01)
- Learning Rate: CosineAnnealingWarmRestarts (η_max=3e-4)
- Batch Size: 64
- Augmentation: RandomHorizontalFlip, ColorJitter, RandomRotation

### Phase 2: Clothes Detector (YOLOv8)

**Training Configuration:**
```yaml
model: yolov8s.pt
data: DeepFashion2 (13 classes)
epochs: 30
imgsz: 640
batch: 16
optimizer: auto
augment: True
```

**Active Learning Iterations:**
1. Initial training on DeepFashion2
2. Error analysis → Identified "Shorts" misclassifications
3. Roboflow v6 dataset with targeted augmentation
4. **Final mAP50: 0.80** (improved from 0.77)

### Phase 3: Accessory Detector (YOLOv8)

**Training Configuration:**
```yaml
model: yolov8s.pt
data: Fashionpedia (11 classes)
epochs: 30
imgsz: 640
batch: 16
```

**Final mAP50: 0.75**

### Shoe Classifier (ResNet50)

**Training Details:**
- Base: ResNet50 pretrained on ImageNet
- Fine-tuned on Fashion Dataset shoe images
- 7 classes
- **Final Accuracy: 82.5%**

---

## 📊 Dataset Information

### DeepFashion2 (Clothes Detection)
- **Source:** [DeepFashion2 Dataset](https://github.com/switchablenorms/DeepFashion2)
- **Classes:** 13 clothing categories
- **Images:** ~190k training, ~32k validation

### Fashionpedia (Accessory Detection)
- **Source:** [Fashionpedia](https://fashionpedia.github.io/home/)
- **Classes:** 11 accessory categories (filtered from 46)
- **Focus:** Accessories, bags, shoes

### Fashion Product Images (Classification)
- **Source:** Kaggle Fashion Product Images Dataset
- **Classes:** 15 categories (8 topwear, 7 bottomwear)
- **Images:** ~44k total

### Custom Additions
- 55 custom images added (IDs 60001-60055)
- Includes Blazers and Waistcoat categories
- Manual annotation for quality control

---

## 🔄 Active Learning with Roboflow

### Problem Identified
During testing, we noticed the "Shorts" class had significant misclassifications (accuracy dropped to ~60%).

### Solution: Roboflow Active Learning Pipeline

1. **Error Collection:** Gathered misclassified samples
2. **Roboflow Upload:** Created v6 dataset with:
   - 200+ additional "Shorts" images
   - Targeted augmentation (brightness, rotation)
3. **Retraining:** Fine-tuned on combined dataset
4. **Results:**
   - Shorts accuracy: **100%** (up from 60%)
   - Overall mAP50: **0.80** (up from 0.77)

### Roboflow Dataset Versions
- `My First Project.v5i.yolov8/` - Initial version
- `My First Project.v6i.yolov8/` - Post-active-learning (current)

---

## 🌐 Web Deployment

### Hugging Face Spaces

**Live Demo:** [https://huggingface.co/spaces/liangjinwong/fashion-detection-system](https://huggingface.co/spaces/liangjinwong/fashion-detection-system)

### Features
- Upload any fashion image
- Adjustable confidence threshold
- Image enhancement options
- All 3 models (Clothes, Accessories, Shoes)

### Deployment Stack
- **Framework:** Gradio 3.50.2
- **Platform:** Hugging Face Spaces
- **Models:** YOLOv8 + ResNet50 (CPU inference)

### Local Gradio Testing

```bash
cd hf_deploy
python app.py
# Opens at http://localhost:7860
```

---

## 📁 Project Structure

```
AAI3001_P10/
├── app.py                          # Main GUI (Phase 1)
├── app_with_preprocessing.py       # Full GUI (Phase 1+2+3)
├── app_huggingface.py              # Gradio web app
│
├── best_model_resnet50_extended_final.pth  # Phase 1 weights
├── best_model_shoes.pth                    # Shoe classifier weights
├── le_product_type_extended.pkl            # Phase 1 label encoder
├── le_shoes.pkl                            # Shoe label encoder
│
├── runs/
│   └── finetune/
│       ├── phase2_clothes_v6/      # Clothes detector (YOLOv8)
│       │   └── weights/best.pt
│       └── phase3_accessories_v6/  # Accessory detector (YOLOv8)
│           └── weights/best.pt
│
├── hf_deploy/                      # Hugging Face deployment
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   └── models/
│       ├── phase2_clothes_best.pt
│       ├── phase3_accessories_best.pt
│       └── best_model_shoes.pth
│
├── Phase2_DeepFashion2_YOLO_Detection.ipynb
├── Phase2_Fashionpedia_YOLO_Setup.ipynb
├── Phase2_Shoe_Classifier_Training.ipynb
├── Roboflow_FineTune.ipynb         # Active Learning notebook
├── AAI3001_model.ipynb             # Phase 1 training notebook
│
├── My First Project.v6i.yolov8/    # Roboflow dataset (post-AL)
├── deepfashion2_yolo_v2_optimized/ # DeepFashion2 YOLO format
├── fashionpedia_yolo/              # Fashionpedia YOLO format
│
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Highlights

### 1. Multi-Model Pipeline
Three specialized models work together, each optimized for its specific task rather than one general-purpose model.

### 2. Hierarchical Classification
When a "shoe" is detected, the Shoe Classifier provides fine-grained classification (7 types) instead of just "shoe".

### 3. Logic-Based Filtering
- **Gravity Check:** Shoes detected in upper 40% of image are filtered out
- **Skin Detection:** HSV-based filtering removes false positives on exposed skin
- **IoU Filtering:** Removes duplicate detections between overlapping models

### 4. Webcam Enhancement
For low-quality laptop webcams:
- Bilateral filtering for noise reduction
- CLAHE for contrast enhancement
- Sharpening kernels for clarity
- Optional upscaling for small images

### 5. Explainable AI
Grad-CAM visualization shows which regions the model focuses on for classification decisions.

---

## 📈 Future Improvements

1. **Segmentation Integration:** Add instance segmentation for precise clothing boundaries
2. **Try-On Feature:** Virtual try-on using generative models
3. **Mobile Deployment:** TensorFlow Lite / ONNX for mobile apps
4. **More Accessory Classes:** Expand to jewelry, watches, etc.
5. **Outfit Recommendation:** AI-powered styling suggestions

---

## 🙏 Acknowledgments

- **Singapore Institute of Technology** - AAI3001 Course
- **Ultralytics** - YOLOv8 framework
- **Roboflow** - Dataset management and active learning
- **Hugging Face** - Model deployment platform
- **DeepFashion2 & Fashionpedia** - Open datasets

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by AAI3001 Group 10**

*Singapore Institute of Technology • 2025*

</div>
