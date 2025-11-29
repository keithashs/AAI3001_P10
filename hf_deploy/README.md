---
title: Fashion Intelligence Suite
emoji: 👗
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: mit
short_description: AI fashion detection for clothes & accessories
---

# 👗 Fashion Intelligence Suite

A multi-model AI system for detecting and classifying fashion items in images.

**Built by AAI3001 Group 10 — Singapore Institute of Technology**

## 🎯 Features

- **Multi-Model Pipeline**: Combines 3 specialized models for comprehensive fashion detection
- **Real-Time Detection**: Fast inference suitable for live applications
- **Image Enhancement**: Built-in preprocessing for low-quality images
- **Smart Post-Processing**: Logic-based filtering to reduce false positives

## 🤖 Models

| Model | Architecture | Dataset | Classes | Performance |
|-------|--------------|---------|---------|-------------|
| Clothes Detector | YOLOv8s | DeepFashion2 | 13 | mAP50: 0.80 |
| Accessory Detector | YOLOv8s | Fashionpedia | 11 | mAP50: 0.75 |
| Shoe Classifier | ResNet50 | Fashion Dataset | 7 | Accuracy: 82.5% |

## 👕 Classes Detected

### Clothing (13 classes)
Short Sleeve Top, Long Sleeve Top, Short Sleeve Dress, Short Sleeve Outwear, Long Sleeve Dress, Vest Dress, Sling, Sling Dress, Shorts, Trousers, Skirt, Long Sleeve Outwear, Vest

### Accessories (11 classes)
Jacket, Coat, Glasses, Hat, Tie, Watch, Belt, Sock, Shoe, Bag, Scarf

### Shoe Types (7 classes)
Casual Shoes, Flats, Flip Flops, Formal Shoes, Heels, Sandals, Sports Shoes

## 🚀 Usage

1. Upload an image containing a person wearing clothes/accessories
2. Adjust the confidence threshold (lower = more detections)
3. Enable image enhancement for low-quality photos
4. Click "Detect Fashion Items"

## 📝 Tips

- **Confidence 30-40%**: More detections, may include false positives
- **Confidence 50-60%**: Balanced detection (recommended)
- **Confidence 70%+**: High precision, fewer detections
- **Light Enhancement**: For slightly blurry webcam images
- **Strong Enhancement**: For grainy/low-resolution images

## 🔧 Technical Stack

- **Detection**: Ultralytics YOLOv8
- **Classification**: PyTorch + torchvision (ResNet50)
- **Interface**: Gradio
- **Image Processing**: OpenCV, Pillow

## 📚 Training Data

- **DeepFashion2**: 50,000+ fashion images with clothing annotations
- **Fashionpedia**: 45,000+ images with accessory annotations
- **Roboflow v6**: 441 curated images for fine-tuning (Active Learning)

## 👥 Team

AAI3001 Group 10 — Singapore Institute of Technology

## 📄 License

MIT License
