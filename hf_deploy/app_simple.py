# app.py — Hugging Face Spaces Deployment (Simplified)
# Fashion Intelligence Suite - Web Version
# AAI3001 Group 10

import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import os

# ----------------------
# CONFIGURATION
# ----------------------
PATH_CLOTHES_DETECTOR = "models/phase2_clothes_best.pt"
PATH_ACCESSORY_DETECTOR = "models/phase3_accessories_best.pt"
PATH_SHOE_CLASSIFIER = "models/best_model_shoes.pth"

# Shoe Classes (must match le_shoes.pkl order)
SHOE_CLASSES = [
    "Casual Shoes", "Flats", "Flip Flops", "Formal Shoes", 
    "Heels", "Sandals", "Sports Shoes"
]

# ----------------------
# GLOBAL MODELS
# ----------------------
model_clothes = None
model_accessories = None
model_shoes = None
shoe_transforms = None

# ----------------------
# MODEL LOADING
# ----------------------
def load_models():
    global model_clothes, model_accessories, model_shoes, shoe_transforms
    
    try:
        from ultralytics import YOLO
        
        if os.path.exists(PATH_CLOTHES_DETECTOR):
            model_clothes = YOLO(PATH_CLOTHES_DETECTOR)
            print("✅ Clothes detector loaded")
            
        if os.path.exists(PATH_ACCESSORY_DETECTOR):
            model_accessories = YOLO(PATH_ACCESSORY_DETECTOR)
            print("✅ Accessory detector loaded")
            
    except ImportError:
        print("❌ ultralytics not installed")
    
    # Load Shoe Classifier (ResNet50)
    if os.path.exists(PATH_SHOE_CLASSIFIER):
        try:
            model_shoes = models.resnet50(weights=None)
            model_shoes.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model_shoes.fc.in_features, len(SHOE_CLASSES))
            )
            model_shoes.load_state_dict(torch.load(PATH_SHOE_CLASSIFIER, map_location="cpu"))
            model_shoes.eval()
            
            shoe_transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("✅ Shoe classifier loaded")
        except Exception as e:
            print(f"❌ Shoe classifier error: {e}")

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def get_class_color(cls_name):
    colors = {
        'short sleeve top': '#FF6B6B', 'long sleeve top': '#FF8E8E',
        'short sleeve outwear': '#4ECDC4', 'long sleeve outwear': '#45B7AA',
        'vest': '#FFE66D', 'sling': '#F7DC6F',
        'shorts': '#95E1D3', 'trousers': '#7DCEA0',
        'skirt': '#BB8FCE', 'short sleeve dress': '#D7BDE2',
        'long sleeve dress': '#C39BD3', 'vest dress': '#AF7AC5',
        'sling dress': '#A569BD',
        'jacket': '#5DADE2', 'coat': '#3498DB',
        'glasses': '#F1948A', 'hat': '#85929E',
        'tie': '#1ABC9C', 'watch': '#D4AC0D',
        'belt': '#6E2C00', 'sock': '#FADBD8',
        'shoe': '#2C3E50', 'bag': '#E74C3C', 'scarf': '#9B59B6',
    }
    for key in colors:
        if key in cls_name.lower():
            return colors[key]
    shoe_colors = {
        'Casual Shoes': '#3498DB', 'Flats': '#E91E63',
        'Flip Flops': '#00BCD4', 'Formal Shoes': '#2C3E50',
        'Heels': '#9C27B0', 'Sandals': '#FF9800', 'Sports Shoes': '#4CAF50'
    }
    return shoe_colors.get(cls_name, '#808080')

def classify_shoe(pil_crop):
    if model_shoes is None or shoe_transforms is None:
        return "Shoe", 0.0
    try:
        tensor = shoe_transforms(pil_crop.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            out = model_shoes(tensor)
            probs = torch.softmax(out, dim=1)
            conf, idx = probs.max(1)
            return SHOE_CLASSES[idx.item()], conf.item()
    except:
        return "Shoe", 0.0

def is_mostly_skin(pil_crop, threshold=0.5):
    try:
        arr = np.array(pil_crop.convert("RGB"))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 20, 70])
        upper = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        ratio = np.sum(mask > 0) / mask.size
        return ratio > threshold
    except:
        return False

# ----------------------
# MAIN DETECTION
# ----------------------
def detect_fashion(image, confidence):
    if image is None:
        return None, "Please upload an image first."
    
    # Load models on first call
    if model_clothes is None and model_accessories is None:
        load_models()
    
    pil_img = image.copy()
    img_w, img_h = pil_img.size
    detections = []
    
    # 1. Clothes Detection
    if model_clothes:
        results = model_clothes(pil_img, verbose=False, conf=confidence)[0]
        for box in results.boxes:
            cls = model_clothes.names[int(box.cls)]
            conf_val = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            label = cls
            if cls == 'shoe' and model_shoes:
                pad = 10
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(img_w, x2+pad), min(img_h, y2+pad)
                if cx2 > cx1 and cy2 > cy1:
                    shoe_crop = pil_img.crop((cx1, cy1, cx2, cy2))
                    s_type, s_conf = classify_shoe(shoe_crop)
                    label = s_type
            
            detections.append({
                "label": label,
                "confidence": conf_val,
                "box": (x1, y1, x2, y2),
                "color": get_class_color(label)
            })
    
    # 2. Accessory Detection
    if model_accessories:
        results = model_accessories(pil_img, verbose=False, conf=confidence)[0]
        for box in results.boxes:
            cls = model_accessories.names[int(box.cls)]
            conf_val = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            label = cls
            if cls == 'shoe' and model_shoes:
                pad = 10
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(img_w, x2+pad), min(img_h, y2+pad)
                if cx2 > cx1 and cy2 > cy1:
                    shoe_crop = pil_img.crop((cx1, cy1, cx2, cy2))
                    s_type, s_conf = classify_shoe(shoe_crop)
                    label = s_type
            
            detections.append({
                "label": label,
                "confidence": conf_val,
                "box": (x1, y1, x2, y2),
                "color": get_class_color(label)
            })
    
    # Draw detections
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    detection_text = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = det["color"]
        label = f"{det['label']} ({det['confidence']:.0%})"
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        try:
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except:
            text_w, text_h = 100, 16
        
        draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - text_h - 3), label, fill="white", font=font)
        
        detection_text.append(f"• {det['label']}: {det['confidence']:.1%}")
    
    if detections:
        summary = f"Found {len(detections)} items:\n" + "\n".join(detection_text)
    else:
        summary = "No items detected. Try lowering the confidence threshold."
    
    return pil_img, summary

# ----------------------
# GRADIO INTERFACE (Simple)
# ----------------------
print("Loading models...")
load_models()

# Use simple Interface instead of Blocks
demo = gr.Interface(
    fn=detect_fashion,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(type="pil", label="Detection Results"),
        gr.Textbox(label="Detected Items")
    ],
    title="👗 Fashion Intelligence Suite",
    description="AAI3001 Group 10 - Upload an image to detect clothing, accessories, and shoes.",
    examples=[],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
