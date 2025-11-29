# app_huggingface.py — Hugging Face Spaces Deployment
# Fashion Intelligence Suite - Web Version
# Deploy to: https://huggingface.co/spaces

import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import os
import pickle

# ----------------------
# CONFIGURATION
# ----------------------
# Model paths (these will be in the HF Space repo)
PATH_CLOTHES_DETECTOR = "models/phase2_clothes_best.pt"
PATH_ACCESSORY_DETECTOR = "models/phase3_accessories_best.pt"
PATH_SHOE_CLASSIFIER = "models/best_model_shoes.pth"
PATH_SHOE_ENCODER = "models/le_shoes.pkl"

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
        
        # Load YOLO models
        if os.path.exists(PATH_CLOTHES_DETECTOR):
            model_clothes = YOLO(PATH_CLOTHES_DETECTOR)
            print("✅ Clothes detector loaded")
        else:
            print(f"⚠️ Clothes detector not found at {PATH_CLOTHES_DETECTOR}")
            
        if os.path.exists(PATH_ACCESSORY_DETECTOR):
            model_accessories = YOLO(PATH_ACCESSORY_DETECTOR)
            print("✅ Accessory detector loaded")
        else:
            print(f"⚠️ Accessory detector not found at {PATH_ACCESSORY_DETECTOR}")
            
    except ImportError:
        print("❌ ultralytics not installed")
    
    # Load Shoe Classifier (ResNet50)
    if os.path.exists(PATH_SHOE_CLASSIFIER):
        try:
            model_shoes = models.resnet50(weights=None)
            model_shoes.fc = nn.Linear(model_shoes.fc.in_features, len(SHOE_CLASSES))
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
    else:
        print(f"⚠️ Shoe classifier not found at {PATH_SHOE_CLASSIFIER}")

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def enhance_image(pil_img, mode=1):
    """Enhance low-quality images for better detection."""
    if mode == 0:
        return pil_img
    
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    if mode >= 1:
        # Denoise + mild sharpen
        img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
        kernel_sharpen = np.array([[-0.5, -0.5, -0.5],
                                   [-0.5,  5.0, -0.5],
                                   [-0.5, -0.5, -0.5]]) / 2.0
        img = cv2.filter2D(img, -1, kernel_sharpen)
    
    if mode >= 2:
        # Upscale if small
        if w < 640 or h < 480:
            scale = 1.5
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # CLAHE contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def is_mostly_skin(pil_crop):
    """Check if crop is mostly skin color."""
    try:
        img = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower1 = np.array([0, 15, 60], dtype=np.uint8)
        upper1 = np.array([25, 170, 255], dtype=np.uint8)
        lower2 = np.array([0, 10, 80], dtype=np.uint8)
        upper2 = np.array([20, 150, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
        return ratio > 0.35
    except:
        return False

def classify_shoe(pil_crop):
    """Classify shoe type using ResNet."""
    if model_shoes is None:
        return "Shoe", 0.0
    
    try:
        img_t = shoe_transforms(pil_crop).unsqueeze(0)
        with torch.no_grad():
            outputs = model_shoes(img_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        idx = pred.item()
        label = SHOE_CLASSES[idx] if idx < len(SHOE_CLASSES) else f"Class_{idx}"
        return label, conf.item()
    except:
        return "Shoe", 0.0

def get_class_color(cls_name):
    """Get consistent color for each class category."""
    cls_lower = cls_name.lower()
    
    # Clothes = Green
    if any(x in cls_lower for x in ['top', 'dress', 'vest', 'shorts', 'trousers', 'skirt', 'sling', 'outwear']):
        return (34, 197, 94)  # Green
    # Outerwear = Blue
    elif any(x in cls_lower for x in ['jacket', 'coat']):
        return (59, 130, 246)  # Blue
    # Shoes = Red
    elif any(x in cls_lower for x in ['shoe', 'heel', 'sandal', 'flip', 'boot', 'flats']):
        return (239, 68, 68)  # Red
    # Accessories = Orange
    elif any(x in cls_lower for x in ['bag', 'hat', 'glass', 'watch', 'belt', 'tie', 'scarf']):
        return (249, 115, 22)  # Orange
    else:
        return (168, 162, 158)  # Gray

# ----------------------
# MAIN DETECTION FUNCTION
# ----------------------
def detect_fashion(image, confidence=0.5, enhance_mode="Off"):
    """Main detection function for Gradio interface."""
    if image is None:
        return None, "Please upload an image."
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image.copy()
    
    # Apply enhancement
    enhance_level = {"Off": 0, "Light": 1, "Strong": 2}.get(enhance_mode, 0)
    if enhance_level > 0:
        pil_img = enhance_image(pil_img, enhance_level)
    
    img_w, img_h = pil_img.size
    detections = []
    
    # 1. Clothes Detection (Phase 2)
    if model_clothes:
        results = model_clothes(pil_img, verbose=False, conf=confidence)[0]
        for box in results.boxes:
            cls = model_clothes.names[int(box.cls)]
            conf_val = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Skip skin false positives for vests/tops
            if cls in ['vest', 'short_sleeve_top', 'long_sleeve_top']:
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(img_w, x2), min(img_h, y2)
                if cx2 > cx1 and cy2 > cy1:
                    crop = pil_img.crop((cx1, cy1, cx2, cy2))
                    if is_mostly_skin(crop):
                        continue
            
            # Shoe specialist
            label = cls
            if cls == 'shoe' and model_shoes:
                pad = 10
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(img_w, x2+pad), min(img_h, y2+pad)
                if cx2 > cx1 and cy2 > cy1:
                    shoe_crop = pil_img.crop((cx1, cy1, cx2, cy2))
                    s_type, s_conf = classify_shoe(shoe_crop)
                    label = f"{s_type}"
            
            detections.append({
                "label": label,
                "confidence": conf_val,
                "box": (x1, y1, x2, y2),
                "color": get_class_color(label)
            })
    
    # 2. Accessory Detection (Phase 3)
    if model_accessories:
        results = model_accessories(pil_img, verbose=False, conf=confidence)[0]
        for box in results.boxes:
            cls = model_accessories.names[int(box.cls)]
            conf_val = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Gravity check for shoes
            if cls in ['shoe', 'heel', 'sandal', 'flip flop', 'boot']:
                y_center = (y1 + y2) / 2
                if y_center < img_h * 0.4:
                    continue
            
            # Shoe specialist
            label = cls
            if cls == 'shoe' and model_shoes:
                pad = 10
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(img_w, x2+pad), min(img_h, y2+pad)
                if cx2 > cx1 and cy2 > cy1:
                    shoe_crop = pil_img.crop((cx1, cy1, cx2, cy2))
                    s_type, s_conf = classify_shoe(shoe_crop)
                    label = f"{s_type}"
            
            detections.append({
                "label": label,
                "confidence": conf_val,
                "box": (x1, y1, x2, y2),
                "color": get_class_color(label)
            })
    
    # Draw detections on image
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    detection_text = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = det["color"]
        label = f"{det['label']} ({det['confidence']:.0%})"
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), label, fill="white", font=font)
        
        detection_text.append(f"• {det['label']}: {det['confidence']:.1%}")
    
    # Generate summary
    if detections:
        summary = f"**Detected {len(detections)} items:**\n" + "\n".join(detection_text)
    else:
        summary = "No fashion items detected. Try:\n• Lowering the confidence threshold\n• Using image enhancement\n• Uploading a clearer image"
    
    return pil_img, summary

# ----------------------
# GRADIO INTERFACE
# ----------------------
def create_interface():
    """Create the Gradio web interface."""
    
    # Load models on startup
    load_models()
    
    with gr.Blocks(title="Fashion Intelligence Suite") as demo:
        gr.Markdown("""
        # 👗 Fashion Intelligence Suite
        ### AAI3001 Group 10 - Multi-Model Fashion Detection System
        
        Upload an image to detect clothing, accessories, and shoes using our specialized AI models:
        - **Phase 2**: Clothes Detector (YOLOv8 fine-tuned on DeepFashion2)
        - **Phase 3**: Accessory Detector (YOLOv8 fine-tuned on Fashionpedia)
        - **Shoe Specialist**: ResNet50 classifier for 7 shoe types
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Image", type="pil")
                
                with gr.Row():
                    confidence_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="Confidence Threshold"
                    )
                    enhance_dropdown = gr.Dropdown(
                        choices=["Off", "Light", "Strong"],
                        value="Off",
                        label="Image Enhancement"
                    )
                
                detect_btn = gr.Button("🔍 Detect Fashion Items", variant="primary")
                
                gr.Markdown("""
                **Tips:**
                - Lower confidence (30-40%) for more detections
                - Use "Light" enhancement for low-quality images
                - Use "Strong" enhancement for very grainy/small images
                """)
            
            with gr.Column(scale=1):
                output_image = gr.Image(label="Detection Results", type="pil")
                output_text = gr.Markdown(label="Detected Items")
        
        # Examples
        gr.Markdown("### 📸 Example Images")
        gr.Examples(
            examples=[
                ["examples/person1.jpg", 0.5, "Off"],
                ["examples/person2.jpg", 0.4, "Light"],
            ],
            inputs=[input_image, confidence_slider, enhance_dropdown],
            outputs=[output_image, output_text],
            fn=detect_fashion,
            cache_examples=False
        )
        
        # Connect button
        detect_btn.click(
            fn=detect_fashion,
            inputs=[input_image, confidence_slider, enhance_dropdown],
            outputs=[output_image, output_text]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Model Performance:**
        | Model | Dataset | Classes | mAP50/Accuracy |
        |-------|---------|---------|----------------|
        | Clothes YOLO | DeepFashion2 | 13 | 0.80 |
        | Accessory YOLO | Fashionpedia | 11 | 0.75 |
        | Shoe Classifier | Fashion DS | 7 | 82.5% |
        
        *Built by AAI3001 Group 10*
        """)
    
    return demo

# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
