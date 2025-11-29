# app_with_preprocessing.py — Phase 1 (Classification) + Phase 2/3 (Detection & Vibe Check)
# Requires: pip install tkinterdnd2 pillow torch torchvision matplotlib ultralytics opencv-python

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except Exception:
    HAS_DND = False

from PIL import Image, ImageTk, ImageFilter, ImageDraw, ImageFont
import pickle, os, io, time
from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# For color mapping the CAM without OpenCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

# For smart preprocessing
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


# ----------------------
# CONFIGURATION & PATHS
# ----------------------
# Phase 1: ResNet50 Classifier
PATH_PHASE1_WEIGHTS = r"d:/AAI3001/best_model_resnet50_extended_final.pth"
PATH_PHASE1_ENCODER = r"d:/AAI3001/le_product_type_extended.pkl"

# Phase 2: Clothes Detector (YOLO)
PATH_CLOTHES_DETECTOR = r"d:/AAI3001/runs/finetune/phase2_clothes_v6/weights/best.pt"

# Phase 3: Accessory Detector (YOLO)
PATH_ACCESSORY_DETECTOR = r"d:/AAI3001/runs/finetune/phase3_accessories_v6/weights/best.pt"

# Phase 3: Shoe Specialist (ResNet)
PATH_SHOE_CLASSIFIER = r"d:/AAI3001/best_model_shoes.pth"

# Shoe Classes (MUST match le_shoes.pkl order!)
SHOE_CLASSES = [
    "Casual Shoes", "Flats", "Flip Flops", "Formal Shoes", 
    "Heels", "Sandals", "Sports Shoes"
]

# ----------------------
# GLOBAL STATE
# ----------------------
# Models
model_phase1 = None
le_phase1 = None
model_clothes = None
model_accessories = None
model_shoes = None
shoe_transforms = None

# UI State
root = None
notebook = None
tab1 = None
tab2 = None

# Tab 1 State (Phase 1)
t1_orig_img = None
t1_display_img = None
t1_display_tk = None
t1_crop_rect_id = None
t1_crop_start = None
t1_crop_end = None
t1_active_roi = None

# Tab 2 State (Phase 2/3)
t2_cap = None
t2_is_running = False
t2_display_tk = None
t2_mode = "Webcam" # or "Image"
t2_uploaded_img = None
t2_shoe_cache = {}
t2_frame_count = 0
t2_conf_threshold = 0.5 # Default confidence
t2_enhance_mode = 0 # 0=Off, 1=Light, 2=Strong (for low-quality webcams)

# ----------------------
# MODEL LOADING
# ----------------------
def load_phase1_models():
    global model_phase1, le_phase1
    if not os.path.exists(PATH_PHASE1_ENCODER) or not os.path.exists(PATH_PHASE1_WEIGHTS):
        print("❌ Phase 1 models missing.")
        return False

    with open(PATH_PHASE1_ENCODER, "rb") as f:
        le_phase1 = pickle.load(f)

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, len(le_phase1.classes_))
    )
    state = torch.load(PATH_PHASE1_WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model_phase1 = model
    print("✅ Phase 1 (ResNet50) Loaded")
    return True

def load_detection_models():
    global model_clothes, model_accessories, model_shoes, shoe_transforms
    
    # 1. Clothes YOLO
    if os.path.exists(PATH_CLOTHES_DETECTOR) and HAS_YOLO:
        model_clothes = YOLO(PATH_CLOTHES_DETECTOR)
        print("✅ Phase 2 (Clothes YOLO) Loaded")
    
    # 2. Accessories YOLO
    if os.path.exists(PATH_ACCESSORY_DETECTOR) and HAS_YOLO:
        model_accessories = YOLO(PATH_ACCESSORY_DETECTOR)
        print("✅ Phase 3 (Accessory YOLO) Loaded")
        
    # 3. Shoe ResNet
    if os.path.exists(PATH_SHOE_CLASSIFIER):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load state dict
        state_dict = torch.load(PATH_SHOE_CLASSIFIER, map_location=device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        # Determine classes
        num_classes = len(SHOE_CLASSES)
        if 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].shape[0]
        elif 'model.fc.weight' in state_dict:
             num_classes = state_dict['model.fc.weight'].shape[0]
        
        # Init Model
        m = models.resnet50(pretrained=False)
        m.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )
        
        # Load weights safely
        try:
            m.load_state_dict(state_dict)
        except:
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            m.load_state_dict(new_state_dict, strict=False)
            
        m.eval()
        m.to(device)
        model_shoes = m
        print("✅ Phase 3 (Shoe ResNet) Loaded")
        
        # Transforms
        shoe_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# ----------------------
# PHASE 1 LOGIC
# ----------------------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform_eval = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean, std),
])

def classify_phase1(pil_img, topk=5):
    if model_phase1 is None: return [], -1
    x = transform_eval(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model_phase1(x)
        probs = torch.softmax(logits, dim=1)[0]
    
    k = min(int(topk), probs.numel())
    conf, idx = torch.topk(probs, k=k)
    labels = [le_phase1.classes_[i] for i in idx.cpu().numpy().tolist()]
    conf = (conf.cpu().numpy() * 100.0).tolist()
    return list(zip(labels, conf)), int(idx[0].item())

def gradcam_phase1(pil_img, target_class=None):
    if model_phase1 is None: return pil_img
    x = transform_eval(pil_img).unsqueeze(0)
    target_layer = model_phase1.layer4[-1].conv2
    
    activations = []; gradients = []
    def fwd_hook(module, inp, out): activations.append(out.detach())
    def bwd_hook(module, grad_in, grad_out): gradients.append(grad_out[0].detach())
    
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    
    logits = model_phase1(x)
    if target_class is None: target_class = int(torch.argmax(logits, dim=1).item())
    
    model_phase1.zero_grad()
    logits[0, target_class].backward()
    
    act = activations[0][0]
    grad = gradients[0][0]
    weights = grad.mean(dim=(1, 2))
    cam = torch.sum(weights[:, None, None] * act, dim=0)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0: cam = cam / cam.max()
    
    cam_np = cam.cpu().numpy()
    cam_img = Image.fromarray(np.uint8(cam_np * 255), mode="L").resize(pil_img.size, resample=Image.BILINEAR)
    heatmap = cm.get_cmap("jet")(np.array(cam_img) / 255.0)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap)
    overlay = Image.blend(pil_img.convert("RGB"), heatmap_pil, alpha=0.45)
    
    h1.remove(); h2.remove()
    return overlay

# ----------------------
# PHASE 2/3 LOGIC
# ----------------------
def classify_shoe_crop(pil_img):
    if model_shoes is None: return "Shoe", 0.0
    device = next(model_shoes.parameters()).device
    img_t = shoe_transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model_shoes(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    
    idx = pred.item()
    label = SHOE_CLASSES[idx] if idx < len(SHOE_CLASSES) else f"Class_{idx}"
    return label, conf.item()

def is_mostly_skin(pil_crop):
    """Checks if a crop is mostly skin color (simple HSV heuristic)."""
    try:
        # Convert PIL -> CV2 (BGR)
        img = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Generic Skin Range (covers most skin tones under indoor lighting)
        # H: 0-25 (Red/Orange), S: 15-170, V: 60-255
        lower1 = np.array([0, 15, 60], dtype=np.uint8)
        upper1 = np.array([25, 170, 255], dtype=np.uint8)
        
        # Extended range for different lighting conditions
        # H: 0-20 with lower saturation (pale skin under bright light)
        lower2 = np.array([0, 10, 80], dtype=np.uint8)
        upper2 = np.array([20, 150, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Calculate ratio
        ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
        return ratio > 0.35  # Lowered threshold: If > 35% is skin-like, it's likely bare skin
    except:
        return False

def enhance_webcam_frame(pil_img, mode=1):
    """
    Enhance low-quality webcam frames for better detection.
    
    Mode 0: Off (no enhancement)
    Mode 1: Light (denoise + mild sharpen) - for average webcams
    Mode 2: Strong (upscale + denoise + sharpen + contrast) - for poor webcams
    
    This helps YOLO detect objects in:
    - Low resolution webcams (720p or less)
    - Noisy/grainy laptop cameras
    - Poor lighting conditions
    """
    if mode == 0:
        return pil_img
    
    # Convert PIL -> CV2 (BGR)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    if mode >= 1:
        # --- LIGHT ENHANCEMENT ---
        # 1. Denoise: Remove webcam grain/noise
        #    FastNlMeansDenoisingColored is slow, use bilateralFilter for real-time
        img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
        
        # 2. Mild Sharpening: Improve edge clarity for detection
        kernel_sharpen = np.array([[-0.5, -0.5, -0.5],
                                   [-0.5,  5.0, -0.5],
                                   [-0.5, -0.5, -0.5]]) / 2.0
        img = cv2.filter2D(img, -1, kernel_sharpen)
    
    if mode >= 2:
        # --- STRONG ENHANCEMENT ---
        # 3. Upscale: If resolution is low, upscale 1.5x for better feature detection
        if w < 1280 or h < 720:
            scale = 1.5
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 4. Contrast Enhancement: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        #    This helps in poor lighting conditions
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 5. Additional edge sharpening for fine details
        kernel_edge = np.array([[0, -1, 0],
                                [-1,  5, -1],
                                [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel_edge * 0.3 + np.eye(3) * 0.7)
    
    # Convert back to PIL (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def process_detection_frame(pil_img, is_video=False):
    """Runs unified detection using the fine-tuned model."""
    global t2_frame_count, t2_shoe_cache, t2_conf_threshold
    
    if is_video:
        t2_frame_count += 1
    else:
        t2_frame_count = 0 # Always run full check on static image
        t2_shoe_cache = {}

    draw = ImageDraw.Draw(pil_img)
    img_w, img_h = pil_img.size
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    detections = [] # (label, conf, box, color)
    
    # Define which classes are "Clothes" vs "Accessories" for color coding
    CLOTHES_CLASSES = {'short_sleeve_top', 'long_sleeve_top', 'shorts', 'trousers', 'vest', 
                       'short_sleeve_dress', 'long_sleeve_dress', 'vest_dress', 'sling_dress', 'skirt'}
    ACCESSORY_CLASSES = {'jacket', 'coat', 'glasses', 'bag', 'belt', 'sock', 'tie', 'watch',
                         'Casual Shoes', 'Flip Flops', 'Formal Shoes', 'Sandals', 'Sports Shoes'}

    # ========== UNIFIED DETECTION ==========
    # Since both models now have identical 18 classes (after Roboflow fine-tuning),
    # we run BOTH and merge their results, keeping the highest confidence for each object.
    
    all_raw_detections = []  # (cls, conf, box, source)
    
    # 1. Run Phase 2 Model (Clothes)
    if model_clothes:
        results = model_clothes(pil_img, verbose=False, conf=t2_conf_threshold)[0]
        for box in results.boxes:
            cls = model_clothes.names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            all_raw_detections.append((cls, conf, (x1, y1, x2, y2), "P2"))
    
    # 2. Run Phase 3 Model (Accessories)  
    if model_accessories:
        results = model_accessories(pil_img, verbose=False, conf=t2_conf_threshold)[0]
        for box in results.boxes:
            cls = model_accessories.names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            all_raw_detections.append((cls, conf, (x1, y1, x2, y2), "P3"))
    
    # 3. Merge detections - if two boxes overlap significantly, keep the higher confidence one
    merged_detections = []
    used_indices = set()
    
    # Sort by confidence (highest first) so we prioritize high-confidence detections
    sorted_dets = sorted(enumerate(all_raw_detections), key=lambda x: x[1][1], reverse=True)
    
    for idx, (cls, conf, box, source) in sorted_dets:
        if idx in used_indices:
            continue
        
        # Check if this box significantly overlaps with any already-kept detection
        dominated = False
        for kept_cls, kept_conf, kept_box, kept_color in merged_detections:
            iou = calculate_iou(box, kept_box)
            if iou > 0.4:  # Significant overlap
                dominated = True
                break
        
        if not dominated:
            # Mark all overlapping lower-confidence boxes as used
            for other_idx, (other_cls, other_conf, other_box, other_source) in sorted_dets:
                if other_idx != idx and other_idx not in used_indices:
                    iou = calculate_iou(box, other_box)
                    if iou > 0.4:
                        used_indices.add(other_idx)
            
            # Assign color based on class category
            if cls in CLOTHES_CLASSES:
                color = "#22c55e"  # Green for clothes
            elif cls == 'shoe':
                color = "#ef4444"  # Red for shoes (will be specialized later)
            elif cls in ['jacket', 'coat']:
                color = "#3b82f6"  # Blue for outerwear
            elif cls in ACCESSORY_CLASSES:
                color = "#f59e0b"  # Orange for accessories
            else:
                color = "#f59e0b"  # Orange for unknown/other
            
            merged_detections.append((cls, conf, box, color))
    
    # 4. Apply sanity filters to merged detections
    for cls, conf, (x1, y1, x2, y2), color in merged_detections:
            
            # --- OPTION A: No strict filter - show all detections ---
            # (Filters removed to diagnose detection issues)
            
            # --- LOGIC CHECK: Trousers Aspect Ratio & Skin Check ---
            # Trousers are usually vertical (tall). If width > height, it's likely a folded object or mistake.
            if cls == 'trousers':
                w = x2 - x1
                h = y2 - y1
                
                # 1. Aspect Ratio Check
                if w > h * 1.2: # Allow slightly wider than tall, but not landscape
                    continue
                
                # 2. Skin Check (Fix for "Knees detected as Trousers")
                # If the "trousers" area is mostly skin, it's likely bare legs (shorts).
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(img_w, x2), min(img_h, y2)
                if cx2 > cx1 and cy2 > cy1:
                    crop = pil_img.crop((cx1, cy1, cx2, cy2))
                    if is_mostly_skin(crop):
                        continue # It's bare legs, not trousers
            # ------------------------------------------

            # --- LOGIC CHECK: Vest/Skin False Positives ---
            # CRITICAL: Vest is the #1 false positive on bare skin.
            # Apply skin check REGARDLESS of confidence (even 0.97 can be wrong)
            if cls == 'vest':
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(img_w, x2), min(img_h, y2)
                if cx2 > cx1 and cy2 > cy1:
                    crop = pil_img.crop((cx1, cy1, cx2, cy2))
                    if is_mostly_skin(crop):
                        continue  # It's bare skin, not a vest
                
                # Also require higher confidence for vest
                if conf < (t2_conf_threshold + 0.20):
                    continue

            # 2. Skin Color Check: If a "Top" is mostly skin color, it's likely bare skin.
            if cls in ['short_sleeve_top', 'long_sleeve_top', 'sling_dress']:
                # Crop the detected area
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(img_w, x2), min(img_h, y2)
                if cx2 > cx1 and cy2 > cy1:
                    crop = pil_img.crop((cx1, cy1, cx2, cy2))
                    if is_mostly_skin(crop):
                        continue
            # ----------------------------------------------
            
            # --- SHOE SPECIALIST ---
            # If we detect a generic "shoe", run the ResNet classifier to get the specific type
            label_text = cls
            if cls == 'shoe' and model_shoes:
                color = "#ef4444"  # Red for shoes
                
                # Optimization for video: only run heavy classifier every 5 frames
                should_run = not is_video or (t2_frame_count % 5 == 0)
                
                # Cache key based on approximate position (grid of 20px)
                key = (x1//20, y1//20)
                
                if should_run:
                    # Crop
                    pad = 10
                    cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                    cx2, cy2 = min(img_w, x2+pad), min(img_h, y2+pad)
                    
                    if cx2 > cx1 and cy2 > cy1:
                        shoe_crop = pil_img.crop((cx1, cy1, cx2, cy2))
                        s_type, s_conf = classify_shoe_crop(shoe_crop)
                        t2_shoe_cache[key] = (s_type, s_conf)
                        label_text = f"{s_type} ({s_conf:.2f})"
                elif key in t2_shoe_cache:
                    s_type, s_conf = t2_shoe_cache[key]
                    label_text = f"{s_type} ({s_conf:.2f})"
                else:
                    label_text = "Shoe (...)"
            # -----------------------
            
            detections.append((label_text, conf, (x1, y1, x2, y2), color))

    # 2. Accessory Detection (Phase 3 - The "Fashion Item" Expert)
    if model_accessories:
        results = model_accessories(pil_img, verbose=False, conf=t2_conf_threshold)[0]
        # Debug print to help understand why detections might be missing
        if len(results.boxes) == 0:
            print(f"DEBUG: No accessories detected at confidence {t2_conf_threshold}")
        
        for box in results.boxes:
            cls = model_accessories.names[int(box.cls)]
            conf = float(box.conf)
            print(f"DEBUG: Raw Detection - {cls} ({conf:.2f})")
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # --- OPTION A: No strict filter - show all detections ---
            # (Filters removed to diagnose detection issues)
            
            # --- LOGIC CHECK: Jacket/Coat False Positives ---
            # Phase 3 often sees "Jacket" on t-shirts or bare skin.
            if cls in ['jacket', 'coat']:
                # 1. Size Check: Jackets are big. If it's a small box (e.g. just the neck), ignore it.
                h = y2 - y1
                if h < img_h * 0.3: # Less than 30% of screen height
                    continue
                
                # 2. Confidence Penalty: Jackets are hard to distinguish from tops.
                # Require 20% higher confidence.
                if conf < (t2_conf_threshold + 0.20):
                    continue
            # ------------------------------------------------

            # --- LOGIC CHECK: Chair/Background Noise ---
            # Chairs often get detected as Backpacks (backrest) or Handbags (headrest).
            
            # Unified Chair Filter:
            # If an object is in the "Chair Zone" (Upper Middle) and looks like a container, it's likely a chair.
            if cls in ['backpack', 'suitcase', 'handbag', 'bag', 'purse']:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Zone: Middle 40% of screen width, Top 60% of screen height
                # This is where a chair back/headrest sits relative to the camera.
                is_centered = abs(cx - img_w/2) < img_w * 0.20 
                is_high = cy < img_h * 0.6
                
                if is_centered and is_high:
                    # It's in the chair spot. Be extremely skeptical.
                    
                    # 1. Backpacks/Suitcases: Almost always chair backs in webcam view.
                    if cls in ['backpack', 'suitcase']:
                        if conf < 0.95: continue # Virtually impossible to be real
                        
                    # 2. Bags/Handbags: Likely headrests or upper back.
                    if cls in ['bag', 'handbag', 'purse']:
                        # If it's wide (headrest shape), it's definitely a chair.
                        w = x2 - x1
                        h = y2 - y1
                        if w > h * 1.1:
                            continue
                            
                        # Even if not wide, if it's in this zone, require 90% confidence.
                        # (User reported false positives around 0.82)
                        if conf < 0.90:
                            continue
            # -------------------------------------------

            # --- LOGIC CHECK: Small Accessory Size & Shape Limits ---
            # Watches, belts, ties are SMALL. Glasses are small AND wide.
            if cls in ['watch', 'belt', 'tie']:
                box_area = (x2 - x1) * (y2 - y1)
                img_area = img_w * img_h
                # These items should be < 3% of screen area
                if box_area > img_area * 0.03:
                    continue
                    
            # Glasses: Must be small, wide (landscape), and near the top of the frame
            if cls == 'glasses':
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                img_area = img_w * img_h
                
                # 1. Size check: < 8% of screen
                if box_area > img_area * 0.08:
                    continue
                    
                # 2. Aspect ratio: Glasses are WIDE (width > height * 1.3)
                # A phone held up would be taller than wide
                if box_w < box_h * 1.2:
                    continue
                    
                # 3. Position: Glasses should be in the upper portion of the frame
                y_center = (y1 + y2) / 2
                if y_center > img_h * 0.5:  # Below the middle = unlikely to be glasses
                    continue
            # ------------------------------------------------

            # --- LOGIC CHECK: Shoe Gravity & Size ---
            # Shoes/Flip-flops shouldn't be floating in the top half of the image
            if cls in ['shoe', 'heel', 'sandal', 'flip flop', 'boot']: # Add other shoe synonyms if needed
                # Only apply physics constraints to live video (assumes full body view)
                # If uploading a static image, shoes might be the main subject (close-up/centered).
                if is_video:
                    y_center = (y1 + y2) / 2
                    # 1. Gravity: Must be in lower 60%
                    if y_center < img_h * 0.4: 
                        continue
                    
                    # 2. Size: Shoes shouldn't be huge (unless close up)
                    box_area = (x2 - x1) * (y2 - y1)
                    img_area = img_w * img_h
                    if box_area > img_area * 0.15: # If > 15% of screen, probably a bag or mistake
                        continue
            # ---------------------------------
            
            # --- CONFLICT RESOLUTION ---
            # Problem: Phase 3 doesn't know "Vest" or "Top", so it calls them "Jacket".
            # Fix: If Phase 3 sees a "Jacket", but Phase 2 sees a "Vest/Top" in the same spot,
            # suppress the Phase 3 "Jacket" label because Phase 2 is the expert on tops.
            # UPDATE: User reported Jackets being misclassified as Tops. 
            # Since both models are now fine-tuned, we should trust the "Jacket" detection if it exists.
            # Disabling this suppression to allow Jackets to appear.
            # if cls in ['jacket', 'coat']:
            #     conflict_found = False
            #     for p2 in phase2_boxes:
            #         # Check if Phase 2 found a light top here
            #         if p2['cls'] in ['short_sleeve_top', 'long_sleeve_top', 'vest', 'sling_dress', 'vest_dress']:
            #             iou = calculate_iou((x1, y1, x2, y2), p2['box'])
            #             if iou > 0.3: # Significant overlap
            #                 conflict_found = True
            #                 break
            #     if conflict_found:
            #         continue # Skip adding this incorrect "Jacket" label
            # ---------------------------
            
            label_text = cls
            color = "#f59e0b" # Orange/Amber for accessories
            
            # 3. Shoe Specialist
            if cls == 'shoe' and model_shoes:
                color = "#ef4444" # Red for shoes
                
                # Optimization for video: only run heavy classifier every 5 frames
                should_run = not is_video or (t2_frame_count % 5 == 0)
                
                # Cache key based on approximate position (grid of 20px)
                key = (x1//20, y1//20)
                
                if should_run:
                    # Crop
                    w, h = pil_img.size
                    pad = 10
                    cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                    cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
                    
                    if cx2 > cx1 and cy2 > cy1:
                        shoe_crop = pil_img.crop((cx1, cy1, cx2, cy2))
                        s_type, s_conf = classify_shoe_crop(shoe_crop)
                        t2_shoe_cache[key] = (s_type, s_conf)
                        label_text = f"{s_type} ({s_conf:.2f})"
                elif key in t2_shoe_cache:
                    s_type, s_conf = t2_shoe_cache[key]
                    label_text = f"{s_type} ({s_conf:.2f})"
                else:
                    label_text = "Shoe (..."

            detections.append((label_text, conf, (x1, y1, x2, y2), color))

    # --- SMART DETECTION FUSION ---
    # Problem: Both models detect the same object, causing duplicates and color flickering.
    # Solution: Merge overlapping detections, keep highest confidence label, use consistent colors.
    
    # Define color mapping by class category (consistent regardless of which model detected it)
    def get_class_color(cls_name):
        # Clothes = Green
        if cls_name in ['short_sleeve_top', 'long_sleeve_top', 'vest', 'shorts', 'trousers', 
                        'skirt', 'dress', 'short_sleeve_dress', 'long_sleeve_dress', 
                        'vest_dress', 'sling_dress', 'sling', 'short_sleeve_outwear', 'long_sleeve_outwear']:
            return "#22c55e"  # Green
        # Outerwear = Blue
        elif cls_name in ['jacket', 'coat']:
            return "#3b82f6"  # Blue
        # Shoes = Red
        elif cls_name in ['shoe', 'Casual Shoes', 'Flip Flops', 'Formal Shoes', 'Sandals', 
                          'Sports Shoes', 'Heels', 'Flats', 'boot', 'heel']:
            return "#ef4444"  # Red
        # Accessories = Orange
        else:
            return "#f59e0b"  # Orange (bag, belt, glasses, watch, sock, tie, hat, scarf)
    
    filtered_detections = []
    used_indices = set()
    
    for i, (label_i, conf_i, box_i, color_i) in enumerate(detections):
        if i in used_indices:
            continue
            
        # Check if this box overlaps with any other box
        duplicates = [(i, conf_i)]
        
        for j, (label_j, conf_j, box_j, color_j) in enumerate(detections):
            if i >= j or j in used_indices:
                continue
            
            # Calculate IoU
            iou = calculate_iou(box_i, box_j)
            
            # If boxes significantly overlap (>50%), they're likely the same object
            if iou > 0.5:
                duplicates.append((j, conf_j))
        
        # Keep only the highest confidence detection, but use consistent color
        if len(duplicates) > 1:
            best_idx = max(duplicates, key=lambda x: x[1])[0]
            used_indices.update([idx for idx, _ in duplicates if idx != best_idx])
            if best_idx == i:
                # Use consistent color based on class, not model
                consistent_color = get_class_color(label_i)
                filtered_detections.append((label_i, conf_i, box_i, consistent_color))
        else:
            # Use consistent color based on class, not model
            consistent_color = get_class_color(label_i)
            filtered_detections.append((label_i, conf_i, box_i, consistent_color))
    
    # Draw filtered detections
    for label, conf, (x1, y1, x2, y2), color in filtered_detections:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Text background
        text_w = draw.textlength(f"{label} {conf:.2f}", font=font)
        draw.rectangle([x1, y1-20, x1+text_w+10, y1], fill=color)
        draw.text((x1+5, y1-20), f"{label} {conf:.2f}", fill="white", font=font)
        
    return pil_img

# ----------------------
# UI FUNCTIONS - TAB 1
# ----------------------
def t1_set_image(pil_img):
    global t1_display_img, t1_display_tk, t1_crop_rect_id, t1_crop_start, t1_crop_end
    t1_display_img = pil_img.copy()
    c_w = max(t1_canvas.winfo_width(), 1)
    c_h = max(t1_canvas.winfo_height(), 1)
    t1_display_img.thumbnail((c_w, c_h))
    t1_display_tk = ImageTk.PhotoImage(t1_display_img)
    t1_canvas.delete("all")
    t1_canvas.create_image(c_w // 2, c_h // 2, image=t1_display_tk, anchor="center")
    t1_crop_rect_id = None; t1_crop_start = None; t1_crop_end = None

def t1_upload():
    fp = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.jpeg *.png")])
    if fp: t1_load(fp)

def t1_load(fp):
    global t1_orig_img, t1_active_roi
    try:
        img = Image.open(fp).convert("RGB")
        t1_orig_img = img
        t1_active_roi = None
        t1_set_image(img)
        t1_headline.config(text=f"Loaded: {os.path.basename(fp)}")
        t1_conf_tbl.delete(*t1_conf_tbl.get_children())
    except Exception as e:
        messagebox.showerror("Error", str(e))

def t1_classify():
    global t1_active_roi
    if t1_orig_img is None: return
    
    # Handle Crop
    roi = t1_orig_img.copy()
    if t1_crop_start and t1_crop_end:
        # (Simplified crop logic for brevity - assumes user drew on scaled image)
        # In full app, we'd map coordinates back to original image size
        pass 
    
    # Smart Preprocess (Phase 2 Detection -> Crop)
    if t1_preprocess_var.get() == 1 and model_clothes:
        results = model_clothes(roi, verbose=False)
        if len(results) > 0 and results[0].boxes:
            # Find best box
            best_box = max(results[0].boxes, key=lambda x: x.conf[0])
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            margin = 10
            x1 = max(0, x1-margin); y1 = max(0, y1-margin)
            x2 = min(roi.width, x2+margin); y2 = min(roi.height, y2+margin)
            roi = roi.crop((x1, y1, x2, y2))
    
    t1_active_roi = roi
    t1_set_image(roi)
    
    # Classify
    preds, top1_idx = classify_phase1(roi, topk=5)
    
    # Update Table
    t1_conf_tbl.delete(*t1_conf_tbl.get_children())
    for i, (lab, p) in enumerate(preds, 1):
        blocks = int(round(p / 5))
        bar = "█" * blocks
        t1_conf_tbl.insert("", "end", values=(i, lab, bar, f"{p:.2f}%"))
        
    t1_headline.config(text=f"Prediction: {preds[0][0]} ({preds[0][1]:.1f}%)")

def t1_heatmap():
    if t1_active_roi is None: return
    overlay = gradcam_phase1(t1_active_roi)
    t1_set_image(overlay)

# ----------------------
# UI FUNCTIONS - TAB 2
# ----------------------
def t2_draw_placeholder():
    t2_canvas.delete("all")
    w = t2_canvas.winfo_width()
    h = t2_canvas.winfo_height()
    t2_canvas.create_rectangle(0, 0, w, h, fill="black")
    t2_canvas.create_text(w//2, h//2, text="Camera Off", fill="white", font=("Segoe UI", 24))

def t2_refresh_cameras():
    t2_camera_combo['values'] = ["Scanning..."]
    t2_camera_combo.current(0)
    root.update()
    
    available = []
    print("📷 Scanning for cameras (Indices 0-10)...")
    # Scan indices 0-10 to find external webcams
    for i in range(10):
        found_this_index = False
        
        # 1. Try DSHOW (DirectShow) - Best for Webcams
        if os.name == 'nt':
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Try reading a few frames to ensure it's active
                for _ in range(3):
                    ret, _ = cap.read()
                    if ret:
                        available.append(f"Camera {i} (DSHOW)")
                        print(f"✅ Found Camera {i} (DSHOW)")
                        found_this_index = True
                        break
            cap.release()
        
        # 2. Try MSMF (Media Foundation) - Fallback
        if not found_this_index:
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
            if cap.isOpened():
                for _ in range(3):
                    ret, _ = cap.read()
                    if ret:
                        available.append(f"Camera {i} (MSMF)")
                        print(f"✅ Found Camera {i} (MSMF)")
                        found_this_index = True
                        break
            cap.release()

        # 3. Try ANY (Auto-detect) - Last Resort
        if not found_this_index:
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                for _ in range(3):
                    ret, _ = cap.read()
                    if ret:
                        available.append(f"Camera {i} (ANY)")
                        print(f"✅ Found Camera {i} (ANY)")
                        found_this_index = True
                        break
            cap.release()
            
    if not available:
        available = ["No Camera Found"]
        print("❌ No cameras found.")
    
    t2_camera_combo['values'] = available
    t2_camera_combo.current(0)

def t2_toggle_webcam():
    global t2_is_running, t2_cap
    if t2_is_running:
        # Stop
        t2_is_running = False
        if t2_cap: t2_cap.release()
        t2_btn_cam.config(text="Start Camera", style="Accent.TButton")
        t2_headline.config(text="Camera Off")
        t2_draw_placeholder()
        t2_camera_combo.state(["!disabled"])
        t2_btn_refresh.state(["!disabled"])
    else:
        # Start
        selection = t2_camera_combo.get()
        if not selection or "Camera" not in selection:
            messagebox.showerror("Error", "Please select a valid camera first.")
            return
            
        idx = int(selection.split(" ")[1])
        
        # Determine backend from label
        if "(DSHOW)" in selection:
            backend = cv2.CAP_DSHOW
        elif "(MSMF)" in selection:
            backend = cv2.CAP_MSMF
        elif "(ANY)" in selection:
            backend = cv2.CAP_ANY
        else:
            backend = cv2.CAP_ANY
            
        t2_headline.config(text=f"Initializing Camera {idx}...")
        print(f"🚀 Starting Camera {idx} with backend {backend}...")
        root.update()
        
        cap = cv2.VideoCapture(idx, backend)
        
        # NVIDIA Broadcast / Virtual Camera Fix:
        # These cameras often require their native resolution (usually 1920x1080) to work correctly.
        # If we don't set it, OpenCV might default to 640x480, causing the "dot pattern" or black screen.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                t2_cap = cap
                t2_is_running = True
                t2_btn_cam.config(text="Stop Camera", style="Warn.TButton")
                t2_headline.config(text="Live Feed Active")
                t2_camera_combo.state(["disabled"])
                t2_btn_refresh.state(["disabled"])
                t2_loop()
                return

        t2_headline.config(text="Camera Error")
        messagebox.showerror("Camera Error", f"Could not open Camera {idx}.\nTry selecting a different camera.")
        t2_draw_placeholder()

def t2_loop():
    if not t2_is_running or t2_cap is None: return
    
    ret, frame = t2_cap.read()
    if ret:
        # CV2 (BGR) -> PIL (RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # --- WEBCAM ENHANCEMENT FOR LOW-QUALITY CAMERAS ---
        # Apply enhancement if enabled (helps with grainy laptop webcams)
        if t2_enhance_mode > 0:
            pil_img = enhance_webcam_frame(pil_img, mode=t2_enhance_mode)
        
        # Process
        processed_img = process_detection_frame(pil_img, is_video=True)
        
        # Display
        t2_update_canvas(processed_img)
    
    if t2_is_running:
        root.after(15, t2_loop) # ~60 FPS cap

def t2_upload():
    # Stop webcam if running
    if t2_is_running: t2_toggle_webcam()
    
    fp = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.jpeg *.png")])
    if fp:
        try:
            img = Image.open(fp).convert("RGB")
            processed = process_detection_frame(img, is_video=False)
            t2_update_canvas(processed)
            t2_headline.config(text=f"Analyzed: {os.path.basename(fp)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def t2_update_canvas(pil_img):
    global t2_display_tk
    c_w = max(t2_canvas.winfo_width(), 1)
    c_h = max(t2_canvas.winfo_height(), 1)
    
    # Resize to fit (contain)
    img_w, img_h = pil_img.size
    ratio = min(c_w/img_w, c_h/img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    t2_display_tk = ImageTk.PhotoImage(resized)
    
    t2_canvas.delete("all")
    # Draw black background
    t2_canvas.create_rectangle(0, 0, c_w, c_h, fill="black")
    # Center image
    t2_canvas.create_image(c_w // 2, c_h // 2, image=t2_display_tk, anchor="center")

# ----------------------
# MAIN SETUP
# ----------------------
def setup_ui():
    global root, notebook, tab1, tab2
    global t1_canvas, t1_headline, t1_conf_tbl, t1_preprocess_var
    global t2_canvas, t2_headline, t2_btn_cam, t2_camera_combo, t2_btn_refresh
    
    RootCls = TkinterDnD.Tk if HAS_DND else tk.Tk
    root = RootCls()
    root.title("AAI3001 Group 10 · Fashion Intelligence Suite")
    root.geometry("1200x850")
    
    # Styles
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TNotebook.Tab", font=("Segoe UI", 12, "bold"), padding=[10, 5])
    style.configure("Accent.TButton", background="#22c55e", foreground="white", font=("Segoe UI", 11, "bold"))
    style.map("Accent.TButton", background=[("active", "#16a34a")])
    style.configure("Warn.TButton", background="#ef4444", foreground="white", font=("Segoe UI", 11, "bold"))
    style.map("Warn.TButton", background=[("active", "#dc2626")])
    
    # Notebook
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # --- TAB 1: Phase 1 Classification ---
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="  Phase 1: Image Classification  ")
    
    # T1 Controls
    t1_ctrl = ttk.Frame(tab1); t1_ctrl.pack(pady=10)
    ttk.Button(t1_ctrl, text="Upload Image", command=t1_upload).pack(side="left", padx=5)
    t1_preprocess_var = tk.IntVar(value=1)
    ttk.Checkbutton(t1_ctrl, text="Smart Preprocess (Crop)", variable=t1_preprocess_var).pack(side="left", padx=10)
    ttk.Button(t1_ctrl, text="Classify", command=t1_classify, style="Accent.TButton").pack(side="left", padx=5)
    ttk.Button(t1_ctrl, text="Explain (Heatmap)", command=t1_heatmap).pack(side="left", padx=5)
    
    # T1 Main Area
    t1_main = ttk.Frame(tab1); t1_main.pack(fill="both", expand=True)
    
    # Left: Canvas
    t1_left = ttk.Frame(t1_main); t1_left.pack(side="left", fill="both", expand=True)
    t1_headline = ttk.Label(t1_left, text="Upload an image to start", font=("Segoe UI", 14))
    t1_headline.pack(pady=5)
    t1_canvas = tk.Canvas(t1_left, bg="#1e293b")
    t1_canvas.pack(fill="both", expand=True, padx=5, pady=5)
    
    # Right: Table
    t1_right = ttk.Frame(t1_main, width=300); t1_right.pack(side="right", fill="y", padx=5)
    t1_conf_tbl = ttk.Treeview(t1_right, columns=("rank","label","bar","prob"), show="headings", height=20)
    t1_conf_tbl.heading("rank", text="#"); t1_conf_tbl.column("rank", width=30)
    t1_conf_tbl.heading("label", text="Class"); t1_conf_tbl.column("label", width=150)
    t1_conf_tbl.heading("bar", text="Conf"); t1_conf_tbl.column("bar", width=100)
    t1_conf_tbl.heading("prob", text="%"); t1_conf_tbl.column("prob", width=60)
    t1_conf_tbl.pack(fill="both", expand=True)
    
    # --- TAB 2: Phase 2/3 Object Detection ---
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="  Phase 2: Object Detection  ")
    
    # Main Layout: Video on top (expanding), Controls at bottom
    
    # 1. Video Area
    t2_video_frame = ttk.Frame(tab2)
    t2_video_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    t2_headline = ttk.Label(t2_video_frame, text="Camera Off", font=("Segoe UI", 10))
    t2_headline.pack(anchor="w", padx=5)
    
    t2_canvas = tk.Canvas(t2_video_frame, bg="black", highlightthickness=0)
    t2_canvas.pack(fill="both", expand=True)
    
    # 2. Control Bar (Bottom)
    t2_ctrl = ttk.Frame(tab2, padding=10)
    t2_ctrl.pack(fill="x", side="bottom")
    
    # Center container for buttons
    t2_btn_container = ttk.Frame(t2_ctrl)
    t2_btn_container.pack(anchor="center")
    
    # Camera Selection
    ttk.Label(t2_btn_container, text="Source:").pack(side="left", padx=5)
    t2_camera_combo = ttk.Combobox(t2_btn_container, state="readonly", width=15)
    t2_camera_combo.pack(side="left", padx=5)
    t2_btn_refresh = ttk.Button(t2_btn_container, text="↻", width=3, command=t2_refresh_cameras)
    t2_btn_refresh.pack(side="left", padx=2)
    
    # Separator
    ttk.Separator(t2_btn_container, orient="vertical").pack(side="left", fill="y", padx=15, pady=5)

    # --- Confidence Slider ---
    def update_threshold(val):
        global t2_conf_threshold
        t2_conf_threshold = float(val)
        t2_thresh_lbl.config(text=f"{int(t2_conf_threshold*100)}%")

    ttk.Label(t2_btn_container, text="Sensitivity:").pack(side="left", padx=5)
    t2_thresh_slider = ttk.Scale(t2_btn_container, from_=0.1, to=1.0, value=0.5, command=update_threshold, length=100)
    t2_thresh_slider.pack(side="left", padx=5)
    t2_thresh_lbl = ttk.Label(t2_btn_container, text="50%", width=4)
    t2_thresh_lbl.pack(side="left", padx=0)
    
    ttk.Separator(t2_btn_container, orient="vertical").pack(side="left", fill="y", padx=15, pady=5)
    
    # --- WEBCAM ENHANCEMENT CONTROL ---
    # For low-quality laptop webcams (grainy, low-res)
    def update_enhance(val):
        global t2_enhance_mode
        t2_enhance_mode = int(float(val))
        modes = ["Off", "Light", "Strong"]
        t2_enhance_lbl.config(text=modes[t2_enhance_mode])
    
    ttk.Label(t2_btn_container, text="Enhance:").pack(side="left", padx=5)
    t2_enhance_slider = ttk.Scale(t2_btn_container, from_=0, to=2, value=0, command=update_enhance, length=60)
    t2_enhance_slider.pack(side="left", padx=2)
    t2_enhance_lbl = ttk.Label(t2_btn_container, text="Off", width=6)
    t2_enhance_lbl.pack(side="left", padx=0)
    
    ttk.Separator(t2_btn_container, orient="vertical").pack(side="left", fill="y", padx=15, pady=5)
    # -------------------------

    t2_btn_cam = ttk.Button(t2_btn_container, text="Start Camera", command=t2_toggle_webcam, style="Accent.TButton")
    t2_btn_cam.pack(side="left", padx=10)
    
    ttk.Button(t2_btn_container, text="Upload Image", command=t2_upload).pack(side="left", padx=10)
    
    # Initial placeholder
    t2_canvas.bind("<Configure>", lambda e: t2_draw_placeholder() if not t2_is_running and t2_display_tk is None else None)
    
    # Initial Scan
    root.after(500, t2_refresh_cameras)
    
    # Footer
    ttk.Label(root, text="AAI3001 Group 10 Project Demo", font=("Segoe UI", 9)).pack(pady=5)

if __name__ == "__main__":
    print("🚀 Initializing Fashion Intelligence Suite...")
    
    # Load Models
    load_phase1_models()
    load_detection_models()
    
    # Start UI
    setup_ui()
    root.mainloop()
