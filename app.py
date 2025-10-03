# app.py — Upload/Drag&Drop → Crop → Reset/Finish(Classify) + Grad-CAM Explain Heatmap
# Requires: pip install tkinterdnd2 pillow torch torchvision matplotlib

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except Exception:
    HAS_DND = False

from PIL import Image, ImageTk
import pickle, os, io
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# For color mapping the CAM without OpenCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm


# ----------------------
# Load encoder + model
# ----------------------
WEIGHTS_PATH = "best_model_resnet50.pth"
ENCODER_PATH = "le_product_type.pkl"

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Missing {ENCODER_PATH}")
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Missing {WEIGHTS_PATH}")

with open(ENCODER_PATH, "rb") as f:
    le_product_type = pickle.load(f)

N_CLASSES = len(le_product_type.classes_)

# ResNet50 (matches training model!)
model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, N_CLASSES)
state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(state)
model.eval()  # keep on CPU; works fine


# ----------------------
# Preprocessing
# ----------------------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform_eval = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

# ----------------------
# Globals
# ----------------------
orig_img: Image.Image = None
display_img: Image.Image = None
display_tk: ImageTk.PhotoImage = None
scale_x = 1.0
scale_y = 1.0
crop_rect_id = None
crop_start = None
crop_end = None


# ----------------------
# Classifier helper
# ----------------------
def classify_pil(pil_img, topk=5):
    x = transform_eval(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(topk, probs.numel())
        conf, idx = torch.topk(probs, k=k)
    labels = le_product_type.inverse_transform(idx.cpu().numpy())
    conf = (conf.cpu().numpy() * 100.0).tolist()
    return list(zip(labels, conf)), int(idx[0].item())  # also return top-1 index


# ----------------------
# Grad-CAM helper (ConvNeXt-Tiny last stage)
# ----------------------
def gradcam_on_pil(pil_img, target_class=None, alpha=0.45):
    """
    Generate Grad-CAM heatmap on pil_img for target_class (int).
    If target_class is None, uses the model's top-1 on this img.
    Returns a PIL image (overlay).
    """
    # 1) Preprocess
    x = transform_eval(pil_img).unsqueeze(0)  # [1,3,224,224]

    # 2) Hook last layer in ResNet50 (layer4)
    target_layer = model.layer4[-1]
    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    # PyTorch new API
    if hasattr(target_layer, "register_full_backward_hook"):
        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())
        handle_bwd = target_layer.register_full_backward_hook(bwd_hook)
    else:  # fallback
        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())
        handle_bwd = target_layer.register_backward_hook(bwd_hook)

    handle_fwd = target_layer.register_forward_hook(fwd_hook)

    # 3) Forward
    logits = model(x)
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())

    # 4) Backward on the target logit
    model.zero_grad(set_to_none=True)
    loss = logits[0, target_class]
    loss.backward()

    # 5) Get activations & gradients
    act = activations[0][0]    # [C,H,W]
    grad = gradients[0][0]     # [C,H,W]

    # 6) Global-average pool gradients to get channel weights
    weights = grad.mean(dim=(1, 2))  # [C]

    # 7) Weighted sum of activations
    cam = torch.sum(weights[:, None, None] * act, dim=0)  # [H,W]
    cam = torch.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_np = cam.cpu().numpy()

    # 8) Resize CAM to original pil_img size and colorize using matplotlib
    cam_img = Image.fromarray(np.uint8(cam_np * 255), mode="L").resize(pil_img.size, resample=Image.BILINEAR)
    heatmap = cm.get_cmap("jet")(np.array(cam_img) / 255.0)  # RGBA in [0,1]
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)     # -> RGB uint8
    heatmap_pil = Image.fromarray(heatmap)

    # 9) Overlay
    overlay = Image.blend(pil_img.convert("RGB"), heatmap_pil, alpha=alpha)

    # 10) Clean hooks
    handle_fwd.remove()
    handle_bwd.remove()

    return overlay


# ----------------------
# UI helpers
# ----------------------
def set_conf_table(pairs):
    for r in conf_tbl.get_children():
        conf_tbl.delete(r)
    for i, (lab, p) in enumerate(pairs, 1):
        conf_tbl.insert("", "end", values=(i, lab, f"{p:.2f}%"))

def set_main_image(pil_img):
    global display_img, display_tk, scale_x, scale_y, crop_rect_id, crop_start, crop_end
    display_img = pil_img.copy()
    c_w = max(canvas.winfo_width(), 1)
    c_h = max(canvas.winfo_height(), 1)
    display_img.thumbnail((c_w, c_h))
    display_tk = ImageTk.PhotoImage(display_img)
    canvas.delete("all")
    canvas.create_image(c_w // 2, c_h // 2, image=display_tk, anchor="center")
    if orig_img is not None and display_img.width > 0 and display_img.height > 0:
        scale_x = orig_img.width / display_img.width
        scale_y = orig_img.height / display_img.height
    else:
        scale_x = scale_y = 1.0
    crop_rect_id = None
    crop_start = None
    crop_end = None
    headline.config(text="")

def upload_image():
    fp = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.webp")])
    if fp: load_and_show(fp)

def handle_drop(filepath):
    fp = filepath.strip("{}")
    if not os.path.exists(fp) or os.path.isdir(fp):
        headline.config(text="Drop a single image file.")
        return
    load_and_show(fp)

def load_and_show(filepath):
    global orig_img
    try:
        img = Image.open(filepath).convert("RGB")
    except Exception as ex:
        messagebox.showerror("Open Image Failed", str(ex))
        return
    orig_img = img
    set_main_image(orig_img)
    set_conf_table([])
    headline.config(text=f"Loaded: {os.path.basename(filepath)}")

def on_window_drop(event):
    if event.data: handle_drop(event.data)
def on_canvas_drop(event):
    if event.data: handle_drop(event.data)
def on_resize(event=None):
    if orig_img is None: return
    current = display_img if display_img is not None else orig_img
    set_main_image(current)

def on_mouse_down(event):
    global crop_start, crop_end, crop_rect_id
    if display_img is None: return
    crop_start = (event.x, event.y); crop_end = (event.x, event.y)
    if crop_rect_id is not None: canvas.delete(crop_rect_id)
    crop_rect_id = canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                           outline="#22c55e", width=3)
def on_mouse_drag(event):
    global crop_end
    if crop_rect_id is None: return
    crop_end = (event.x, event.y)
    canvas.coords(crop_rect_id, crop_start[0], crop_start[1], crop_end[0], crop_end[1])
def on_mouse_up(event):
    global crop_end
    if crop_rect_id is not None: crop_end = (event.x, event.y)

def reset_crop():
    global crop_rect_id, crop_start, crop_end
    if orig_img is None: return
    set_main_image(orig_img)
    set_conf_table([])
    headline.config(text="Crop reset — full image restored.")


# Helper to get current ROI (crop or whole image), plus mapping rect used
def get_current_roi():
    if orig_img is None:
        return None
    if crop_start and crop_end and crop_rect_id is not None:
        x0,y0 = crop_start; x1,y1 = crop_end
        x0,x1 = sorted([x0,x1]); y0,y1 = sorted([y0,y1])
        c_w = canvas.winfo_width(); c_h = canvas.winfo_height()
        img_left = (c_w - display_img.width)//2; img_top = (c_h - display_img.height)//2
        x0 = max(x0,img_left); y0=max(y0,img_top)
        x1 = min(x1,img_left+display_img.width); y1=min(y1,img_top+display_img.height)
        if x1<=x0 or y1<=y0:
            return orig_img.copy()
        disp_x0,disp_y0=x0-img_left,y0-img_top
        disp_x1,disp_y1=x1-img_left,y1-img_top
        ori_x0=int(round(disp_x0*scale_x)); ori_y0=int(round(disp_y0*scale_y))
        ori_x1=int(round(disp_x1*scale_x)); ori_y1=int(round(disp_y1*scale_y))
        return orig_img.crop((ori_x0,ori_y0,ori_x1,ori_y1))
    else:
        return orig_img.copy()


def finish_and_classify():
    if orig_img is None:
        headline.config(text="Please upload or drop an image first."); return
    roi = get_current_roi()
    preds, _ = classify_pil(roi, topk=8)
    set_conf_table(preds)
    top1_label, top1_conf = preds[0]
    headline.config(text=f"Predicted: {top1_label} ({top1_conf:.2f}%)")
    set_main_image(roi)


def explain_heatmap():
    """Run Grad-CAM on the current ROI for the top-1 predicted class and show overlay."""
    if orig_img is None:
        headline.config(text="Please upload or drop an image first."); return
    roi = get_current_roi()
    # get top-1 class for this ROI
    _, top1_idx = classify_pil(roi, topk=1)
    overlay = gradcam_on_pil(roi, target_class=top1_idx, alpha=0.45)
    set_main_image(overlay)
    headline.config(text="Explanation heatmap (Grad-CAM) for top-1 class")


# ----------------------
# Build GUI
# ----------------------
RootCls = TkinterDnD.Tk if HAS_DND else tk.Tk
root = RootCls()
root.title("AAI3001 Group 10 Fashion Product Type Classifier Tool")

try:
    root.state("zoomed")
except:
    try: root.attributes("-zoomed", True)
    except: root.geometry("1200x800")
root.configure(bg="#f8fafc")

title = tk.Label(
    root,
    text="AAI3001 Group 10 Fashion Product Type Classifier",
    font=("Segoe UI", 18, "bold"),
    bg="#f8fafc", fg="#111827"
)
title.pack(pady=(10, 4))

# Controls bar
ctrl = tk.Frame(root, bg="#f8fafc"); ctrl.pack(pady=4)
tk.Button(ctrl, text="Upload Image", command=upload_image,
          bg="#10b981", fg="white", font=("Segoe UI", 12), padx=14, pady=6).grid(row=0, column=0, padx=6)
tk.Button(ctrl, text="Reset Crop", command=reset_crop,
          bg="#6b7280", fg="white", font=("Segoe UI", 12), padx=10, pady=6).grid(row=0, column=1, padx=6)
tk.Button(ctrl, text="Finish (Classify)", command=finish_and_classify,
          bg="#2563eb", fg="white", font=("Segoe UI", 12), padx=14, pady=6).grid(row=0, column=2, padx=6)
tk.Button(ctrl, text="Explain (Heatmap)", command=explain_heatmap,
          bg="#f59e0b", fg="white", font=("Segoe UI", 12), padx=14, pady=6).grid(row=0, column=3, padx=6)

# Main layout
main = tk.Frame(root, bg="#f8fafc"); main.pack(fill="both", expand=True, padx=10, pady=8)
left = tk.Frame(main, bg="#f8fafc"); left.pack(side="left", fill="both", expand=True)
canvas = tk.Canvas(left, bg="#e5e7eb", highlightthickness=1, highlightbackground="#d1d5db")
canvas.pack(fill="both", expand=True)

if HAS_DND:
    root.drop_target_register(DND_FILES); root.dnd_bind("<<Drop>>", on_window_drop)
    canvas.drop_target_register(DND_FILES); canvas.dnd_bind("<<Drop>>", on_canvas_drop)

canvas.bind("<Configure>", on_resize)
canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

headline = tk.Label(left, text="", font=("Segoe UI", 14, "bold"),
                    bg="#f8fafc", fg="#0f766e")
headline.pack(pady=(6,0))

# Right: predictions
right = tk.Frame(main, bg="#f8fafc"); right.pack(side="right", fill="y")
tk.Label(right, text="Top Predictions", font=("Segoe UI", 16, "bold"),
         bg="#f8fafc", fg="#111827").pack(pady=(0,6))

conf_tbl = ttk.Treeview(right, columns=("rank","label","prob"), show="headings", height=18)
conf_tbl.heading("rank", text="#");      conf_tbl.column("rank", width=40, anchor="center")
conf_tbl.heading("label", text="Class"); conf_tbl.column("label", width=200, anchor="w")
conf_tbl.heading("prob", text="Conf.");  conf_tbl.column("prob", width=90, anchor="e")
conf_tbl.pack(padx=6, pady=6, fill="y")

root.mainloop()
