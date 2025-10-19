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
import pickle, os, io, time
from pathlib import Path
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
# Load encoder + model (LOCAL PATHS - updated for Windows)
# ----------------------
WEIGHTS_PATH = r"d:/AAI3001/best_model_resnet50_extended.pth"
ENCODER_PATH = r"d:/AAI3001/le_product_type_extended.pkl"

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Missing {ENCODER_PATH}")
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Missing {WEIGHTS_PATH}")

with open(ENCODER_PATH, "rb") as f:
    le_product_type = pickle.load(f)

N_CLASSES = len(le_product_type.classes_)

# Label helpers (UPDATED to include Blazers + Waistcoat from your training)
LABELS = list(le_product_type.classes_)
LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}
TOPS_SET = {"Tshirts", "Shirts", "Sweatshirts", "Jackets", "Sweaters", "Tops", "Blazers", "Waistcoat"}
BOTTOMS_SET = {"Jeans", "Trousers", "Shorts", "Skirts", "Track Pants", "Leggings", "Swimwear"}
DRESSES_SET = set()  # No dresses in your dataset

# ----------------------
# Model
# ----------------------
model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),                     # ✅ added dropout like training
    nn.Linear(in_features, len(le_product_type.classes_))
)

state = torch.load(WEIGHTS_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()  # keep on CPU

# ----------------------
# Preprocessing (match training eval pipeline)
# ----------------------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform_eval = T.Compose([
    T.Resize(256),      # keep aspect ratio like training
    T.CenterCrop(224),
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

# NEW: persist the last ROI you classified on (so Explain uses it even after crop clears)
active_roi: Image.Image = None

# ----------------------
# UI Theming helpers
# ----------------------
APP_COLORS = {
    "bg": "#0b1220",
    "surface": "#0f172a",
    "muted": "#1e293b",
    "accent": "#22c55e",
    "accent2": "#2563eb",
    "warn": "#f59e0b",
    "danger": "#ef4444",
    "text": "#e5e7eb",
    "subtext": "#94a3b8",
    "panel": "#111827",
}

def create_styles(root):
    style = ttk.Style(root)
    # Use a platform-stable theme
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure("App.TFrame", background=APP_COLORS["surface"]) 
    style.configure("App.TLabel", background=APP_COLORS["surface"], foreground=APP_COLORS["text"], font=("Segoe UI", 11))
    style.configure("Title.TLabel", background=APP_COLORS["surface"], foreground=APP_COLORS["text"], font=("Segoe UI", 18, "bold"))

    # Buttons
    style.configure("Accent.TButton", font=("Segoe UI", 12, "bold"), padding=8,
                    background=APP_COLORS["accent"], foreground="#ffffff")
    style.map("Accent.TButton",
              background=[("active", "#16a34a")])

    style.configure("Primary.TButton", font=("Segoe UI", 12, "bold"), padding=8,
                    background=APP_COLORS["accent2"], foreground="#ffffff")
    style.map("Primary.TButton",
              background=[("active", "#1d4ed8")])

    style.configure("Warn.TButton", font=("Segoe UI", 12, "bold"), padding=8,
                    background=APP_COLORS["warn"], foreground="#111827")
    style.map("Warn.TButton",
              background=[("active", "#d97706")])

    style.configure("Secondary.TButton", font=("Segoe UI", 12), padding=8,
                    background=APP_COLORS["muted"], foreground=APP_COLORS["text"])
    style.map("Secondary.TButton",
              background=[("active", "#334155")])

    # Treeview
    style.configure("App.Treeview",
                    background=APP_COLORS["panel"],
                    fieldbackground=APP_COLORS["panel"],
                    foreground=APP_COLORS["text"],
                    bordercolor=APP_COLORS["muted"],
                    rowheight=26)
    style.configure("App.Treeview.Heading",
                    background=APP_COLORS["muted"],
                    foreground=APP_COLORS["text"],
                    font=("Segoe UI", 11, "bold"))
    style.map("App.Treeview.Heading", background=[("active", "#334155")])

    # Progressbar
    style.configure("App.Horizontal.TProgressbar", troughcolor=APP_COLORS["muted"], background=APP_COLORS["accent"]) 

    return style

# ----------------------
# Classifier helper
# ----------------------
def classify_pil(pil_img, topk=5, allow_labels=None, downweight_bottoms=1.0):
    """
    Classify a PIL image with optional label filtering and bottom-class downweighting.
    - allow_labels: iterable of label strings to keep; if None or empty, keep all
    - downweight_bottoms: multiply probabilities of bottom classes by this factor (<=1)
    """
    x = transform_eval(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    adjusted = probs.clone()
    # Apply category mask if provided
    if allow_labels:
        allow_idx = [LABEL_TO_IDX[l] for l in allow_labels if l in LABEL_TO_IDX]
        if len(allow_idx) > 0:
            mask = torch.zeros_like(adjusted)
            mask[allow_idx] = 1.0
            adjusted = adjusted * mask
    # Downweight bottoms if requested
    if downweight_bottoms < 1.0:
        bottom_idx = [LABEL_TO_IDX[l] for l in LABELS if l in BOTTOMS_SET]
        if bottom_idx:
            adjusted[bottom_idx] = adjusted[bottom_idx] * downweight_bottoms

    # If all-zero after masking, fall back to original probs
    if float(adjusted.sum().item()) == 0.0:
        adjusted = probs

    k = min(int(topk), adjusted.numel())
    conf, idx = torch.topk(adjusted, k=k)
    labels = [LABELS[i] for i in idx.cpu().numpy().tolist()]
    conf = (conf.cpu().numpy() * 100.0).tolist()
    return list(zip(labels, conf)), int(idx[0].item())  # also return top-1 index


# ----------------------
# Grad-CAM helper (ResNet-18 last conv layer)
# ----------------------
def gradcam_on_pil(pil_img, target_class=None, alpha=0.45):
    """
    Generate Grad-CAM heatmap on pil_img for target_class (int).
    If target_class is None, uses the model's top-1 on this img.
    Returns a PIL image (overlay).
    """
    # 1) Preprocess
    x = transform_eval(pil_img).unsqueeze(0)  # [1,3,224,224]

    # 2) Hook last conv layer in layer4[-1].conv2
    target_layer = model.layer4[-1].conv2
    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

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
    logits[0, target_class].backward()

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
        # Visual confidence bar using block characters
        blocks = int(round(p / 5))  # 0..20
        bar = "█" * blocks
        conf_tbl.insert("", "end", values=(i, lab, bar, f"{p:.2f}%"))

def set_main_image(pil_img):
    """Render the given PIL on the canvas and (intentionally) clear the crop box."""
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
    # clearing the crop rect is fine; we now persist ROI separately
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
    global orig_img, active_roi
    try:
        img = Image.open(filepath).convert("RGB")
    except Exception as ex:
        messagebox.showerror("Open Image Failed", str(ex))
        return
    orig_img = img
    active_roi = None  # reset persistent ROI when a new image is loaded
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
    global crop_rect_id, crop_start, crop_end, active_roi
    if orig_img is None: return
    set_main_image(orig_img)
    set_conf_table([])
    active_roi = None  # clear persisted ROI because you explicitly reset
    headline.config(text="Crop reset — full image restored.")


# --- ROI helpers ---
def _roi_from_current_rect():
    """Return ROI from the current crop rectangle, or None if no valid rect."""
    if not (orig_img and crop_start and crop_end and crop_rect_id is not None):
        return None
    x0,y0 = crop_start; x1,y1 = crop_end
    x0,x1 = sorted([x0,x1]); y0,y1 = sorted([y0,y1])
    c_w = canvas.winfo_width(); c_h = canvas.winfo_height()
    img_left = (c_w - display_img.width)//2; img_top = (c_h - display_img.height)//2
    x0 = max(x0,img_left); y0=max(y0,img_top)
    x1 = min(x1,img_left+display_img.width); y1=min(y1,img_top+display_img.height)
    if x1<=x0 or y1<=y0:
        return None
    disp_x0,disp_y0=x0-img_left,y0-img_top
    disp_x1,disp_y1=x1-img_left,y1-img_top
    ori_x0=int(round(disp_x0*scale_x)); ori_y0=int(round(disp_y0*scale_y))
    ori_x1=int(round(disp_x1*scale_x)); ori_y1=int(round(disp_y1*scale_y))
    return orig_img.crop((ori_x0,ori_y0,ori_x1,ori_y1))

def get_current_roi():
    """
    Preferred ROI to use now:
      1) if a persisted active_roi exists → use it
      2) else if a crop rectangle exists → use that
      3) else fall back to the full original image
    """
    if active_roi is not None:
        return active_roi.copy()
    rect_roi = _roi_from_current_rect()
    if rect_roi is not None:
        return rect_roi
    return orig_img.copy() if orig_img is not None else None


def finish_and_classify():
    global active_roi
    if orig_img is None:
        headline.config(text="Please upload or drop an image first."); return

    # prefer current rect; if none, use orig
    rect_roi = _roi_from_current_rect()
    roi = rect_roi if rect_roi is not None else orig_img.copy()

    # persist this ROI so Explain uses the same region even after crop clears
    active_roi = roi.copy()

    # Build optional label filter from UI
    selected_cat = category_var.get()
    allow_labels = None
    if selected_cat == "Tops":
        allow_labels = [l for l in LABELS if l in TOPS_SET]
    elif selected_cat == "Bottoms":
        allow_labels = [l for l in LABELS if l in BOTTOMS_SET]
    elif selected_cat == "Dresses":
        allow_labels = [l for l in LABELS if l in DRESSES_SET]

    # Optional upper-body bias: downweight bottoms
    downweight = 0.6 if upper_bias_var.get() else 1.0

    start_t = time.perf_counter()
    preds, _ = classify_pil(roi, topk=topk_var.get(), allow_labels=allow_labels, downweight_bottoms=downweight)
    infer_ms = (time.perf_counter() - start_t) * 1000.0
    set_conf_table(preds)
    top1_label, top1_conf = preds[0]
    bias_note = " · bias↑" if downweight < 1.0 else ""
    cat_note = f" · {selected_cat}" if selected_cat != "All" else ""
    headline.config(text=f"Predicted: {top1_label} ({top1_conf:.2f}%) · {infer_ms:.0f} ms{cat_note}{bias_note}")

    # show ROI (this clears the crop rect, but we have active_roi stored)
    set_main_image(roi)


def explain_heatmap():
    """Run Grad-CAM on the active ROI (or current crop, or full img)."""
    if orig_img is None:
        headline.config(text="Please upload or drop an image first."); return

    roi = get_current_roi()
    if roi is None:
        headline.config(text="No image available."); return

    # get top-1 class for this ROI
    _, top1_idx = classify_pil(roi, topk=1)
    start_t = time.perf_counter()
    overlay = gradcam_on_pil(roi, target_class=top1_idx, alpha=0.45)
    cam_ms = (time.perf_counter() - start_t) * 1000.0

    # do NOT overwrite active_roi; we want to keep the ROI stable
    set_main_image(overlay)
    headline.config(text=f"Explanation heatmap (Grad-CAM) · {cam_ms:.0f} ms")


# ----------------------
# Menu actions & helpers
# ----------------------
def save_current_view():
    # Save whatever is displayed on the canvas (display_img) to a PNG
    if display_img is None:
        messagebox.showinfo("Save", "Nothing to save yet. Load and classify first.")
        return
    default = Path.cwd() / "aai3001_result.png"
    fp = filedialog.asksaveasfilename(title="Save current view", defaultextension=".png",
                                      initialfile=str(default.name),
                                      filetypes=[("PNG", "*.png")])
    if not fp:
        return
    try:
        display_img.save(fp)
        messagebox.showinfo("Saved", f"Saved view to\n{fp}")
    except Exception as e:
        messagebox.showerror("Save failed", str(e))

def show_about():
    messagebox.showinfo(
        "About",
        "AAI3001 Group 10 — Fashion Product Type Classifier\n"
        "ResNet50 · Grad-CAM · Tkinter UI\n"
        "UX-enhanced edition"
    )

def open_help():
    messagebox.showinfo(
        "Help",
        "1) Upload or drop an image.\n"
        "2) Drag to draw a crop box (optional).\n"
        "3) Click ‘Finish (Classify)’ or press Enter.\n"
        "4) Click ‘Explain (Heatmap)’ to see Grad-CAM.\n\n"
        "Shortcuts:\n"
        "  Ctrl+O — Upload\n  Enter — Classify\n  H — Heatmap\n  R — Reset crop\n  Ctrl+S — Save view"
    )


# ----------------------
# Build GUI
# ----------------------
RootCls = TkinterDnD.Tk if HAS_DND else tk.Tk
root = RootCls()
root.title("AAI3001 Group 10 · Fashion Product Type Classifier")

try:
    root.state("zoomed")
except:
    try: root.attributes("-zoomed", True)
    except: root.geometry("1200x800")
root.configure(bg=APP_COLORS["surface"]) 

# Create styles/theme
create_styles(root)

title = ttk.Label(
    root,
    text="AAI3001 Group 10 · Fashion Product Type Classifier",
    style="Title.TLabel"
)
title.pack(pady=(10, 6))

# Menu bar
menubar = tk.Menu(root)
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="Upload…	Ctrl+O", command=upload_image)
file_menu.add_command(label="Save view…	Ctrl+S", command=save_current_view)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.destroy)
menubar.add_cascade(label="File", menu=file_menu)

help_menu = tk.Menu(menubar, tearoff=0)
help_menu.add_command(label="How to use", command=open_help)
help_menu.add_command(label="About", command=show_about)
menubar.add_cascade(label="Help", menu=help_menu)

root.config(menu=menubar)

# Controls bar
ctrl = ttk.Frame(root, style="App.TFrame"); ctrl.pack(pady=6)
ttk.Button(ctrl, text="Upload Image", command=upload_image, style="Accent.TButton").grid(row=0, column=0, padx=6)
ttk.Button(ctrl, text="Reset Crop", command=reset_crop, style="Secondary.TButton").grid(row=0, column=1, padx=6)
ttk.Button(ctrl, text="Finish (Classify)", command=finish_and_classify, style="Primary.TButton").grid(row=0, column=2, padx=6)
ttk.Button(ctrl, text="Explain (Heatmap)", command=explain_heatmap, style="Warn.TButton").grid(row=0, column=3, padx=6)

# Top-K selector
ttk.Label(ctrl, text="Top-K:", style="App.TLabel").grid(row=0, column=4, padx=(14,4))
topk_var = tk.IntVar(value=8)
topk_spin = ttk.Spinbox(ctrl, from_=1, to=20, width=4, textvariable=topk_var)
topk_spin.grid(row=0, column=5, padx=4)

# Category filter
ttk.Label(ctrl, text="Category:", style="App.TLabel").grid(row=0, column=6, padx=(14,4))
category_var = tk.StringVar(value="All")
category_combo = ttk.Combobox(ctrl, values=["All", "Tops", "Bottoms", "Dresses"],
                              state="readonly", width=10, textvariable=category_var)
category_combo.grid(row=0, column=7, padx=4)

# Upper-body bias checkbox
upper_bias_var = tk.IntVar(value=1)
upper_bias_chk = ttk.Checkbutton(ctrl, text="Upper-body bias", variable=upper_bias_var)
upper_bias_chk.grid(row=0, column=8, padx=(12,4))

# Main layout
main = ttk.Frame(root, style="App.TFrame"); main.pack(fill="both", expand=True, padx=10, pady=8)
left = ttk.Frame(main, style="App.TFrame"); left.pack(side="left", fill="both", expand=True)
canvas = tk.Canvas(left, bg=APP_COLORS["muted"], highlightthickness=1, highlightbackground="#334155")
canvas.pack(fill="both", expand=True)

if HAS_DND:
    root.drop_target_register(DND_FILES); root.dnd_bind("<<Drop>>", on_window_drop)
    canvas.drop_target_register(DND_FILES); canvas.dnd_bind("<<Drop>>", on_canvas_drop)

canvas.bind("<Configure>", on_resize)
canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

headline = ttk.Label(left, text="", style="App.TLabel", font=("Segoe UI", 14, "bold"))
headline.pack(pady=(6,0))

# Right: predictions
right = ttk.Frame(main, style="App.TFrame"); right.pack(side="right", fill="y")
ttk.Label(right, text="Top Predictions", style="Title.TLabel").pack(pady=(0,6))

conf_tbl = ttk.Treeview(right, columns=("rank","label","bar","prob"), show="headings", height=18, style="App.Treeview")
conf_tbl.heading("rank", text="#", anchor="center");      conf_tbl.column("rank", width=40, anchor="center")
conf_tbl.heading("label", text="Class");                    conf_tbl.column("label", width=180, anchor="w")
conf_tbl.heading("bar", text="");                           conf_tbl.column("bar", width=160, anchor="w")
conf_tbl.heading("prob", text="Conf.");                     conf_tbl.column("prob", width=90, anchor="e")
conf_tbl.pack(padx=6, pady=6, fill="y")

# Status bar
status_var = tk.StringVar(value="Drop an image or click ‘Upload Image’.")
status = ttk.Label(root, textvariable=status_var, style="App.TLabel")
status.pack(fill="x", padx=10, pady=(0,8))

# Keyboard shortcuts
root.bind_all("<Control-o>", lambda e: upload_image())
root.bind_all("<Control-s>", lambda e: save_current_view())
root.bind_all("<Return>", lambda e: finish_and_classify())
root.bind_all("h", lambda e: explain_heatmap())
root.bind_all("H", lambda e: explain_heatmap())
root.bind_all("r", lambda e: reset_crop())
root.bind_all("R", lambda e: reset_crop())

root.mainloop()
