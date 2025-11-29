# 🚀 Deployment Checklist

## Pre-Deployment

- [ ] Hugging Face account created at https://huggingface.co/join
- [ ] Git LFS installed (`git lfs install`)
- [ ] huggingface-cli installed (`pip install huggingface_hub`)
- [ ] Logged in (`huggingface-cli login`)

## Files Ready (in `hf_deploy/` folder)

| File | Purpose | Size |
|------|---------|------|
| ✅ `app.py` | Main Gradio application | 14 KB |
| ✅ `requirements.txt` | Python dependencies | 0.2 KB |
| ✅ `README.md` | Space description | 2.6 KB |
| ✅ `.gitattributes` | Git LFS config | 0.4 KB |
| ✅ `models/phase2_clothes_best.pt` | Clothes YOLO | 6 MB |
| ✅ `models/phase3_accessories_best.pt` | Accessories YOLO | 22 MB |
| ✅ `models/best_model_shoes.pth` | Shoe ResNet | 90 MB |
| ✅ `deploy.bat` | Windows deploy script | 1.8 KB |
| ✅ `deploy.sh` | Mac/Linux deploy script | 1.7 KB |

**Total Size: ~118 MB** (within HF free tier 10GB limit)

---

## Deployment Options

### Option 1: Run Deploy Script (Recommended)

**Windows:**
```cmd
cd d:\AAI3001\hf_deploy
deploy.bat
```

**Mac/Linux:**
```bash
cd /path/to/hf_deploy
chmod +x deploy.sh
./deploy.sh
```

### Option 2: Manual Deployment

1. **Create Space on Hugging Face:**
   - Go to https://huggingface.co/new-space
   - Name: `fashion-intelligence-suite`
   - SDK: Gradio
   - Click "Create Space"

2. **Clone your Space:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/fashion-intelligence-suite
   cd fashion-intelligence-suite
   ```

3. **Copy files:**
   ```bash
   cp -r d:/AAI3001/hf_deploy/* .
   ```

4. **Push to Hugging Face:**
   ```bash
   git lfs install
   git add .
   git commit -m "Initial deployment"
   git push
   ```

### Option 3: Web Upload

1. Go to your Space's "Files" tab
2. Upload files one by one
3. For model files (>10MB), use Git LFS

---

## After Deployment

1. **Wait for Build:** 5-10 minutes for first build
2. **Check Logs:** Click "Logs" tab to see build progress
3. **Test:** Upload an image and verify detection works
4. **Share URL:** `https://huggingface.co/spaces/YOUR_USERNAME/fashion-intelligence-suite`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Check `requirements.txt` has all packages |
| "Model not found" | Verify models in `models/` folder |
| Build timeout | Try GPU hardware (costs $0.60/hr) |
| Push rejected | Run `git lfs install` then push again |
| Out of memory | Models are ~118MB total, should fit |

---

## Quick Test Locally

Before deploying, test locally:
```bash
cd d:\AAI3001\hf_deploy
python app.py
```

Then open http://127.0.0.1:7860 in your browser.
