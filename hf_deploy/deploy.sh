#!/bin/bash
# ============================================
# Hugging Face Spaces Deployment Script
# Fashion Intelligence Suite - AAI3001 Group 10
# ============================================

echo ""
echo "========================================"
echo " Fashion Intelligence Suite Deployment"
echo "========================================"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "[!] huggingface-cli not found. Installing..."
    pip install huggingface_hub
fi

# Login to Hugging Face
echo "[1/4] Checking Hugging Face login..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "Please log in to Hugging Face:"
    huggingface-cli login
fi

# Set your Space name
SPACE_NAME="fashion-intelligence-suite"
read -p "Enter your Hugging Face username: " USERNAME

echo ""
echo "[2/4] Creating Space: $USERNAME/$SPACE_NAME"
huggingface-cli repo create $SPACE_NAME --type space --space_sdk gradio 2>/dev/null || true

echo ""
echo "[3/4] Cloning and uploading files..."
cd "$(dirname "$0")"

# Initialize git if needed
if [ ! -d ".git" ]; then
    git init
    git lfs install
fi

# Add remote
git remote remove origin 2>/dev/null
git remote add origin https://huggingface.co/spaces/$USERNAME/$SPACE_NAME

# Add all files
git add .
git commit -m "Initial deployment of Fashion Intelligence Suite"

echo ""
echo "[4/4] Pushing to Hugging Face Spaces..."
git push -u origin main --force

echo ""
echo "========================================"
echo " Deployment Complete!"
echo "========================================"
echo ""
echo "Your Space is now live at:"
echo "https://huggingface.co/spaces/$USERNAME/$SPACE_NAME"
echo ""
echo "Note: First build may take 5-10 minutes."
