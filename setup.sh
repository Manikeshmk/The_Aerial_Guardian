#!/bin/bash
# Setup script for The Aerial Guardian
# Installs dependencies and downloads models

set -e

echo "================================================"
echo "The Aerial Guardian - Setup Script"
echo "================================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✓ Pip upgraded"

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Download YOLOv8n model
echo "Downloading YOLOv8n model..."
python3 << 'EOF'
from ultralytics import YOLO
import os

model_path = 'yolov8n.pt'
if not os.path.exists(model_path):
    print("  Downloading YOLOv8n...")
    model = YOLO('yolov8n.pt')
    print("✓ Model downloaded successfully")
else:
    print(f"✓ Model already exists at {model_path}")
EOF

# Verify installation
echo ""
echo "Verifying installation..."
python3 << 'EOF'
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.detector import AerialDetector
from src.tracker import DroneByteTrac
from src.pipeline import DroneVideoProcessor

print("✓ All core modules imported successfully")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ OpenCV version: {cv2.__version__}")
EOF

echo ""
echo "================================================"
echo "Setup Complete! ✓"
echo "================================================"
echo ""
echo "Quick start:"
echo "  python main.py --video <video_path> --output output/tracked.mp4"
echo ""
echo "For more options:"
echo "  python main.py --help"
echo ""
echo "To run benchmarks:"
echo "  python benchmark.py --video <video_path> --num-frames 300"
echo ""
