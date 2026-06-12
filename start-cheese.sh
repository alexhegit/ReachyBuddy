#!/bin/bash
# Start ReachyBuddy Cheese mode with webcam (Reachy Mini Camera)

cd "$(dirname "$0")"

# Use Reachy Mini Camera (index 4) with cv2 GUI backend
exec .venv/bin/python main.py \
    --cheese \
    --camera-source webcam \
    --camera-index 4 \
    --piper-model models/en-us-ryan-medium.onnx \
    --gui-backend cv2 \
    --save-dir ./photos \
    "$@"
