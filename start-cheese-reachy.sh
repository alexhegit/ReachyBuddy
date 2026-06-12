#!/bin/bash
# Start ReachyBuddy Cheese mode with Reachy Mini camera via SDK
# This uses the robot's camera and enables head movement in simulator

cd "$(dirname "$0")"

# Use --camera-source reachy to connect via Reachy SDK
# The daemon must be running first: ./start-daemon.sh
exec .venv/bin/python main.py \
    --cheese \
    --camera-source reachy \
    --piper-model models/en-us-ryan-medium.onnx \
    --gui-backend cv2 \
    --save-dir ./photos \
    "$@"
