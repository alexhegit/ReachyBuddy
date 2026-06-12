#!/bin/bash
# Start Reachy Mini daemon for REAL robot (not simulation)
# Uses --no-media because we handle camera directly via OpenCV

cd "$(dirname "$0")"
exec .venv/bin/python -m reachy_mini.daemon.app.main --no-media "$@"
