#!/bin/bash
# Start Reachy Mini daemon in simulation mode using the project venv

cd "$(dirname "$0")"
exec .venv/bin/python -m reachy_mini.daemon.app.main --sim --headless "$@"
