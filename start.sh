#!/usr/bin/env bash
# GuardEye edge device startup script
# Place in ~/edgedevice/start.sh and run:  bash start.sh
#
# What this does:
#   1. Loads credentials from .env in the same folder
#   2. Starts proctor_edge.py in headless mode
#   3. The script prints your Device ID — type it into the GuardEye browser UI
#   4. Stays running until you Ctrl+C

set -e
cd "$(dirname "$0")"

# Load environment variables from .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Validate required variables
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_ANON_KEY" ]; then
    echo ""
    echo "ERROR: SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env"
    echo "Edit .env and add your Supabase credentials."
    echo ""
    exit 1
fi

echo ""
echo "Starting GuardEye edge device..."
echo "Device ID: ${GUARDEYE_DEVICE_ID:-pi-edge-001}"
echo ""

python proctor_edge.py \
    --source pi \
    --raspi \
    --headless \
    --alert-threshold 0.75 \
    --alert-cooldown-sec 10 \
    --camera-fps 30 \
    --status-every-frames 60
