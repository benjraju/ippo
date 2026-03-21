#!/bin/bash
# deploy.sh -- Push code from local machine to VPS
# Usage: bash deploy/deploy.sh user@your-vps-ip
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash deploy/deploy.sh user@your-vps-ip"
    echo "  e.g. bash deploy/deploy.sh root@167.99.123.45"
    exit 1
fi

VPS="$1"
REMOTE_DIR="/opt/ippo"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Deploying Ippo to ${VPS}:${REMOTE_DIR} ==="
echo "Local: ${LOCAL_DIR}"

# Sync code (exclude venv, git, pycache, logs, env secrets)
rsync -avz --delete \
    --exclude '.venv/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude 'output/*.log' \
    --exclude 'output/*.html' \
    --exclude 'output/*.csv' \
    --exclude 'output/*.png' \
    --exclude '.env' \
    --exclude 'kalshi_private_key.pem' \
    --exclude 'autoresearch/launchd.log' \
    --exclude 'autoresearch/overnight_*.log' \
    "${LOCAL_DIR}/" "${VPS}:${REMOTE_DIR}/"

echo ""
echo "=== Code synced ==="
echo ""
echo "If this is the first deploy:"
echo "  1. SSH into the VPS: ssh ${VPS}"
echo "  2. Copy your .env:   scp ${LOCAL_DIR}/.env ${VPS}:${REMOTE_DIR}/.env"
echo "  3. Copy your PEM:    scp ${LOCAL_DIR}/kalshi_private_key.pem ${VPS}:${REMOTE_DIR}/"
echo "  4. Fix .env paths:   Edit KALSHI_PRIVATE_KEY_PATH=${REMOTE_DIR}/kalshi_private_key.pem"
echo "  5. Install services: sudo bash ${REMOTE_DIR}/deploy/install-services.sh"
echo ""
echo "If updating existing deploy:"
echo "  ssh ${VPS} 'sudo systemctl restart ippo-arb-runner ippo-health'"
echo ""
