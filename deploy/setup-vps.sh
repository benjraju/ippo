#!/bin/bash
# setup-vps.sh -- One-time setup for a fresh Ubuntu 24.04 VPS
# Run as root: bash setup-vps.sh
set -euo pipefail

echo "=== Ippo Trading Bot — VPS Setup ==="

# --- System packages ---
apt-get update
apt-get install -y \
    software-properties-common \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    curl \
    logrotate

# --- Python 3.14 (deadsnakes PPA) ---
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.14 python3.14-venv python3.14-dev

# --- Create ippo user ---
if ! id -u ippo &>/dev/null; then
    useradd -r -m -s /bin/bash ippo
    echo "Created user 'ippo'"
fi

# --- Create project directory ---
mkdir -p /opt/ippo/output
chown -R ippo:ippo /opt/ippo

echo ""
echo "=== VPS setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Run deploy.sh from your local machine to push code"
echo "  2. SSH in and configure /opt/ippo/.env"
echo "  3. Run: sudo bash /opt/ippo/deploy/install-services.sh"
echo ""
