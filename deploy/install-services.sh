#!/bin/bash
# install-services.sh -- Install systemd services on the VPS
# Run as root after deploy.sh has pushed code: bash install-services.sh
set -euo pipefail

IPPO_DIR="/opt/ippo"
SYSTEMD_DIR="${IPPO_DIR}/deploy/systemd"

echo "=== Installing Ippo systemd services ==="

# --- Create venv and install deps ---
echo "Setting up Python venv..."
sudo -u ippo python3.14 -m venv "${IPPO_DIR}/.venv"
sudo -u ippo "${IPPO_DIR}/.venv/bin/pip" install --upgrade pip
sudo -u ippo "${IPPO_DIR}/.venv/bin/pip" install -r "${IPPO_DIR}/requirements.txt"

# --- Install systemd unit files ---
echo "Installing systemd units..."
SYSTEMD_USER_DIR="/etc/systemd/system"

for unit in \
    ippo-arb-runner.service \
    ippo-auto-trade.service \
    ippo-auto-trade.timer \
    ippo-autoresearch.service \
    ippo-autoresearch.timer \
    ippo-settlement.service \
    ippo-settlement.timer \
    ippo-health.service; do
    cp "${SYSTEMD_DIR}/${unit}" "${SYSTEMD_USER_DIR}/${unit}"
    echo "  Installed ${unit}"
done

# --- Install logrotate ---
cp "${SYSTEMD_DIR}/ippo-logrotate" /etc/logrotate.d/ippo
echo "  Installed logrotate config"

# --- Reload and enable ---
systemctl daemon-reload

# Enable and start services
systemctl enable --now ippo-arb-runner.service
systemctl enable --now ippo-health.service

# Enable timers
systemctl enable --now ippo-auto-trade.timer
systemctl enable --now ippo-autoresearch.timer
systemctl enable --now ippo-settlement.timer

echo ""
echo "=== Services installed and started ==="
echo ""
echo "Status:"
systemctl status ippo-arb-runner.service --no-pager -l || true
echo ""
systemctl list-timers ippo-* --no-pager || true
echo ""
echo "Commands:"
echo "  systemctl status ippo-arb-runner    # Check arb runner"
echo "  systemctl status ippo-health        # Check health endpoint"
echo "  systemctl list-timers ippo-*        # See timer schedule"
echo "  journalctl -u ippo-arb-runner -f    # Follow arb runner logs"
echo "  journalctl -u ippo-auto-trade -n 50 # Last 50 auto-trade lines"
echo "  curl localhost:8787/health           # Health check"
echo ""
