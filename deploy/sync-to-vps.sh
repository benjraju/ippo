#!/bin/bash
# sync-to-vps.sh -- Sync latest code to VPS, restart services, smoke test
# Usage: bash deploy/sync-to-vps.sh user@your-vps-ip
#   e.g. bash deploy/sync-to-vps.sh root@167.99.123.45
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: bash deploy/sync-to-vps.sh user@your-vps-ip"
    echo "  e.g. bash deploy/sync-to-vps.sh root@167.99.123.45"
    exit 1
fi

VPS="$1"
REMOTE_DIR="/opt/ippo"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Services to restart after sync (persistent / long-running)
RESTART_SERVICES=(
    ippo-arb-runner
    ippo-autoresearch
    ippo-health
    ippo-telegram
)

# Timers to reload (they pick up new unit files automatically after daemon-reload)
RELOAD_TIMERS=(
    ippo-auto-trade.timer
    ippo-data-collector.timer
    ippo-settlement.timer
    ippo-hourly-update.timer
    ippo-daily-recap.timer
    ippo-autoresearch.timer
)

echo "======================================================"
echo "  Ippo VPS Sync"
echo "  Local:  ${LOCAL_DIR}"
echo "  Remote: ${VPS}:${REMOTE_DIR}"
echo "======================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Rsync code
# ---------------------------------------------------------------------------
echo "[1/4] Syncing code..."
rsync -avz --delete \
    --exclude '.venv/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'output/*.log' \
    --exclude 'output/*.html' \
    --exclude 'output/*.csv' \
    --exclude 'output/*.png' \
    --exclude '.env' \
    --exclude 'kalshi_private_key.pem' \
    --exclude 'autoresearch/launchd.log' \
    --exclude 'autoresearch/overnight_*.log' \
    "${LOCAL_DIR}/" "${VPS}:${REMOTE_DIR}/"

echo "  Code synced."
echo ""

# ---------------------------------------------------------------------------
# Step 2: Install updated unit files and reload systemd
# ---------------------------------------------------------------------------
echo "[2/4] Installing unit files and reloading systemd..."
ssh "${VPS}" bash <<'REMOTE'
set -euo pipefail
SYSTEMD_DIR="/opt/ippo/deploy/systemd"
SYSTEMD_SYSTEM_DIR="/etc/systemd/system"

for unit in \
    ippo-arb-runner.service \
    ippo-auto-trade.service \
    ippo-auto-trade.timer \
    ippo-autoresearch.service \
    ippo-autoresearch.timer \
    ippo-data-collector.service \
    ippo-data-collector.timer \
    ippo-settlement.service \
    ippo-settlement.timer \
    ippo-hourly-update.service \
    ippo-hourly-update.timer \
    ippo-daily-recap.service \
    ippo-daily-recap.timer \
    ippo-health.service \
    ippo-telegram.service; do
    cp "${SYSTEMD_DIR}/${unit}" "${SYSTEMD_SYSTEM_DIR}/${unit}"
done

cp "${SYSTEMD_DIR}/ippo-logrotate" /etc/logrotate.d/ippo
systemctl daemon-reload
echo "  Unit files installed, daemon reloaded."
REMOTE

echo ""

# ---------------------------------------------------------------------------
# Step 3: Restart affected services
# ---------------------------------------------------------------------------
echo "[3/4] Restarting services..."
for svc in "${RESTART_SERVICES[@]}"; do
    echo -n "  ${svc}... "
    if ssh "${VPS}" "systemctl is-active --quiet ${svc} 2>/dev/null || true; systemctl restart ${svc} 2>/dev/null || systemctl start ${svc} 2>/dev/null || true"; then
        echo "restarted"
    else
        echo "WARNING: could not restart ${svc} (may not be enabled yet)"
    fi
done

# Ensure timers are active (re-enable if needed)
for timer in "${RELOAD_TIMERS[@]}"; do
    echo -n "  ${timer}... "
    ssh "${VPS}" "systemctl enable --now ${timer} 2>/dev/null || true" && echo "active" || echo "WARNING: ${timer} not activated"
done

echo ""

# ---------------------------------------------------------------------------
# Step 4: Smoke test
# ---------------------------------------------------------------------------
echo "[4/4] Running smoke tests..."
PASS=0
FAIL=0

# Test: health endpoint responds
echo -n "  Health endpoint (localhost:8787/health)... "
if ssh "${VPS}" "curl -sf --max-time 5 http://localhost:8787/health > /dev/null 2>&1"; then
    echo "OK"
    PASS=$((PASS + 1))
else
    echo "FAIL (health service may still be starting)"
    FAIL=$((FAIL + 1))
fi

# Test: arb-runner is running
echo -n "  ippo-arb-runner active... "
if ssh "${VPS}" "systemctl is-active --quiet ippo-arb-runner"; then
    echo "OK"
    PASS=$((PASS + 1))
else
    echo "FAIL"
    FAIL=$((FAIL + 1))
fi

# Test: autoresearch is running
echo -n "  ippo-autoresearch active... "
if ssh "${VPS}" "systemctl is-active --quiet ippo-autoresearch"; then
    echo "OK"
    PASS=$((PASS + 1))
else
    echo "FAIL"
    FAIL=$((FAIL + 1))
fi

# Test: timers are scheduled
echo -n "  Timers listed (ippo-*)... "
TIMER_COUNT=$(ssh "${VPS}" "systemctl list-timers 'ippo-*' --no-pager 2>/dev/null | grep -c 'ippo-' || true")
if [ "${TIMER_COUNT}" -ge 4 ]; then
    echo "OK (${TIMER_COUNT} timers active)"
    PASS=$((PASS + 1))
else
    echo "FAIL (only ${TIMER_COUNT} timers found, expected >=4)"
    FAIL=$((FAIL + 1))
fi

# Test: python import check
echo -n "  Python imports OK... "
if ssh "${VPS}" "/opt/ippo/.venv/bin/python -c 'import config; import kalshi_client; import settlement_tracker' 2>/dev/null"; then
    echo "OK"
    PASS=$((PASS + 1))
else
    echo "FAIL (check /opt/ippo/.venv and requirements.txt)"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "======================================================"
if [ "${FAIL}" -eq 0 ]; then
    echo "  SYNC COMPLETE — ${PASS}/${PASS} smoke tests passed"
else
    echo "  SYNC COMPLETE — ${PASS} passed, ${FAIL} FAILED"
    echo ""
    echo "  Troubleshooting:"
    echo "    ssh ${VPS}"
    echo "    journalctl -u ippo-arb-runner -n 30"
    echo "    journalctl -u ippo-autoresearch -n 30"
    echo "    journalctl -u ippo-health -n 30"
fi
echo "======================================================"
echo ""
echo "Useful commands on the VPS:"
echo "  ssh ${VPS} 'systemctl list-timers ippo-*'"
echo "  ssh ${VPS} 'journalctl -u ippo-autoresearch -f'"
echo "  ssh ${VPS} 'journalctl -u ippo-data-collector -n 20'"
echo "  ssh ${VPS} 'tail -f /opt/ippo/output/autoresearch_systemd.log'"
echo ""

[ "${FAIL}" -eq 0 ]
