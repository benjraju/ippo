#!/usr/bin/env bash
# Full analysis: collect data, backtest, report
# Run this to get actionable trade recommendations
set -uo pipefail
WORKDIR="/Users/benjamin/Desktop/ippo"
VENV="$WORKDIR/.venv/bin/python3"

cd "$WORKDIR"
echo "============================================"
echo "  IPPO FULL ANALYSIS"
echo "  $(date)"
echo "============================================"
echo

echo "[1/5] Collecting real settlement data..."
"$VENV" real_data_collector.py

echo
echo "[2/5] Running real-data backtest..."
"$VENV" real_backtest.py

echo
echo "[3/5] Scanning live markets for edges..."
"$VENV" bayesian_weather.py

echo
echo "[4/5] Checking polytope arbitrage..."
"$VENV" polytope_scanner.py

echo
echo "[5/5] Performance report..."
"$VENV" performance_tracker.py

echo
echo "============================================"
echo "  ANALYSIS COMPLETE"
echo "  Check output/ directory for detailed logs"
echo "============================================"
