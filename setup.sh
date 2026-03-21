#!/bin/bash
# =============================================================================
# KALSHI SELF-IMPROVING BOT — One-Command Setup
# =============================================================================
# Usage: bash setup.sh
# This installs everything you need. Just paste this ONE command in Terminal.
# =============================================================================

set -e

echo ""
echo "============================================="
echo "  KALSHI SELF-IMPROVING TRADING BOT SETUP"
echo "  Paper Mode Only — No Real Money Risk"
echo "============================================="
echo ""

# --- Check Python ---
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[1/6] Found Python $PYTHON_VERSION"

# --- Create virtual environment ---
echo "[2/6] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# --- Install dependencies ---
echo "[3/6] Installing dependencies (this may take 1-2 minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# --- Copy .env template if no .env exists ---
if [ ! -f .env ]; then
    echo "[4/6] Creating .env from template..."
    cp .env.template .env
    echo ""
    echo "  >>> IMPORTANT: Open .env and paste your API keys! <<<"
    echo "  >>> Get Kalshi demo keys: https://demo.kalshi.co <<<"
    echo "  >>> Get Anthropic key: https://console.anthropic.com <<<"
    echo ""
else
    echo "[4/6] .env already exists, skipping..."
fi

# --- Initialize git for AutoResearch tracking ---
echo "[5/6] Initializing git for strategy versioning..."
if [ ! -d .git ]; then
    git init -q
    git add -A
    git commit -m "Initial bot setup" -q 2>/dev/null || true
fi

# --- Create output directory ---
echo "[6/6] Creating output directories..."
mkdir -p output autoresearch

echo ""
echo "============================================="
echo "  SETUP COMPLETE!"
echo "============================================="
echo ""
echo "  Next steps:"
echo "  1. Edit .env with your API keys:"
echo "     nano .env"
echo ""
echo "  2. Run the bot in PAPER mode:"
echo "     source .venv/bin/activate"
echo "     python run_bot.py --paper"
echo ""
echo "  3. Or scan markets first:"
echo "     python cli.py scan-markets"
echo ""
echo "  4. Run overnight AutoResearch:"
echo "     python cli.py research --overnight"
echo ""
echo "============================================="
