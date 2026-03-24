#!/bin/bash
# autoresearch/run_experiments.sh — Karpathy-style autonomous experiment loop.
#
# Launches Claude Code as the autonomous researcher. Claude reads program.md,
# modifies candidate_strategy.py, runs backtest_harness.py, keeps/discards,
# and loops forever.
#
# Usage:
#   cd /opt/ippo && bash autoresearch/run_experiments.sh
#
# To stop: Ctrl+C or kill the process.
#
# Logs: output/autoresearch_agent.log

set -uo pipefail
# NOTE: intentionally no `set -e` — Claude exits non-zero on max-turns,
# and we want the while loop to catch that and restart.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/output/autoresearch_agent.log"

cd "$PROJECT_DIR"

# Ensure we're on a dedicated branch
BRANCH="autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "none")

if [[ "$CURRENT_BRANCH" != autoresearch/* ]]; then
    echo "Creating experiment branch: $BRANCH"
    git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
fi

# Initialize results.tsv if it doesn't exist
RESULTS_TSV="$SCRIPT_DIR/results.tsv"
if [ ! -f "$RESULTS_TSV" ]; then
    printf "commit\ttest_pnl\ttest_trades\tz_score\tstatus\tdescription\n" > "$RESULTS_TSV"
fi

echo "=============================================="
echo "  Ippo AutoResearch — Karpathy-style loop"
echo "  Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "  Log: $LOG_FILE"
echo "  Started: $(date)"
echo "=============================================="

# Data refresh: settlement_tracker handles this on its 4h schedule
# The backtest harness reads whatever data is already in output/
echo "Using existing settlement data ($(wc -l < output/historical_settlements_with_prices.json 2>/dev/null || echo '?') lines)"

# Authentication: uses OAuth credentials from `claude` login (Max subscription)
# Do NOT set ANTHROPIC_API_KEY or it will bill to API instead of subscription
unset ANTHROPIC_API_KEY

# Track total experiment batches across restarts
RESTART_COUNT=0

# Forever loop — when Claude hits max-turns it exits, we restart immediately
while true; do
    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "=== Batch #$RESTART_COUNT starting at $(date) ===" | tee -a "$LOG_FILE"

    claude -p \
      --allowedTools "Read,Write,Edit,Bash(python3:*),Bash(grep:*),Bash(git:*),Bash(cat:*),Bash(head:*),Bash(tail:*),Bash(wc:*),Glob,Grep" \
      --model claude-sonnet-4-6 \
      --max-turns 200 \
      "Read autoresearch/program.md for full context. Then read candidate_strategy.py \
and run a baseline backtest with: python3 autoresearch/backtest_harness.py \
After that, begin the experiment loop described in program.md. \
Log every experiment to autoresearch/results.tsv. Never stop. \
If you run out of ideas, load the historical settlements JSON and explore the data." \
      2>&1 | stdbuf -oL tee -a "$LOG_FILE"

    EXIT_CODE=$?
    echo "" | tee -a "$LOG_FILE"
    echo "=== Batch #$RESTART_COUNT ended (exit code $EXIT_CODE) at $(date). Restarting in 10s... ===" | tee -a "$LOG_FILE"
    sleep 10
done
