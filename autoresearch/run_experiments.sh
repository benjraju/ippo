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

set -euo pipefail

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

# Refresh settlement data before starting
echo "Refreshing settlement data..."
python3 autoresearch/backtest_harness.py --refresh > /dev/null 2>&1 || true

# Launch Claude Code with the program.md prompt
# The agent will loop autonomously until interrupted
claude --print \
  --model claude-sonnet-4-6 \
  --allowedTools "Read,Write,Edit,Bash,Grep,Glob" \
  "Hi, read autoresearch/program.md and let's kick off experiments. \
Start by reading the program.md for full context, then read candidate_strategy.py \
and run a baseline backtest. After that, begin the experiment loop. \
Log results to autoresearch/results.tsv. Never stop." \
  2>&1 | tee -a "$LOG_FILE"
