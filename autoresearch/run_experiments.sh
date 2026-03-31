#!/bin/bash
# autoresearch/run_experiments.sh — Hardened Karpathy-style autonomous experiment loop.
#
# v2: Tighter constraints to prevent agent from breaking things.
# Changes from v1:
#   - Removed Write tool from allowedTools (agent was creating rogue files)
#   - Added pre/post-batch file integrity checks
#   - Added candidate_strategy.py line count guard
#   - Added git stash safety net
#   - Stricter prompt instructions about file creation
#
# Usage:
#   cd /opt/ippo && bash autoresearch/run_experiments.sh
#
# To stop: Ctrl+C or kill the process.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/output/autoresearch_agent.log"
STRATEGY_FILE="$SCRIPT_DIR/candidate_strategy.py"

cd "$PROJECT_DIR"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "none")
echo "Working on branch: $CURRENT_BRANCH"

# Initialize results.tsv if missing
RESULTS_TSV="$SCRIPT_DIR/results.tsv"
if [ ! -f "$RESULTS_TSV" ]; then
    printf "commit\ttest_pnl\ttest_trades\tz_score\tstatus\tdescription\n" > "$RESULTS_TSV"
fi

echo "=============================================="
echo "  Ippo AutoResearch v2 — Hardened Loop"
echo "  Branch: $CURRENT_BRANCH"
echo "  Log: $LOG_FILE"
echo "  Started: $(date)"
echo "=============================================="

# ── Snapshot known-good state ──
# Save a copy of candidate_strategy.py so we can recover if agent corrupts it
cp "$STRATEGY_FILE" "$STRATEGY_FILE.known_good" 2>/dev/null

# ── Function: remove ALL rogue files ──
cleanup_rogue_files() {
    local FOUND=0
    # Check project root for shadow files
    for shadow in numpy.py dotenv.py pandas.py requests.py scipy.py math.py json.py sys.py os.py pathlib.py; do
        if [ -f "$PROJECT_DIR/$shadow" ]; then
            echo "ALERT: Removing rogue shadow file $shadow" | tee -a "$LOG_FILE"
            rm -f "$PROJECT_DIR/$shadow"
            rm -f "$PROJECT_DIR/__pycache__/${shadow%.py}"*
            FOUND=1
        fi
    done
    # Check autoresearch/ directory
    for shadow in numpy.py dotenv.py pandas.py requests.py scipy.py math.py json.py sys.py os.py; do
        if [ -f "$SCRIPT_DIR/$shadow" ]; then
            echo "ALERT: Removing rogue autoresearch/$shadow" | tee -a "$LOG_FILE"
            rm -f "$SCRIPT_DIR/$shadow"
            FOUND=1
        fi
    done
    # NOTE: Do NOT delete untracked .py files generically. Files like
    # pnl_reconciliation.py, settlement_timing_strategy.py, ss.py, watchdog.py
    # are legitimate bot files deployed via rsync that may not be in git.
    # Only the explicit shadow file list above should be removed.
    return $FOUND
}

# ── Function: verify imports work ──
check_imports() {
    local RESULT
    RESULT=$("$PROJECT_DIR/.venv/bin/python" -c "import numpy; from dotenv import load_dotenv; import config; print('OK')" 2>&1)
    if [ "$RESULT" != "OK" ]; then
        echo "ALERT: Import check FAILED: $RESULT" | tee -a "$LOG_FILE"
        return 1
    fi
    return 0
}

# ── Function: check strategy file health ──
check_strategy_health() {
    # Check line count
    local LINES
    LINES=$(wc -l < "$STRATEGY_FILE" 2>/dev/null || echo "0")
    if [ "$LINES" -gt 600 ]; then
        echo "WARNING: candidate_strategy.py is $LINES lines (limit 500). Restoring known good." | tee -a "$LOG_FILE"
        cp "$STRATEGY_FILE.known_good" "$STRATEGY_FILE"
        return 1
    fi
    # Check it actually imports
    local IMPORT_OK
    IMPORT_OK=$("$PROJECT_DIR/.venv/bin/python" -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('cs', '$STRATEGY_FILE')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('OK' if hasattr(mod, 'evaluate_market') else 'MISSING')
" 2>&1)
    if [ "$IMPORT_OK" != "OK" ]; then
        echo "ALERT: candidate_strategy.py is broken ($IMPORT_OK). Restoring known good." | tee -a "$LOG_FILE"
        cp "$STRATEGY_FILE.known_good" "$STRATEGY_FILE"
        return 1
    fi
    return 0
}

# ── Initial cleanup ──
cleanup_rogue_files
check_imports || {
    echo "FATAL: Imports broken on startup. Manual intervention needed." | tee -a "$LOG_FILE"
    exit 1
}

# Don't set ANTHROPIC_API_KEY — use OAuth from claude login (Max subscription)
unset ANTHROPIC_API_KEY

# Refresh data
echo "Refreshing settlement data..." | tee -a "$LOG_FILE"
"$PROJECT_DIR/.venv/bin/python" "$SCRIPT_DIR/backtest_harness.py" --refresh 2>&1 | tail -5 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track restarts
RESTART_COUNT=0

# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════
while true; do
    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "=== Batch #$RESTART_COUNT starting at $(date) ===" | tee -a "$LOG_FILE"

    # Pre-batch: save known-good state
    check_strategy_health && cp "$STRATEGY_FILE" "$STRATEGY_FILE.known_good"

    # ── Launch Claude Code ──
    # CRITICAL: allowedTools is LOCKED DOWN.
    # - NO Write tool (prevents creating new files)
    # - Only Edit tool (for modifying candidate_strategy.py)
    # - Only specific Bash commands (python, git, grep, etc.)
    # - Read tool for viewing files
    claude -p \
      --allowedTools "Read,Edit,Bash(python3:*),Bash(python:*),Bash(grep:*),Bash(git:add*),Bash(git:commit*),Bash(git:checkout*),Bash(git:diff*),Bash(git:log*),Bash(git:status*),Bash(head:*),Bash(tail:*),Bash(wc:*)" \
      --model claude-sonnet-4-6 \
      --max-turns 150 \
      "Read autoresearch/program.md for your complete instructions.

CRITICAL SAFETY RULES:
- You may ONLY edit autoresearch/candidate_strategy.py using the Edit tool.
- You may NOT create any new files. Do not use Write. Do not use Bash to create files.
- If you need to test something, edit candidate_strategy.py and run the backtest.
- Keep candidate_strategy.py under 500 lines. Refactor if needed.
- Keep all contract counts as integers between 1 and 200. Use min(int(...), 200).

START:
1. Run baseline: python3 autoresearch/backtest_harness.py
2. Run calibration: python3 autoresearch/calibration_analyzer.py
3. Read candidate_strategy.py in small sections (use offset/limit, not full file)
4. Begin the experiment loop from program.md section 2.
5. Log every experiment to autoresearch/results.tsv
6. Never stop." \
      2>&1 | stdbuf -oL tee -a "$LOG_FILE"

    EXIT_CODE=$?
    echo "" | tee -a "$LOG_FILE"
    echo "=== Batch #$RESTART_COUNT ended (exit $EXIT_CODE) at $(date) ===" | tee -a "$LOG_FILE"

    # ── Post-batch health checks ──
    POISONED=0

    # 1. Remove rogue files
    cleanup_rogue_files && POISONED=1

    # 2. Verify imports
    check_imports || POISONED=1

    # 3. Verify strategy file
    check_strategy_health || POISONED=1

    # 4. Send alert if anything went wrong
    if [ "$POISONED" -eq 1 ]; then
        "$PROJECT_DIR/.venv/bin/python" -c "
from alerts import send_telegram
send_telegram(
    '<b>AUTORESEARCH ALERT</b>\n\n'
    'Issues detected after batch #$RESTART_COUNT.\n'
    'Auto-cleaned. Check: journalctl -u ippo-autoresearch -n 20'
)" 2>/dev/null
        echo "Telegram alert sent." | tee -a "$LOG_FILE"
    fi

    sleep 10
done
