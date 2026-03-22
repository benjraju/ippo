#!/usr/bin/env bash
# Scheduled autoresearch runner called by launchd.
# Using a shell wrapper avoids the Python fork()+EDEADLK deadlock that
# occurs when a launchd-spawned Python process uses subprocess.run() to
# spawn child Python processes (the children inherit import locks).
set -uo pipefail

WORKDIR="/Users/benjamin/Desktop/ippo"
VENV_PYTHON="$WORKDIR/.venv/bin/python"
LOG_FILE="$WORKDIR/autoresearch/launchd.log"
LOCK_FILE="/tmp/kalshi_autoresearch.lock"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

# Prevent overlapping runs
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        log "Skipping: previous run (PID $OLD_PID) still running"
        exit 0
    fi
    # Stale lock file — remove it
    log "Removing stale lock file (PID $OLD_PID no longer running)"
    rm -f "$LOCK_FILE"
fi

echo $$ > "$LOCK_FILE"
# Always clean up lock on exit (even on error)
trap 'rm -f "$LOCK_FILE"; log "=== Run ended (exit $?) ==="' EXIT

cd "$WORKDIR"
log "=== Starting scheduled autoresearch run (PID $$) ==="

# Phase 1: AutoResearch (100 iterations)
log "Phase 1: AutoResearch..."
"$VENV_PYTHON" -m autoresearch.research_loop \
    --iterations 100 \
    --no-claude \
    --use-real-data \
    >> "$LOG_FILE" 2>&1
PHASE1_EXIT=$?
if [ $PHASE1_EXIT -ne 0 ]; then
    log "Phase 1 exited with code $PHASE1_EXIT"
fi

# Phase 2: Auto-trade (live, no confirmation)
log "Phase 2: Auto-trade..."
"$VENV_PYTHON" auto_trade.py --live --no-confirm \
    >> "$LOG_FILE" 2>&1
PHASE2_EXIT=$?
if [ $PHASE2_EXIT -ne 0 ]; then
    log "Phase 2 exited with code $PHASE2_EXIT"
fi

# Phase 3: Settlement tracking
log "Phase 3: Settlement tracking..."
"$VENV_PYTHON" -c "
from settlement_tracker import SettlementTracker
t = SettlementTracker()
t.check_settlements()
t.daily_summary()
" >> "$LOG_FILE" 2>&1
PHASE3_EXIT=$?
if [ $PHASE3_EXIT -ne 0 ]; then
    log "Phase 3 exited with code $PHASE3_EXIT"
fi

log "All phases complete (p1=$PHASE1_EXIT p2=$PHASE2_EXIT p3=$PHASE3_EXIT)"
