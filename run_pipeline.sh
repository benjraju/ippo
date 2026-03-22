#!/usr/bin/env bash
# Ippo Automated Edge Collection Pipeline
# Run every 2 hours to accumulate real edge data for validation
# Cron: 0 */2 * * * cd /Users/benjamin/Desktop/ippo && bash run_pipeline.sh

set -uo pipefail
WORKDIR="/Users/benjamin/Desktop/ippo"
VENV="$WORKDIR/.venv/bin/python3"
LOG="$WORKDIR/output/pipeline.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

cd "$WORKDIR"
log "=== Pipeline started ==="

# Step 1: Log edges from live markets (both old and bayesian approach)
log "Step 1: Logging edges..."
"$VENV" edge_logger.py --log >> "$LOG" 2>&1
log "Step 1 done (exit $?)"

# Step 2: Check if any previously logged edges have settled
log "Step 2: Checking settlements..."
"$VENV" edge_logger.py --check >> "$LOG" 2>&1
log "Step 2 done (exit $?)"

# Step 3: Run polytope scanner for structural arb
log "Step 3: Polytope scan..."
"$VENV" polytope_scanner.py --json >> "$LOG" 2>&1
log "Step 3 done (exit $?)"

# Step 4: Collect real settlement data (for backtesting)
log "Step 4: Collecting settlements..."
"$VENV" real_data_collector.py >> "$LOG" 2>&1
log "Step 4 done (exit $?)"

# Step 5: Run performance tracker
log "Step 5: Performance tracking..."
"$VENV" performance_tracker.py >> "$LOG" 2>&1
log "Step 5 done (exit $?)"

log "=== Pipeline complete ==="
