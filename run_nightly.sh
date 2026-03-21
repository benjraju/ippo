#!/bin/bash
# run_nightly.sh -- Kalshi trading bot: research, trade, settle
# Mirrors the 3-phase workflow from the launchd plist.
# Safe to run manually: uses dry-run mode (not --live).

set -o pipefail

PROJECT_DIR="/Users/benjamin/Desktop/ippo"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
LOG_DIR="${PROJECT_DIR}/autoresearch"
LOG_FILE="${LOG_DIR}/nightly_$(date +%Y%m%d_%H%M%S).log"

cd "${PROJECT_DIR}" || { echo "FATAL: cannot cd to ${PROJECT_DIR}"; exit 1; }

# Activate venv (for any subshells that need it)
source "${PROJECT_DIR}/.venv/bin/activate"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "=== Nightly run starting ==="
log "Project dir: ${PROJECT_DIR}"
log "Python: ${VENV_PYTHON}"

OVERALL_EXIT=0

# ─── Phase 1: AutoResearch (100 iterations, no Claude API) ───
log "--- Phase 1: AutoResearch ---"
if "${VENV_PYTHON}" -m autoresearch.research_loop --iterations 100 --no-claude >> "${LOG_FILE}" 2>&1; then
    log "Phase 1 completed successfully."
else
    log "Phase 1 FAILED (exit $?). Continuing to next phase."
    OVERALL_EXIT=1
fi

# ─── Phase 2: Auto-trade (DRY-RUN for safety) ───
log "--- Phase 2: Auto-trade (dry-run) ---"
if "${VENV_PYTHON}" auto_trade.py --dry-run --no-confirm >> "${LOG_FILE}" 2>&1; then
    log "Phase 2 completed successfully."
else
    log "Phase 2 FAILED (exit $?). Continuing to next phase."
    OVERALL_EXIT=1
fi

# ─── Phase 3: Settlement tracking ───
log "--- Phase 3: Settlement tracking ---"
if "${VENV_PYTHON}" -c 'from settlement_tracker import SettlementTracker; t=SettlementTracker(); t.daily_summary()' >> "${LOG_FILE}" 2>&1; then
    log "Phase 3 completed successfully."
else
    log "Phase 3 FAILED (exit $?)."
    OVERALL_EXIT=1
fi

log "=== Nightly run finished (overall exit: ${OVERALL_EXIT}) ==="
exit ${OVERALL_EXIT}
