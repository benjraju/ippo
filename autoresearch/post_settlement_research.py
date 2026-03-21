"""
autoresearch/post_settlement_research.py -- Trigger quick research after settlements.

Called by settlement_tracker.daily_summary() when new trades have settled.
Spawns a short WeatherResearch run (default 15 iterations) in the background,
using the freshly-written real_outcomes.json so the loop learns from the latest
settlement data immediately rather than waiting for the next scheduled cycle.

Skip conditions:
  - A research_loop process is already running (lock file check + pgrep fallback).
  - The settled strategy has no research loop (sports / btc / other are stubs for now).

State file:  autoresearch/settlement_trigger_state.json
Lock file:   output/research.lock   (written with the subprocess PID)
Trigger log: output/post_settlement_research.log
Output log:  output/post_settlement_research_out.log
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow importing config from the parent directory when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

LOCK_FILE = config.OUTPUT_DIR / "research.lock"
STATE_FILE = config.AUTORESEARCH_DIR / "settlement_trigger_state.json"
TRIGGER_LOG = config.OUTPUT_DIR / "post_settlement_research.log"
OUT_LOG = config.OUTPUT_DIR / "post_settlement_research_out.log"

# Default iterations for a post-settlement burst.
DEFAULT_ITERATIONS = 15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log(message: str) -> None:
    """Append a timestamped line to the trigger log."""
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(TRIGGER_LOG, "a") as f:
        f.write(f"[{ts}] {message}\n")


# ---------------------------------------------------------------------------
# Running-check logic
# ---------------------------------------------------------------------------

def is_research_running() -> bool:
    """
    Return True if a research_loop process is already active.

    Two checks are performed in order:
    1. Lock file written by a previous trigger (PID liveness check).
    2. pgrep scan for *any* process whose command line contains
       'research_loop' — this also catches the systemd ippo-autoresearch
       service so we never run concurrently with it.
    """
    # --- Lock file ---
    if LOCK_FILE.exists():
        try:
            data = json.loads(LOCK_FILE.read_text())
            pid = int(data["pid"])
            os.kill(pid, 0)   # signal 0 = existence check, raises if dead
            return True
        except (ProcessLookupError, OSError):
            # Process is gone; clean up the stale lock.
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            LOCK_FILE.unlink(missing_ok=True)

    # --- pgrep fallback: catches the systemd-managed autoresearch service ---
    try:
        result = subprocess.run(
            ["pgrep", "-f", "research_loop"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    return False


def _write_lock(pid: int) -> None:
    """Write lock file recording the PID of the spawned research process."""
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    LOCK_FILE.write_text(json.dumps({
        "pid": pid,
        "started": datetime.now(timezone.utc).isoformat(),
    }))


# ---------------------------------------------------------------------------
# State tracking (persists settled-trade count between service invocations)
# ---------------------------------------------------------------------------

def get_last_trigger_settled_count() -> int:
    """
    Return the total settled-trade count recorded at the last trigger.
    Returns 0 if the state file doesn't exist or is unreadable.
    """
    if STATE_FILE.exists():
        try:
            return int(json.loads(STATE_FILE.read_text()).get("settled_count", 0))
        except Exception:
            pass
    return 0


def _save_trigger_state(settled_count: int) -> None:
    """Persist the current settled-trade count so we don't re-trigger."""
    config.AUTORESEARCH_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "settled_count": settled_count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def trigger_post_settlement_research(
    new_settlement_count: int,
    settled_total: int,
    strategy: str = "weather",
    iterations: int = DEFAULT_ITERATIONS,
) -> bool:
    """
    Trigger a quick AutoResearch run after new settlements are detected.

    This function is designed to be called from settlement_tracker.daily_summary()
    right after real_outcomes.json has been written, so the research loop
    immediately trains on the freshest real-world data.

    Args:
        new_settlement_count: How many trades newly settled this run.
        settled_total:        Running total of settled trades after this batch.
        strategy:             Dominant strategy of the new settlements.
                              Only 'weather' has a research loop right now.
        iterations:           Number of research iterations to run (default 15).

    Returns:
        True if a research process was spawned, False if skipped.
    """
    # --- Strategy gate: only weather research is implemented ---
    if strategy not in ("weather",):
        _log(
            f"No research loop for strategy '{strategy}' — "
            f"skipping ({new_settlement_count} new settlements, "
            f"total settled={settled_total})"
        )
        _save_trigger_state(settled_total)
        return False

    # --- Skip if any research process is already active ---
    if is_research_running():
        _log(
            f"Skipped: research already running "
            f"({new_settlement_count} new {strategy} settlements, "
            f"total settled={settled_total})"
        )
        # Still update state so we don't accumulate a growing delta.
        _save_trigger_state(settled_total)
        return False

    # --- Spawn the research subprocess ---
    python = sys.executable
    cmd = [
        python, "-m", "autoresearch.research_loop",
        "--iterations", str(iterations),
        "--no-claude",
        "--use-real-data",
        "--quiet",
    ]

    try:
        out_log = open(OUT_LOG, "a")
        proc = subprocess.Popen(
            cmd,
            cwd=str(config.PROJECT_ROOT),
            stdout=out_log,
            stderr=subprocess.STDOUT,
            # Detach from the settlement service's process group so the
            # research run continues even after the oneshot service exits.
            start_new_session=True,
        )
        _write_lock(proc.pid)
        _save_trigger_state(settled_total)
        _log(
            f"Spawned {strategy} research: {iterations} iterations "
            f"(pid={proc.pid}, new_settlements={new_settlement_count}, "
            f"total_settled={settled_total})"
        )
        return True

    except Exception as exc:
        _log(f"Failed to spawn research subprocess: {exc}")
        return False


# ---------------------------------------------------------------------------
# Standalone usage (for testing / manual trigger)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manually trigger a post-settlement research run"
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help="Research iterations to run"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass the 'already running' check"
    )
    args = parser.parse_args()

    if args.force and LOCK_FILE.exists():
        LOCK_FILE.unlink()

    triggered = trigger_post_settlement_research(
        new_settlement_count=1,
        settled_total=get_last_trigger_settled_count() + 1,
        strategy="weather",
        iterations=args.iterations,
    )
    print("Research spawned." if triggered else "Skipped (already running or no weather settlements).")
