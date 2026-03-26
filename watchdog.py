#!/usr/bin/env python3
"""
watchdog.py — Ippo bot health monitor. Runs every 30 min via systemd timer.

Checks:
1. Shadow files that could poison imports
2. Core services running
3. auto_trade has run recently (within 3 hours)
4. Balance is above minimum
5. Import health

Sends Telegram alert on any failure. Silent when healthy.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, "/opt/ippo")

PROJECT = Path("/opt/ippo")
OUTPUT = PROJECT / "output"


def check_shadow_files():
    """Check and auto-clean rogue shadow files."""
    issues = []
    for name in ["numpy.py", "dotenv.py", "pandas.py", "requests.py", "scipy.py"]:
        for check_dir in [PROJECT, PROJECT / "autoresearch"]:
            path = check_dir / name
            if path.exists():
                path.unlink()
                # Also clean pycache
                cache = check_dir / "__pycache__"
                for cached in cache.glob(f"{name.replace('.py', '')}*"):
                    cached.unlink()
                issues.append(f"Removed {path.relative_to(PROJECT)}")
    return issues


def check_imports():
    """Verify critical imports work."""
    try:
        result = subprocess.run(
            [str(PROJECT / ".venv/bin/python"), "-c",
             "import numpy; from dotenv import load_dotenv; import config; "
             "from kalshi_client import KalshiClient; print('OK')"],
            capture_output=True, text=True, timeout=15, cwd=str(PROJECT),
        )
        if result.stdout.strip() != "OK":
            return [f"Import failed: {result.stderr[:200]}"]
    except Exception as e:
        return [f"Import check error: {e}"]
    return []


def check_services():
    """Check critical services are running."""
    issues = []
    critical = ["ippo-arb-runner", "ippo-telegram", "ippo-health"]
    for svc in critical:
        try:
            result = subprocess.run(
                ["systemctl", "is-active", svc],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip() != "active":
                issues.append(f"{svc} is {result.stdout.strip()}")
        except Exception:
            issues.append(f"{svc} check failed")
    return issues


def check_auto_trade_recent():
    """Verify auto_trade has run in the last 3 hours."""
    issues = []
    # Check the systemd log for last successful run
    try:
        result = subprocess.run(
            ["journalctl", "-u", "ippo-auto-trade", "--no-pager", "-n", "5",
             "--output=short-iso"],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.strip().split("\n")
        success_lines = [l for l in lines if "Finished" in l or "SESSION COMPLETE" in l]
        if not success_lines:
            # Check for failures
            fail_lines = [l for l in lines if "Failed" in l or "FAILURE" in l]
            if fail_lines:
                issues.append(f"auto_trade is FAILING: {fail_lines[-1][:100]}")
    except Exception:
        pass

    # Also check the log file modification time
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = OUTPUT / f"auto_trade_{today}.log"
    if log_file.exists():
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)
        age = datetime.now(timezone.utc) - mtime
        if age > timedelta(hours=3):
            issues.append(f"auto_trade log is {age.seconds // 3600}h old (should be <3h)")
    else:
        # Check yesterday's
        pass  # First run of the day might not have today's log yet

    return issues


def check_balance():
    """Verify balance is above critical threshold."""
    issues = []
    try:
        from kalshi_client import KalshiClient
        client = KalshiClient()
        bal = client.get_balance()
        cash_cents = bal.get("balance", 0) if isinstance(bal, dict) else bal
        cash = cash_cents / 100.0
        if cash < 2.0:
            issues.append(f"LOW BALANCE: ${cash:.2f} (< $2 minimum)")
    except Exception as e:
        issues.append(f"Balance check failed: {e}")
    return issues


def send_alert(issues):
    """Send Telegram alert."""
    try:
        from alerts import send_telegram
        msg = "<b>IPPO WATCHDOG ALERT</b>\n\n"
        for issue in issues:
            msg += f"- {issue}\n"
        msg += f"\nTime: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        send_telegram(msg)
    except Exception:
        pass


if __name__ == "__main__":
    all_issues = []

    all_issues.extend(check_shadow_files())
    all_issues.extend(check_imports())
    all_issues.extend(check_services())
    all_issues.extend(check_auto_trade_recent())
    all_issues.extend(check_balance())

    if all_issues:
        print(f"WATCHDOG: {len(all_issues)} issues found")
        for issue in all_issues:
            print(f"  - {issue}")
        send_alert(all_issues)
    else:
        print(f"WATCHDOG: all healthy ({datetime.now(timezone.utc).strftime('%H:%M UTC')})")
