"""
health_check.py -- Simple HTTP health check endpoint for uptime monitoring.

Returns JSON with service status, last run times, and account info.
Monitored by UptimeRobot (free tier) to alert on downtime.

Usage:
    python health_check.py              # Start on port 8787
    python health_check.py --port 9090  # Custom port

Endpoints:
    GET /health  → {"status": "ok", ...}  (200 if healthy, 503 if degraded)
    GET /        → same as /health
"""

import json
import os
import subprocess
import sys
import argparse
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config

OUTPUT = config.OUTPUT_DIR
AUTORESEARCH = config.AUTORESEARCH_DIR


def get_service_status(service_name: str) -> dict:
    """Check if a systemd service is active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True, text=True, timeout=5,
        )
        active = result.stdout.strip() == "active"
        return {"running": active, "state": result.stdout.strip()}
    except Exception:
        return {"running": False, "state": "unknown"}


def get_last_log_time(log_pattern: str) -> str:
    """Get the most recent log file's last modified time."""
    try:
        logs = sorted(OUTPUT.glob(log_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            mtime = logs[0].stat().st_mtime
            return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        pass
    return ""


def get_error_count_today() -> int:
    """Count errors in today's arb runner log."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = OUTPUT / f"arb_runner_{today}.log"
    if not log_file.exists():
        return 0
    try:
        text = log_file.read_text()
        return text.count("ERROR")
    except Exception:
        return -1


def get_autoresearch_stats() -> dict:
    """Get autoresearch iteration count and last improvement."""
    results_log = AUTORESEARCH / "results.log"
    if not results_log.exists():
        return {"iterations": 0, "improvements": 0, "last_run": ""}
    try:
        lines = results_log.read_text().strip().split("\n")
        iterations = len(lines)
        improvements = sum(1 for l in lines if '"kept": true' in l)
        last_entry = json.loads(lines[-1]) if lines else {}
        return {
            "iterations": iterations,
            "improvements": improvements,
            "last_run": last_entry.get("timestamp", ""),
        }
    except Exception:
        return {"iterations": 0, "improvements": 0, "last_run": ""}


def get_account_balance() -> float:
    """Try to get account balance. Returns -1 on failure."""
    try:
        from kalshi_client import KalshiClient
        client = KalshiClient()
        bal = client.get_balance()
        return bal.get("balance", 0) / 100.0
    except Exception:
        return -1


def build_health_response() -> dict:
    """Build the full health check response."""
    services = {
        "arb_runner": get_service_status("ippo-arb-runner"),
        "auto_trade": get_service_status("ippo-auto-trade"),
        "autoresearch": get_service_status("ippo-autoresearch"),
        "settlement": get_service_status("ippo-settlement"),
    }

    all_critical_running = services["arb_runner"]["running"]
    error_count = get_error_count_today()
    balance = get_account_balance()

    status = "ok"
    if not all_critical_running:
        status = "degraded"
    if error_count > 50:
        status = "degraded"
    if balance == 0:
        status = "critical"

    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": services,
        "balance": balance,
        "errors_today": error_count,
        "last_auto_trade": get_last_log_time("auto_trade_*.log"),
        "last_arb_scan": get_last_log_time("arb_runner_*.log"),
        "autoresearch": get_autoresearch_stats(),
    }


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/health"):
            data = build_health_response()
            status_code = 200 if data["status"] == "ok" else 503
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default logging


def main():
    parser = argparse.ArgumentParser(description="Ippo Health Check Server")
    parser.add_argument("--port", type=int, default=8787, help="Port (default: 8787)")
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), HealthHandler)
    print(f"Health check server running on port {args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
