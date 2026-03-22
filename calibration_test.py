"""
calibration_test.py -- Validate the core assumption of the weather strategy:
that NWS forecast errors are approximately normally distributed with a known
standard deviation.

Fetches historical NWS forecasts and actual observations for 6 cities,
computes forecast errors, and runs statistical tests to check normality
and calibration against the FORECAST_STDEV values used by candidate_strategy.py.

Usage:
    python calibration_test.py
"""

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# City configuration
# ---------------------------------------------------------------------------

# NWS grid points for forecasts (from weather_strategy.py)
NWS_GRIDS = {
    "NYC":     ("OKX", 34, 38),
    "Chicago": ("LOT", 72, 69),
    "Miami":   ("MFL", 106, 51),
    "LA":      ("LOX", 154, 44),
    "DC":      ("LWX", 97, 69),
    "Denver":  ("BOU", 74, 66),
}

# NWS observation station IDs (from settlement_tracker.py)
NWS_STATIONS = {
    "NYC":     "KNYC",   # Central Park
    "Chicago": "KMDW",   # Midway
    "Miami":   "KMIA",   # Miami Intl
    "LA":      "KLAX",   # LAX
    "DC":      "KDCA",   # Reagan National
    "Denver":  "KDEN",   # Denver Intl
}

# Assumed FORECAST_STDEV from candidate_strategy.py
# These are the values the trading model uses to price weather buckets.
ASSUMED_STDEV = {
    0: 1.5,   # Same day
    1: 2.5,   # Tomorrow
    2: 3.5,   # Day after tomorrow
    3: 4.5,   # 3 days out
}

HEADERS = {"User-Agent": "ippo-calibration/1.0"}
REQUEST_TIMEOUT = 15


# ---------------------------------------------------------------------------
# NWS API helpers
# ---------------------------------------------------------------------------

def fetch_forecast_for_date(city: str, date_str: str) -> Optional[float]:
    """
    Fetch the NWS forecast high temperature for a specific city and date.

    Note: The NWS forecast API only returns upcoming forecasts (next ~7 days),
    not historical forecasts. For a true calibration we would need archived
    forecasts. This function is used for recent/current dates only.

    Returns high temp in Fahrenheit, or None if unavailable.
    """
    grid = NWS_GRIDS.get(city)
    if not grid:
        return None

    office, x, y = grid
    url = f"https://api.weather.gov/gridpoints/{office}/{x},{y}/forecast"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        for period in data.get("properties", {}).get("periods", []):
            if period.get("isDaytime", False):
                start = period.get("startTime", "")
                if start and start[:10] == date_str:
                    return float(period["temperature"])
        return None
    except Exception as e:
        console.print(f"[dim]Forecast fetch failed for {city} on {date_str}: {e}[/dim]")
        return None


def fetch_observations_bulk(station_id: str, start_date: str, end_date: str) -> dict:
    """
    Fetch NWS observations for a station over a date range.

    The NWS observations API may only return recent data (typically last
    ~7-30 days depending on station). We fetch whatever is available.

    Returns dict mapping date_str -> high_temp_fahrenheit.
    """
    url = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {
        "start": f"{start_date}T00:00:00Z",
        "end": f"{end_date}T23:59:59Z",
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            return {}

        # Group observations by date, find max temp per day
        daily_temps: dict[str, list[float]] = {}
        daily_max24h: dict[str, list[float]] = {}

        for obs in features:
            props = obs.get("properties", {})
            timestamp = props.get("timestamp", "")
            if not timestamp:
                continue
            obs_date = timestamp[:10]

            # Try maxTemperatureLast24Hours first (more reliable)
            max24 = props.get("maxTemperatureLast24Hours", {})
            if max24 and max24.get("value") is not None:
                temp_c = max24["value"]
                daily_max24h.setdefault(obs_date, []).append(temp_c)

            # Also collect individual temperature readings as fallback
            temp = props.get("temperature", {})
            if temp and temp.get("value") is not None:
                daily_temps.setdefault(obs_date, []).append(temp["value"])

        # Build final result: prefer maxTemperatureLast24Hours, fall back to max(temps)
        result = {}
        all_dates = set(list(daily_max24h.keys()) + list(daily_temps.keys()))
        for date_str in all_dates:
            temp_c = None
            if date_str in daily_max24h:
                temp_c = max(daily_max24h[date_str])
            elif date_str in daily_temps:
                temp_c = max(daily_temps[date_str])

            if temp_c is not None:
                temp_f = round(temp_c * 9 / 5 + 32, 1)
                result[date_str] = temp_f

        return result

    except Exception as e:
        console.print(f"[dim]Observations fetch failed for {station_id}: {e}[/dim]")
        return {}


def fetch_forecast_griddata(city: str) -> dict:
    """
    Fetch raw gridpoint forecast data which may contain more historical
    data than the standard /forecast endpoint.

    Returns dict mapping date_str -> high_temp_fahrenheit.
    """
    grid = NWS_GRIDS.get(city)
    if not grid:
        return {}

    office, x, y = grid
    url = f"https://api.weather.gov/gridpoints/{office}/{x},{y}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        props = data.get("properties", {})
        max_temp = props.get("maxTemperature", {})
        values = max_temp.get("values", [])

        result = {}
        for entry in values:
            valid_time = entry.get("validTime", "")
            value = entry.get("value")
            if value is None:
                continue
            # validTime format: "2026-03-22T08:00:00+00:00/P1D"
            date_str = valid_time[:10]
            temp_f = round(value * 9 / 5 + 32, 1)
            result[date_str] = temp_f

        return result

    except Exception as e:
        console.print(f"[dim]Grid data fetch failed for {city}: {e}[/dim]")
        return {}


# ---------------------------------------------------------------------------
# Statistical helpers (no scipy needed)
# ---------------------------------------------------------------------------

def calc_skewness(data: np.ndarray) -> float:
    """Calculate skewness: E[(x - mu)^3] / sigma^3."""
    n = len(data)
    if n < 3:
        return 0.0
    mu = np.mean(data)
    sigma = np.std(data, ddof=0)
    if sigma == 0:
        return 0.0
    return float(np.mean(((data - mu) / sigma) ** 3))


def calc_kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis: E[(x - mu)^4] / sigma^4 - 3."""
    n = len(data)
    if n < 4:
        return 0.0
    mu = np.mean(data)
    sigma = np.std(data, ddof=0)
    if sigma == 0:
        return 0.0
    return float(np.mean(((data - mu) / sigma) ** 4) - 3.0)


def check_normality(skew: float, kurt: float) -> tuple[bool, str]:
    """
    Simple normality check without scipy.
    Approximately normal if |skewness| < 0.5 and |kurtosis| < 1.0.
    """
    skew_ok = abs(skew) < 0.5
    kurt_ok = abs(kurt) < 1.0

    if skew_ok and kurt_ok:
        return True, "PASS"
    elif abs(skew) < 1.0 and abs(kurt) < 2.0:
        return False, "MARGINAL"
    else:
        return False, "FAIL"


# ---------------------------------------------------------------------------
# Main calibration logic
# ---------------------------------------------------------------------------

def collect_city_data(city: str) -> dict:
    """
    Collect forecast and observation data for a city.
    Returns dict with errors, sample_size, and raw data.
    """
    station_id = NWS_STATIONS[city]

    today = datetime.now(timezone.utc).date()
    start_date = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    console.print(f"  [dim]Fetching observations for {city} ({station_id})...[/dim]")
    actuals = fetch_observations_bulk(station_id, start_date, end_date)
    time.sleep(0.5)

    console.print(f"  [dim]Fetching forecast grid data for {city}...[/dim]")
    forecasts = fetch_forecast_griddata(city)
    time.sleep(0.5)

    # Also try the standard forecast endpoint for current dates
    console.print(f"  [dim]Fetching current forecast for {city}...[/dim]")
    current_forecasts = {}
    grid = NWS_GRIDS.get(city)
    if grid:
        office, x, y = grid
        url = f"https://api.weather.gov/gridpoints/{office}/{x},{y}/forecast"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            for period in data.get("properties", {}).get("periods", []):
                if period.get("isDaytime", False):
                    start = period.get("startTime", "")
                    if start:
                        d = start[:10]
                        current_forecasts[d] = float(period["temperature"])
        except Exception:
            pass
    time.sleep(0.5)

    # Merge forecasts (current forecast overrides grid data for overlapping dates)
    all_forecasts = {**forecasts, **current_forecasts}

    # Find overlapping dates where we have both forecast AND actual
    errors = []
    matched_dates = []
    for date_str in sorted(set(all_forecasts.keys()) & set(actuals.keys())):
        forecast_temp = all_forecasts[date_str]
        actual_temp = actuals[date_str]
        error = actual_temp - forecast_temp
        errors.append(error)
        matched_dates.append({
            "date": date_str,
            "forecast": forecast_temp,
            "actual": actual_temp,
            "error": error,
        })

    return {
        "city": city,
        "station": station_id,
        "observation_days": len(actuals),
        "forecast_days": len(all_forecasts),
        "matched_days": len(errors),
        "errors": np.array(errors) if errors else np.array([]),
        "details": matched_dates,
    }


def analyze_city(data: dict) -> dict:
    """Run statistical analysis on a city's error data."""
    errors = data["errors"]
    n = len(errors)

    if n < 5:
        return {
            "city": data["city"],
            "station": data["station"],
            "sample_size": n,
            "observation_days": data["observation_days"],
            "forecast_days": data["forecast_days"],
            "status": "INSUFFICIENT DATA",
            "mean": None,
            "stdev": None,
            "skewness": None,
            "kurtosis": None,
            "normality": "N/A",
            "normal_pass": False,
        }

    mean_err = float(np.mean(errors))
    stdev_err = float(np.std(errors, ddof=1))  # sample stdev
    skew = calc_skewness(errors)
    kurt = calc_kurtosis(errors)
    normal_pass, normality_label = check_normality(skew, kurt)

    # Determine bias direction
    if abs(mean_err) < 0.5:
        bias_desc = "minimal bias"
    elif mean_err > 0:
        bias_desc = "slight warm bias"
    else:
        bias_desc = "slight cool bias"

    return {
        "city": data["city"],
        "station": data["station"],
        "sample_size": n,
        "observation_days": data["observation_days"],
        "forecast_days": data["forecast_days"],
        "status": "OK",
        "mean": mean_err,
        "stdev": stdev_err,
        "skewness": skew,
        "kurtosis": kurt,
        "normality": normality_label,
        "normal_pass": normal_pass,
        "bias_desc": bias_desc,
    }


def print_report(results: list[dict]) -> None:
    """Print a Rich-formatted calibration report."""
    console.print()
    console.print(
        Panel(
            "[bold white]FORECAST CALIBRATION TEST[/bold white]\n"
            "[dim]Validating NWS forecast error distribution assumptions[/dim]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )
    console.print()

    # Per-city results
    valid_results = []
    for r in results:
        if r["status"] == "INSUFFICIENT DATA":
            console.print(f"[bold]{r['city']}[/bold] ({r['station']})")
            console.print(
                f"  [yellow]INSUFFICIENT DATA[/yellow] -- "
                f"observations: {r['observation_days']}, "
                f"forecasts: {r['forecast_days']}, "
                f"matched: {r['sample_size']}"
            )
            console.print()
            continue

        valid_results.append(r)

        # Determine assumed stdev for comparison (use day-1 = 2.5 as baseline)
        assumed = ASSUMED_STDEV[1]
        stdev_diff_pct = ((r["stdev"] - assumed) / assumed) * 100
        if abs(stdev_diff_pct) < 10:
            stdev_label = f"[green]{stdev_diff_pct:+.0f}% vs model[/green]"
        elif abs(stdev_diff_pct) < 25:
            stdev_label = f"[yellow]{stdev_diff_pct:+.0f}% vs model[/yellow]"
        else:
            stdev_label = f"[red]{stdev_diff_pct:+.0f}% vs model[/red]"

        # Normality coloring
        if r["normality"] == "PASS":
            norm_color = "green"
        elif r["normality"] == "MARGINAL":
            norm_color = "yellow"
        else:
            norm_color = "red"

        # Skewness assessment
        if abs(r["skewness"]) < 0.5:
            skew_desc = "approximately symmetric"
        elif r["skewness"] > 0:
            skew_desc = "right-skewed"
        else:
            skew_desc = "left-skewed"

        # Kurtosis assessment
        if abs(r["kurtosis"]) < 1.0:
            kurt_desc = "approximately normal"
        elif r["kurtosis"] > 0:
            kurt_desc = "heavy-tailed"
        else:
            kurt_desc = "light-tailed"

        console.print(f"[bold]City: {r['city']}[/bold] ({r['station']})")
        console.print(f"  Sample size: {r['sample_size']} days")
        console.print(f"  Mean error:  {r['mean']:+.1f}F ({r['bias_desc']})")
        console.print(
            f"  Stdev:       {r['stdev']:.1f}F "
            f"(assumed: {assumed:.1f}F) -- {stdev_label}"
        )
        console.print(f"  Skewness:    {r['skewness']:.2f} ({skew_desc})")
        console.print(f"  Kurtosis:    {r['kurtosis']:.2f} ({kurt_desc})")
        console.print(f"  Normality:   [{norm_color}]{r['normality']}[/{norm_color}]")
        console.print()

    # Summary table
    if valid_results:
        table = Table(
            title="Summary: All Cities",
            border_style="bright_blue",
            show_lines=True,
        )
        table.add_column("City", style="bold")
        table.add_column("N", justify="right")
        table.add_column("Mean Err", justify="right")
        table.add_column("Stdev", justify="right")
        table.add_column("Assumed", justify="right")
        table.add_column("Diff %", justify="right")
        table.add_column("Skew", justify="right")
        table.add_column("Kurt", justify="right")
        table.add_column("Normal?", justify="center")

        for r in valid_results:
            assumed = ASSUMED_STDEV[1]
            diff_pct = ((r["stdev"] - assumed) / assumed) * 100
            if abs(diff_pct) < 10:
                diff_style = "green"
            elif abs(diff_pct) < 25:
                diff_style = "yellow"
            else:
                diff_style = "red"

            if r["normality"] == "PASS":
                norm_style = "green"
            elif r["normality"] == "MARGINAL":
                norm_style = "yellow"
            else:
                norm_style = "red"

            table.add_row(
                r["city"],
                str(r["sample_size"]),
                f"{r['mean']:+.1f}",
                f"{r['stdev']:.1f}",
                f"{assumed:.1f}",
                f"[{diff_style}]{diff_pct:+.0f}%[/{diff_style}]",
                f"{r['skewness']:.2f}",
                f"{r['kurtosis']:.2f}",
                f"[{norm_style}]{r['normality']}[/{norm_style}]",
            )

        console.print(table)
        console.print()

    # Overall verdict
    total_cities = len(results)
    valid_count = len(valid_results)
    normal_pass_count = sum(1 for r in valid_results if r["normal_pass"])
    insufficient_count = total_cities - valid_count

    # Stdev calibration assessment
    stdev_diffs = []
    for r in valid_results:
        assumed = ASSUMED_STDEV[1]
        diff_pct = abs((r["stdev"] - assumed) / assumed) * 100
        stdev_diffs.append(diff_pct)

    avg_stdev_diff = np.mean(stdev_diffs) if stdev_diffs else 0
    stdevs_need_update = avg_stdev_diff > 15

    console.print(
        Panel(
            "[bold white]OVERALL VERDICT[/bold white]",
            border_style="bright_blue",
            padding=(0, 2),
        )
    )

    if insufficient_count > 0:
        console.print(
            f"  [yellow]Data availability:[/yellow] "
            f"{valid_count}/{total_cities} cities had sufficient data "
            f"({insufficient_count} skipped)"
        )

    if valid_count > 0:
        # Normal assumption
        if normal_pass_count == valid_count:
            console.print(
                f"  [green]Normal assumption: PASS[/green] "
                f"({normal_pass_count}/{valid_count} cities)"
            )
        elif normal_pass_count >= valid_count * 0.5:
            console.print(
                f"  [yellow]Normal assumption: PARTIAL PASS[/yellow] "
                f"({normal_pass_count}/{valid_count} cities)"
            )
        else:
            console.print(
                f"  [red]Normal assumption: FAIL[/red] "
                f"({normal_pass_count}/{valid_count} cities)"
            )

        # Stdev calibration
        if stdevs_need_update:
            avg_actual = np.mean([r["stdev"] for r in valid_results])
            console.print(
                f"  [yellow]Stdev calibration: NEEDS UPDATE[/yellow] "
                f"(actual stdevs are {avg_stdev_diff:.0f}% off on average)"
            )
            console.print(
                f"  [yellow]Recommendation:[/yellow] Update FORECAST_STDEV values. "
                f"Observed avg stdev: {avg_actual:.1f}F vs assumed day-1: {ASSUMED_STDEV[1]:.1f}F"
            )

            # Suggest updated values based on ratio
            if avg_actual > 0 and ASSUMED_STDEV[1] > 0:
                ratio = avg_actual / ASSUMED_STDEV[1]
                console.print()
                console.print("  [bold]Suggested FORECAST_STDEV updates:[/bold]")
                for day, assumed_val in sorted(ASSUMED_STDEV.items()):
                    suggested = round(assumed_val * ratio, 1)
                    console.print(
                        f"    Day {day}: {assumed_val} -> {suggested}"
                    )
        else:
            console.print(
                f"  [green]Stdev calibration: OK[/green] "
                f"(within 15% of model, avg diff: {avg_stdev_diff:.0f}%)"
            )

        # Mean bias check
        mean_biases = [r["mean"] for r in valid_results]
        overall_bias = np.mean(mean_biases)
        if abs(overall_bias) > 1.0:
            console.print(
                f"  [yellow]Systematic bias: {overall_bias:+.1f}F[/yellow] "
                f"-- consider adding bias correction"
            )
        else:
            console.print(
                f"  [green]Systematic bias: {overall_bias:+.1f}F (acceptable)[/green]"
            )
    else:
        console.print(
            "  [red]No cities had sufficient data for analysis.[/red]\n"
            "  [dim]The NWS observations API may only return recent data.\n"
            "  Try running this script when more overlap between forecast and\n"
            "  observation windows is available.[/dim]"
        )

    console.print()


def main():
    """Run the full calibration test."""
    console.print()
    console.print("[bold bright_blue]Ippo Calibration Test[/bold bright_blue]")
    console.print("[dim]Fetching NWS forecast and observation data...[/dim]")
    console.print()

    cities = list(NWS_GRIDS.keys())
    all_data = []

    for city in cities:
        console.print(f"[bold]{city}[/bold]:")
        try:
            data = collect_city_data(city)
            all_data.append(data)
            console.print(
                f"  [dim]Got {data['observation_days']} observation days, "
                f"{data['forecast_days']} forecast days, "
                f"{data['matched_days']} matched[/dim]"
            )
        except Exception as e:
            console.print(f"  [red]ERROR: {e}[/red]")
            all_data.append({
                "city": city,
                "station": NWS_STATIONS[city],
                "observation_days": 0,
                "forecast_days": 0,
                "matched_days": 0,
                "errors": np.array([]),
                "details": [],
            })
        console.print()

    # Analyze each city
    results = [analyze_city(d) for d in all_data]

    # Print the report
    print_report(results)


if __name__ == "__main__":
    main()
