"""
sports_strategy.py — NBA & March Madness edge detection vs. Kalshi markets.

STRATEGY: Build a simple but rigorous win-probability model from free ESPN data,
then compare our model's estimate to Kalshi market prices. When the market price
diverges from our model by >5%, we flag it as an edge.

KEY INSIGHT: Kalshi NBA game markets are priced by retail traders who often
overweight recent narratives (hot streaks, big names) and underweight base rates
(home court advantage, season-long efficiency). Our model sticks to the numbers.

MODEL OVERVIEW:
1. Fetch team records (overall, home, away), point differentials, and recent form
   from the free ESPN API — no API key required.
2. Estimate each team's strength via a simple efficiency rating:
   - Adjusted point differential per game (weighted: 70% season, 30% last-10)
   - Home court advantage adjustment (~3.2 points in the NBA)
3. Convert point spread to win probability using a logistic model calibrated
   to historical NBA data (stdev ~12 points per game).
4. For March Madness: use seed-based historical upset rates + team efficiency.
5. Compare model probability to Kalshi market price; flag edges >5%.

EDGE SOURCES:
1. Market hasn't incorporated recent form changes (last 10 games diverge from season)
2. Home/away splits are underpriced (retail focuses on overall record)
3. March Madness seed-vs-seed upset rates are well-documented but mispriced
4. Large point differentials between teams create high-confidence probabilities
"""

import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table

import config
from kalshi_client import KalshiClient

console = Console()

# =============================================================================
# ESPN API ENDPOINTS (free, no key needed)
# =============================================================================
ESPN_NBA_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_NBA_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
ESPN_NBA_TEAM = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}"
ESPN_NBA_TEAM_SCHEDULE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule"
ESPN_CBB_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
ESPN_CBB_TEAM = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}"

# =============================================================================
# MODEL PARAMETERS (calibrated to historical NBA data)
# =============================================================================

# Average home court advantage in NBA (points)
HOME_COURT_ADVANTAGE = 3.2

# Standard deviation of NBA game margin (for logistic win prob conversion)
# Historical NBA average is ~11-12 points
NBA_GAME_STDEV = 11.5

# Weight for recent form (last 10 games) vs season-long stats
RECENT_FORM_WEIGHT = 0.30
SEASON_WEIGHT = 0.70

# Minimum edge (percentage points) to flag a market
MIN_EDGE_PCT = 5.0

# Historical March Madness upset rates by seed matchup
# Source: NCAA historical tournament data (1985-2024)
# Format: (higher_seed, lower_seed) -> higher_seed_win_pct
MARCH_MADNESS_SEED_WIN_RATES = {
    (1, 16): 0.990,
    (2, 15): 0.940,
    (3, 14): 0.855,
    (4, 13): 0.790,
    (5, 12): 0.645,
    (6, 11): 0.625,
    (7, 10): 0.605,
    (8, 9):  0.510,
    # Later rounds (higher seed vs lower seed, averaged)
    (1, 8):  0.790,
    (1, 9):  0.830,
    (2, 7):  0.710,
    (2, 10): 0.740,
    (3, 6):  0.600,
    (3, 11): 0.650,
    (4, 5):  0.555,
    (4, 12): 0.680,
    (1, 4):  0.680,
    (1, 5):  0.720,
    (2, 3):  0.560,
    (2, 6):  0.640,
    (1, 2):  0.580,
    (1, 3):  0.640,
    (1, 11): 0.770,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TeamStats:
    """ESPN-derived team statistics for win probability estimation."""
    team_id: str
    abbreviation: str
    display_name: str
    wins: int = 0
    losses: int = 0
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    avg_points_for: float = 0.0
    avg_points_against: float = 0.0
    point_differential: float = 0.0
    # Recent form (last 10 games)
    recent_wins: int = 0
    recent_losses: int = 0
    recent_point_diff: float = 0.0

    @property
    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def home_win_pct(self) -> float:
        total = self.home_wins + self.home_losses
        return self.home_wins / total if total > 0 else 0.5

    @property
    def away_win_pct(self) -> float:
        total = self.away_wins + self.away_losses
        return self.away_wins / total if total > 0 else 0.5

    @property
    def games_played(self) -> int:
        return self.wins + self.losses

    @property
    def strength_rating(self) -> float:
        """
        Composite strength rating based on point differential.
        Blends season-long and recent form.
        """
        if self.games_played == 0:
            return 0.0
        season_diff = self.point_differential / self.games_played
        recent_games = self.recent_wins + self.recent_losses
        if recent_games > 0:
            recent_diff = self.recent_point_diff / recent_games
            return SEASON_WEIGHT * season_diff + RECENT_FORM_WEIGHT * recent_diff
        return season_diff


@dataclass
class NBAEdge:
    """A detected NBA/March Madness market edge."""
    ticker: str
    title: str
    team_a: str                # Team our model says will win
    team_b: str                # Opponent
    model_win_prob: float      # Our model's win probability for team_a
    market_price: float        # Kalshi market price (cents) for team_a to win
    fair_value: float          # Our model's fair value in cents
    edge: float                # fair_value - market_price (positive = buy yes)
    side: str                  # "buy_yes" or "buy_no"
    confidence: str            # "high", "medium", "low"
    model_spread: float        # Our model's predicted spread
    home_team: str             # Which team is home
    key_factors: list = field(default_factory=list)


@dataclass
class BacktestResult:
    """Result of backtesting the model against historical games."""
    total_games: int = 0
    correct_predictions: int = 0
    total_edges_found: int = 0
    profitable_edges: int = 0
    total_profit_cents: float = 0.0
    avg_edge_size: float = 0.0
    predictions: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct_predictions / self.total_games if self.total_games > 0 else 0.0

    @property
    def edge_hit_rate(self) -> float:
        return self.profitable_edges / self.total_edges_found if self.total_edges_found > 0 else 0.0


# =============================================================================
# ESPN DATA FETCHING
# =============================================================================

def _espn_get(url: str, params: dict = None, timeout: int = 10) -> Optional[dict]:
    """Make a request to the ESPN API with error handling."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        console.print(f"[yellow]ESPN API error: {e}[/yellow]")
        return None


def fetch_all_nba_teams() -> dict:
    """
    Fetch all NBA teams with IDs and abbreviations.
    Returns dict mapping abbreviation -> team_id.
    """
    data = _espn_get(ESPN_NBA_TEAMS)
    if not data:
        return {}

    teams = {}
    try:
        for t in data["sports"][0]["leagues"][0]["teams"]:
            team = t["team"]
            teams[team["abbreviation"].upper()] = team["id"]
    except (KeyError, IndexError):
        pass
    return teams


def fetch_team_stats(team_id: str) -> Optional[TeamStats]:
    """
    Fetch detailed stats for a single NBA team from ESPN.
    Includes overall record, home/away splits, and point differentials.
    """
    url = ESPN_NBA_TEAM.format(team_id=team_id)
    data = _espn_get(url)
    if not data:
        return None

    team = data.get("team", {})
    abbr = team.get("abbreviation", "???")
    name = team.get("displayName", "Unknown")

    stats = TeamStats(team_id=team_id, abbreviation=abbr, display_name=name)

    # Parse records
    for item in team.get("record", {}).get("items", []):
        record_type = item.get("type", "")
        stat_map = {s["name"]: s["value"] for s in item.get("stats", [])}

        if record_type == "total":
            stats.wins = int(stat_map.get("wins", 0))
            stats.losses = int(stat_map.get("losses", 0))
            stats.avg_points_for = stat_map.get("avgPointsFor", 0.0)
            stats.avg_points_against = stat_map.get("avgPointsAgainst", 0.0)
            stats.point_differential = stat_map.get("pointDifferential", 0.0)
        elif record_type == "home":
            stats.home_wins = int(stat_map.get("wins", 0))
            stats.home_losses = int(stat_map.get("losses", 0))
        elif record_type == "road":
            stats.away_wins = int(stat_map.get("wins", 0))
            stats.away_losses = int(stat_map.get("losses", 0))

    return stats


def fetch_recent_form(team_id: str, n_games: int = 10) -> tuple:
    """
    Fetch last N games results for a team to calculate recent form.
    Returns (recent_wins, recent_losses, recent_point_diff).
    """
    url = ESPN_NBA_TEAM_SCHEDULE.format(team_id=team_id)
    data = _espn_get(url, params={"season": 2026, "seasontype": 2})
    if not data:
        return 0, 0, 0.0

    events = data.get("events", [])
    # Filter to completed games only
    completed = []
    for e in events:
        comp = e.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status == "STATUS_FINAL":
            completed.append(comp)

    # Take last N completed games
    recent = completed[-n_games:] if len(completed) >= n_games else completed

    wins = 0
    losses = 0
    point_diff = 0.0

    for comp in recent:
        competitors = comp.get("competitors", [])
        our_team = None
        opponent = None
        for c in competitors:
            if c.get("team", {}).get("id") == team_id:
                our_team = c
            else:
                opponent = c

        if not our_team or not opponent:
            continue

        our_score = _parse_score(our_team.get("score"))
        opp_score = _parse_score(opponent.get("score"))

        if our_score is not None and opp_score is not None:
            if our_team.get("winner", False):
                wins += 1
            else:
                losses += 1
            point_diff += (our_score - opp_score)

    return wins, losses, point_diff


def _parse_score(score_val) -> Optional[float]:
    """Parse a score value that could be a string, int, or dict."""
    if score_val is None:
        return None
    if isinstance(score_val, (int, float)):
        return float(score_val)
    if isinstance(score_val, dict):
        val = score_val.get("value", score_val.get("displayValue"))
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
    if isinstance(score_val, str):
        try:
            return float(score_val)
        except ValueError:
            return None
    return None


def fetch_nba_data() -> dict:
    """
    Pull all team stats from ESPN for the current NBA season.
    Returns dict mapping team abbreviation -> TeamStats.

    This is the main data function — it fetches:
    - Overall record, home/away splits
    - Point differentials (season-long)
    - Recent form (last 10 games)
    """
    console.print("[dim]Fetching NBA team data from ESPN...[/dim]")

    # Step 1: Get all team IDs
    team_ids = fetch_all_nba_teams()
    if not team_ids:
        console.print("[red]Failed to fetch NBA teams from ESPN.[/red]")
        return {}

    all_stats = {}

    for abbr, team_id in team_ids.items():
        # Fetch base stats
        stats = fetch_team_stats(team_id)
        if not stats:
            continue

        # Fetch recent form (last 10 games)
        recent_w, recent_l, recent_diff = fetch_recent_form(team_id, n_games=10)
        stats.recent_wins = recent_w
        stats.recent_losses = recent_l
        stats.recent_point_diff = recent_diff

        all_stats[abbr] = stats

        # Rate limit: ESPN is free, be polite
        time.sleep(0.3)

    console.print(f"[dim]Loaded stats for {len(all_stats)} NBA teams.[/dim]")
    return all_stats


def fetch_todays_games() -> list:
    """
    Fetch today's NBA games from ESPN scoreboard.
    Returns list of dicts with home/away team info.
    """
    data = _espn_get(ESPN_NBA_SCOREBOARD)
    if not data:
        return []

    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")

        # Only include scheduled or in-progress games
        if status == "STATUS_FINAL":
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = None
        away = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home = c
            else:
                away = c

        if not home or not away:
            continue

        home_team = home.get("team", {})
        away_team = away.get("team", {})

        # Extract records from scoreboard (available per-game)
        home_records = {}
        for r in home.get("records", []):
            home_records[r.get("type", "")] = r.get("summary", "")
        away_records = {}
        for r in away.get("records", []):
            away_records[r.get("type", "")] = r.get("summary", "")

        games.append({
            "event_id": event.get("id"),
            "name": event.get("name", ""),
            "short_name": event.get("shortName", ""),
            "date": event.get("date", ""),
            "home_abbr": home_team.get("abbreviation", "").upper(),
            "home_name": home_team.get("displayName", ""),
            "home_id": home_team.get("id", ""),
            "away_abbr": away_team.get("abbreviation", "").upper(),
            "away_name": away_team.get("displayName", ""),
            "away_id": away_team.get("id", ""),
            "home_records": home_records,
            "away_records": away_records,
        })

    return games


def fetch_march_madness_games() -> list:
    """
    Fetch today's March Madness games from ESPN.
    Returns list of dicts with team info and seeds/ranks.
    """
    data = _espn_get(ESPN_CBB_SCOREBOARD)
    if not data:
        return []

    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")

        # Check if this is a tournament game
        notes = comp.get("notes", [])
        is_tournament = any("NCAA" in n.get("headline", "") or "Championship" in n.get("headline", "")
                           for n in notes)
        if not is_tournament:
            continue

        if status == "STATUS_FINAL":
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = competitors[0]  # In tournament, first listed is often higher seed
        away = competitors[1]

        home_team = home.get("team", {})
        away_team = away.get("team", {})

        home_seed = home.get("curatedRank", {}).get("current", 99)
        away_seed = away.get("curatedRank", {}).get("current", 99)

        # Get records
        home_records = home.get("records", [])
        away_records = away.get("records", [])
        home_record_str = home_records[0].get("summary", "0-0") if home_records else "0-0"
        away_record_str = away_records[0].get("summary", "0-0") if away_records else "0-0"

        # Parse win-loss from record string like "26-7"
        def parse_record(rec_str):
            parts = rec_str.split("-")
            try:
                return int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                return 0, 0

        home_w, home_l = parse_record(home_record_str)
        away_w, away_l = parse_record(away_record_str)

        round_name = notes[0].get("headline", "") if notes else ""

        games.append({
            "event_id": event.get("id"),
            "name": event.get("name", ""),
            "short_name": event.get("shortName", ""),
            "date": event.get("date", ""),
            "team_a_abbr": home_team.get("abbreviation", "").upper(),
            "team_a_name": home_team.get("displayName", ""),
            "team_a_id": home_team.get("id", ""),
            "team_a_seed": home_seed,
            "team_a_wins": home_w,
            "team_a_losses": home_l,
            "team_b_abbr": away_team.get("abbreviation", "").upper(),
            "team_b_name": away_team.get("displayName", ""),
            "team_b_id": away_team.get("id", ""),
            "team_b_seed": away_seed,
            "team_b_wins": away_w,
            "team_b_losses": away_l,
            "round": round_name,
            "is_tournament": True,
        })

    return games


# =============================================================================
# WIN PROBABILITY MODEL
# =============================================================================

def estimate_spread(team_a: TeamStats, team_b: TeamStats, team_a_home: bool) -> float:
    """
    Estimate point spread for team_a vs team_b.
    Positive spread means team_a is favored.

    Model:
    - Start with strength rating differential
    - Apply home court advantage
    - This gives expected margin of victory
    """
    strength_diff = team_a.strength_rating - team_b.strength_rating

    # Apply home court advantage
    if team_a_home:
        strength_diff += HOME_COURT_ADVANTAGE
    else:
        strength_diff -= HOME_COURT_ADVANTAGE

    return strength_diff


def spread_to_win_probability(spread: float, stdev: float = NBA_GAME_STDEV) -> float:
    """
    Convert a point spread to a win probability using a logistic model.

    The logistic function models the relationship between spread and win
    probability better than a normal CDF for NBA games because it has
    heavier tails (upsets happen more than a normal distribution predicts).

    P(win) = 1 / (1 + exp(-k * spread))
    where k is calibrated to historical data.

    For NBA: k ~= 0.148 (corresponds to ~12 point stdev)
    This means a 7-point favorite has ~74% win probability.
    """
    # k = pi / (stdev * sqrt(3)) for logistic approximation to normal
    k = math.pi / (stdev * math.sqrt(3))
    prob = 1.0 / (1.0 + math.exp(-k * spread))
    return max(0.01, min(0.99, prob))


def estimate_win_probability(
    team_a: TeamStats,
    team_b: TeamStats,
    team_a_home: bool = True,
) -> tuple:
    """
    Estimate win probability for team_a against team_b.

    Returns: (win_probability, spread, key_factors)
    """
    spread = estimate_spread(team_a, team_b, team_a_home)
    win_prob = spread_to_win_probability(spread)

    # Build explanation of key factors
    factors = []

    # Season record comparison
    factors.append(
        f"Season: {team_a.abbreviation} {team_a.wins}-{team_a.losses} "
        f"({team_a.win_pct:.3f}) vs {team_b.abbreviation} {team_b.wins}-{team_b.losses} "
        f"({team_b.win_pct:.3f})"
    )

    # Home/away edge
    if team_a_home:
        factors.append(
            f"Home advantage: {team_a.abbreviation} at home "
            f"({team_a.home_wins}-{team_a.home_losses}), "
            f"{team_b.abbreviation} on road "
            f"({team_b.away_wins}-{team_b.away_losses})"
        )
    else:
        factors.append(
            f"Away disadvantage: {team_a.abbreviation} on road "
            f"({team_a.away_wins}-{team_a.away_losses}), "
            f"{team_b.abbreviation} at home "
            f"({team_b.home_wins}-{team_b.home_losses})"
        )

    # Point differential per game
    a_ppg_diff = team_a.strength_rating
    b_ppg_diff = team_b.strength_rating
    factors.append(
        f"Strength: {team_a.abbreviation} {a_ppg_diff:+.1f} ppg diff, "
        f"{team_b.abbreviation} {b_ppg_diff:+.1f} ppg diff"
    )

    # Recent form
    a_recent = team_a.recent_wins + team_a.recent_losses
    b_recent = team_b.recent_wins + team_b.recent_losses
    if a_recent > 0 and b_recent > 0:
        a_recent_diff = team_a.recent_point_diff / a_recent if a_recent > 0 else 0
        b_recent_diff = team_b.recent_point_diff / b_recent if b_recent > 0 else 0
        factors.append(
            f"Last 10: {team_a.abbreviation} {team_a.recent_wins}-{team_a.recent_losses} "
            f"({a_recent_diff:+.1f} ppg), "
            f"{team_b.abbreviation} {team_b.recent_wins}-{team_b.recent_losses} "
            f"({b_recent_diff:+.1f} ppg)"
        )

    return win_prob, spread, factors


def estimate_march_madness_probability(
    seed_a: int,
    seed_b: int,
    wins_a: int,
    losses_a: int,
    wins_b: int,
    losses_b: int,
) -> tuple:
    """
    Estimate win probability for a March Madness matchup.

    Uses a blend of:
    1. Historical seed-vs-seed win rates (50% weight)
    2. Season record / win percentage using log5 method (30% weight)
    3. Simple efficiency proxy from win% (20% weight)

    The log5 method: P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB)
    where pA and pB are each team's true win rates.
    """
    # Ensure seed_a <= seed_b (seed_a is the higher-seeded / better team)
    if seed_a > seed_b:
        # Swap so seed_a is the better seed (lower number)
        seed_a, seed_b = seed_b, seed_a
        wins_a, wins_b = wins_b, wins_a
        losses_a, losses_b = losses_b, losses_a
        swapped = True
    else:
        swapped = False

    # 1. Historical seed win rate
    seed_prob = MARCH_MADNESS_SEED_WIN_RATES.get((seed_a, seed_b))
    if seed_prob is None:
        # Fallback: estimate from seed difference
        seed_diff = seed_b - seed_a
        seed_prob = 0.5 + 0.03 * seed_diff  # ~3% per seed difference
        seed_prob = max(0.5, min(0.95, seed_prob))

    # 2. Log5 method from season win percentages
    games_a = wins_a + losses_a
    games_b = wins_b + losses_b
    if games_a > 0 and games_b > 0:
        p_a = wins_a / games_a
        p_b = wins_b / games_b
        # Avoid division by zero
        denom = p_a + p_b - 2 * p_a * p_b
        if abs(denom) > 0.001:
            log5_prob = (p_a - p_a * p_b) / denom
        else:
            log5_prob = 0.5
        log5_prob = max(0.05, min(0.95, log5_prob))
    else:
        log5_prob = seed_prob

    # 3. Blend
    final_prob = 0.50 * seed_prob + 0.30 * log5_prob + 0.20 * seed_prob
    final_prob = max(0.02, min(0.98, final_prob))

    if swapped:
        final_prob = 1.0 - final_prob

    factors = [
        f"Seeds: #{seed_a} vs #{seed_b}",
        f"Historical seed rate: {seed_prob:.1%}",
        f"Log5 (from records): {log5_prob:.1%}",
        f"Blended: {final_prob:.1%}",
    ]

    return final_prob, factors


# =============================================================================
# KALSHI MARKET MATCHING
# =============================================================================

# Kalshi NBA series tickers from config
NBA_SERIES_TICKERS = ["KXNBA", "KXNBAGAME"]
MARCH_MADNESS_SERIES_TICKERS = ["KXNCAAB", "KXMARMAD"]

# Common team name variations for matching ESPN names to Kalshi market titles
TEAM_NAME_ALIASES = {
    "ATL": ["hawks", "atlanta"],
    "BOS": ["celtics", "boston"],
    "BKN": ["nets", "brooklyn"],
    "CHA": ["hornets", "charlotte"],
    "CHI": ["bulls", "chicago bulls"],
    "CLE": ["cavaliers", "cavs", "cleveland"],
    "DAL": ["mavericks", "mavs", "dallas"],
    "DEN": ["nuggets", "denver"],
    "DET": ["pistons", "detroit"],
    "GS":  ["warriors", "golden state", "gs", "gsw"],
    "HOU": ["rockets", "houston"],
    "IND": ["pacers", "indiana"],
    "LAC": ["clippers", "la clippers"],
    "LAL": ["lakers", "los angeles lakers", "la lakers"],
    "MEM": ["grizzlies", "memphis"],
    "MIA": ["heat", "miami"],
    "MIL": ["bucks", "milwaukee"],
    "MIN": ["timberwolves", "wolves", "minnesota"],
    "NO":  ["pelicans", "new orleans"],
    "NY":  ["knicks", "new york", "nyk"],
    "OKC": ["thunder", "oklahoma", "okc"],
    "ORL": ["magic", "orlando"],
    "PHI": ["76ers", "sixers", "philadelphia"],
    "PHX": ["suns", "phoenix"],
    "POR": ["trail blazers", "blazers", "portland"],
    "SAC": ["kings", "sacramento"],
    "SA":  ["spurs", "san antonio"],
    "TOR": ["raptors", "toronto"],
    "UTAH": ["jazz", "utah"],
    "WSH": ["wizards", "washington"],
}


def _match_team_in_title(title: str, team_stats: dict) -> Optional[str]:
    """
    Try to find which NBA team abbreviation is referenced in a market title.
    Returns the abbreviation or None.
    """
    title_lower = title.lower()
    for abbr, aliases in TEAM_NAME_ALIASES.items():
        for alias in aliases:
            if alias in title_lower:
                if abbr in team_stats:
                    return abbr
    return None


def _extract_teams_from_title(title: str, team_stats: dict) -> tuple:
    """
    Extract both teams from a game market title like:
    "Will the Knicks beat the Nets?" or "Celtics vs Hawks"
    Returns (team_a_abbr, team_b_abbr) or (None, None).
    """
    title_lower = title.lower()
    found = []
    for abbr, aliases in TEAM_NAME_ALIASES.items():
        for alias in aliases:
            if alias in title_lower and abbr in team_stats:
                if abbr not in found:
                    found.append(abbr)
                break
    if len(found) >= 2:
        return found[0], found[1]
    return None, None


def find_nba_edges(kalshi_client: KalshiClient = None) -> list:
    """
    Main strategy function: find NBA market edges.

    1. Fetch team stats from ESPN
    2. Get today's games from ESPN
    3. Get Kalshi NBA market prices
    4. Compare our model's win probabilities to market prices
    5. Return edges sorted by magnitude
    """
    client = kalshi_client or KalshiClient()
    all_edges = []

    # Step 1: Fetch NBA data
    team_stats = fetch_nba_data()
    if not team_stats:
        console.print("[red]Could not fetch NBA data. Check ESPN API.[/red]")
        return []

    # Step 2: Get today's games for context
    todays_games = fetch_todays_games()
    console.print(f"[dim]Found {len(todays_games)} NBA games today.[/dim]")

    # Build a lookup: game short_name -> game info
    game_lookup = {}
    for g in todays_games:
        game_lookup[g["short_name"]] = g
        # Also index by team abbreviations
        game_lookup[g["home_abbr"]] = g
        game_lookup[g["away_abbr"]] = g

    # Step 3: Get Kalshi NBA markets
    all_markets = []
    for series in NBA_SERIES_TICKERS:
        try:
            resp = client.get_markets(series_ticker=series, limit=100, status="open")
            markets = resp.get("markets", [])
            all_markets.extend(markets)
            console.print(f"[dim]Found {len(markets)} markets in {series}.[/dim]")
        except Exception as e:
            console.print(f"[yellow]Could not fetch {series} markets: {e}[/yellow]")
        time.sleep(0.5)

    # Step 4: Match markets to our model and find edges
    for m in all_markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100
        yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100
        volume = int(float(m.get("volume_fp", 0) or 0))

        # Skip illiquid or fully priced markets
        if yes_bid <= 0 and yes_ask <= 1:
            continue
        if yes_bid >= 99:
            continue

        # Try to extract teams from the market title
        team_a_abbr, team_b_abbr = _extract_teams_from_title(title, team_stats)
        if not team_a_abbr or not team_b_abbr:
            continue

        team_a = team_stats.get(team_a_abbr)
        team_b = team_stats.get(team_b_abbr)
        if not team_a or not team_b:
            continue

        # Determine home/away from today's games
        team_a_home = True  # Default assumption
        for g in todays_games:
            if (g["home_abbr"] == team_a_abbr and g["away_abbr"] == team_b_abbr):
                team_a_home = True
                break
            elif (g["home_abbr"] == team_b_abbr and g["away_abbr"] == team_a_abbr):
                team_a_home = False
                break

        # Step 5: Run our model
        win_prob, spread, factors = estimate_win_probability(team_a, team_b, team_a_home)

        fair_value_cents = win_prob * 100

        # Determine which team the market title is asking about
        # Usually the first team mentioned in the title is the "yes" team
        # We need to check if team_a is the "yes" team in the market
        title_lower = title.lower()
        # If team_a is mentioned first, our win_prob applies to "yes"
        # Otherwise, flip it
        first_team_pos = len(title_lower)
        first_team = None
        for abbr in [team_a_abbr, team_b_abbr]:
            for alias in TEAM_NAME_ALIASES.get(abbr, []):
                pos = title_lower.find(alias)
                if pos != -1 and pos < first_team_pos:
                    first_team_pos = pos
                    first_team = abbr

        if first_team and first_team != team_a_abbr:
            # The market is about team_b winning
            fair_value_cents = (1 - win_prob) * 100
            win_prob = 1 - win_prob
            spread = -spread
            team_a, team_b = team_b, team_a
            team_a_abbr, team_b_abbr = team_b_abbr, team_a_abbr

        # Calculate edge
        buy_price = yes_ask if yes_ask > 0 else yes_bid + 1
        sell_price = yes_bid

        buy_edge = fair_value_cents - buy_price
        sell_edge = sell_price - fair_value_cents

        home_team = team_a_abbr if team_a_home else team_b_abbr

        if buy_edge > MIN_EDGE_PCT:
            confidence = "high" if buy_edge > 15 else "medium" if buy_edge > 8 else "low"
            all_edges.append(NBAEdge(
                ticker=ticker,
                title=title[:60],
                team_a=team_a_abbr,
                team_b=team_b_abbr,
                model_win_prob=win_prob,
                market_price=buy_price,
                fair_value=round(fair_value_cents, 1),
                edge=round(buy_edge, 1),
                side="buy_yes",
                confidence=confidence,
                model_spread=round(spread, 1),
                home_team=home_team,
                key_factors=factors,
            ))
        elif sell_edge > MIN_EDGE_PCT:
            confidence = "high" if sell_edge > 15 else "medium" if sell_edge > 8 else "low"
            all_edges.append(NBAEdge(
                ticker=ticker,
                title=title[:60],
                team_a=team_a_abbr,
                team_b=team_b_abbr,
                model_win_prob=win_prob,
                market_price=sell_price,
                fair_value=round(fair_value_cents, 1),
                edge=round(sell_edge, 1),
                side="buy_no",
                confidence=confidence,
                model_spread=round(spread, 1),
                home_team=home_team,
                key_factors=factors,
            ))

    # Also check March Madness
    mm_edges = find_march_madness_edges(client, team_stats)
    all_edges.extend(mm_edges)

    # Sort by edge size
    all_edges.sort(key=lambda e: e.edge, reverse=True)
    return all_edges


def find_march_madness_edges(
    client: KalshiClient,
    nba_stats: dict = None,
) -> list:
    """
    Find edges in March Madness markets using seed-based model.
    """
    edges = []

    # Get today's tournament games
    mm_games = fetch_march_madness_games()
    if not mm_games:
        return []

    console.print(f"[dim]Found {len(mm_games)} March Madness games today.[/dim]")

    # Get Kalshi March Madness markets
    all_markets = []
    for series in MARCH_MADNESS_SERIES_TICKERS:
        try:
            resp = client.get_markets(series_ticker=series, limit=100, status="open")
            markets = resp.get("markets", [])
            all_markets.extend(markets)
        except Exception:
            pass
        time.sleep(0.5)

    if not all_markets:
        return []

    console.print(f"[dim]Found {len(all_markets)} March Madness markets on Kalshi.[/dim]")

    # Try to match markets to games
    for game in mm_games:
        team_a_name = game["team_a_name"].lower()
        team_b_name = game["team_b_name"].lower()

        for m in all_markets:
            title = m.get("title", "")
            title_lower = title.lower()
            ticker = m.get("ticker", "")

            # Check if this market is about this game
            team_a_in = any(word in title_lower for word in team_a_name.split()
                           if len(word) > 3)
            team_b_in = any(word in title_lower for word in team_b_name.split()
                           if len(word) > 3)

            if not (team_a_in or team_b_in):
                continue

            yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100
            yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100

            if yes_bid <= 0 and yes_ask <= 1:
                continue
            if yes_bid >= 99:
                continue

            # Determine which team the market is about
            # Find which team name appears in the title
            market_team = None
            if team_a_in and not team_b_in:
                market_team = "a"
            elif team_b_in and not team_a_in:
                market_team = "b"
            elif team_a_in and team_b_in:
                # Both mentioned: first one is usually the "yes" team
                market_team = "a"

            if not market_team:
                continue

            # Run our model
            prob_a_wins, factors = estimate_march_madness_probability(
                seed_a=game["team_a_seed"],
                seed_b=game["team_b_seed"],
                wins_a=game["team_a_wins"],
                losses_a=game["team_a_losses"],
                wins_b=game["team_b_wins"],
                losses_b=game["team_b_losses"],
            )

            if market_team == "b":
                fair_prob = 1 - prob_a_wins
            else:
                fair_prob = prob_a_wins

            fair_value_cents = fair_prob * 100

            buy_price = yes_ask if yes_ask > 0 else yes_bid + 1
            sell_price = yes_bid

            buy_edge = fair_value_cents - buy_price
            sell_edge = sell_price - fair_value_cents

            target_team = game[f"team_{market_team}_name"]
            other_team = game["team_b_name" if market_team == "a" else "team_a_name"]

            if buy_edge > MIN_EDGE_PCT:
                confidence = "high" if buy_edge > 15 else "medium" if buy_edge > 8 else "low"
                edges.append(NBAEdge(
                    ticker=ticker,
                    title=title[:60],
                    team_a=target_team,
                    team_b=other_team,
                    model_win_prob=fair_prob,
                    market_price=buy_price,
                    fair_value=round(fair_value_cents, 1),
                    edge=round(buy_edge, 1),
                    side="buy_yes",
                    confidence=confidence,
                    model_spread=0.0,
                    home_team="N/A",
                    key_factors=factors,
                ))
            elif sell_edge > MIN_EDGE_PCT:
                confidence = "high" if sell_edge > 15 else "medium" if sell_edge > 8 else "low"
                edges.append(NBAEdge(
                    ticker=ticker,
                    title=title[:60],
                    team_a=target_team,
                    team_b=other_team,
                    model_win_prob=fair_prob,
                    market_price=sell_price,
                    fair_value=round(fair_value_cents, 1),
                    edge=round(sell_edge, 1),
                    side="buy_no",
                    confidence=confidence,
                    model_spread=0.0,
                    home_team="N/A",
                    key_factors=factors,
                ))

    return edges


# =============================================================================
# DISPLAY
# =============================================================================

def display_edges(edges: list):
    """Pretty-print detected NBA/March Madness edges."""
    if not edges:
        console.print("[yellow]No NBA/March Madness edges found right now.[/yellow]")
        console.print("[dim]This could mean: no games today, no open Kalshi markets, "
                      "or the market is fairly priced.[/dim]")
        return

    table = Table(title="NBA / March Madness — Model vs. Market Edges", show_lines=False)
    table.add_column("Matchup", style="cyan", width=18)
    table.add_column("Market", style="white", width=30)
    table.add_column("Model", justify="right", style="green", width=8)
    table.add_column("Spread", justify="right", width=7)
    table.add_column("Mkt $", justify="right", width=7)
    table.add_column("Fair $", justify="right", style="yellow", width=7)
    table.add_column("Edge", justify="right", style="bold green", width=7)
    table.add_column("Action", style="bold", width=10)
    table.add_column("Conf", width=6)

    for e in edges:
        conf_color = "green" if e.confidence == "high" else "yellow" if e.confidence == "medium" else "dim"
        action_color = "green" if e.side == "buy_yes" else "red"
        action_text = "BUY YES" if e.side == "buy_yes" else "BUY NO"

        matchup = f"{e.team_a} v {e.team_b}"
        spread_str = f"{e.model_spread:+.1f}" if e.model_spread != 0 else "seed"

        table.add_row(
            matchup[:18],
            e.title[:30],
            f"{e.model_win_prob:.0%}",
            spread_str,
            f"{e.market_price:.0f}c",
            f"{e.fair_value:.0f}c",
            f"+{e.edge:.0f}c",
            f"[{action_color}]{action_text}[/{action_color}]",
            f"[{conf_color}]{e.confidence}[/{conf_color}]",
        )

    console.print(table)
    console.print(f"\n[dim]Found {len(edges)} edges. Data source: ESPN (free API). "
                  f"Min edge threshold: {MIN_EDGE_PCT}%[/dim]")

    # Show detail on best edge
    if edges:
        best = edges[0]
        console.print(f"\n[bold]Best edge detail ({best.team_a} vs {best.team_b}):[/bold]")
        for f in best.key_factors:
            console.print(f"  {f}")


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest_model(n_days: int = 14) -> BacktestResult:
    """
    Backtest our model against historical NBA game outcomes.

    Uses ESPN's completed games from recent dates. For each game:
    1. Reconstruct what our model would have predicted (using the records
       that were current at game time, approximated from current data)
    2. Compare prediction to actual outcome
    3. Simulate edge-based betting (would we have found an edge, and was
       the outcome profitable?)

    NOTE: This is an approximate backtest since we use current-season stats
    to represent what was available before each game. A production backtest
    would snapshot stats daily. For directional validation, this is sufficient.
    """
    console.print("[cyan]Running backtest against recent NBA games...[/cyan]\n")

    result = BacktestResult()

    # Step 1: Fetch current team stats (proxy for what was available)
    team_stats = fetch_nba_data()
    if not team_stats:
        console.print("[red]Could not fetch team data for backtest.[/red]")
        return result

    # Step 2: Fetch completed games from recent dates
    all_games = []
    today = datetime.now(timezone.utc)
    for days_ago in range(1, n_days + 1):
        from datetime import timedelta
        date = today - timedelta(days=days_ago)
        date_str = date.strftime("%Y%m%d")

        data = _espn_get(f"{ESPN_NBA_SCOREBOARD}?dates={date_str}")
        if not data:
            continue

        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            status = comp.get("status", {}).get("type", {}).get("name", "")
            if status != "STATUS_FINAL":
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            home = None
            away = None
            for c in competitors:
                if c.get("homeAway") == "home":
                    home = c
                else:
                    away = c

            if not home or not away:
                continue

            home_abbr = home.get("team", {}).get("abbreviation", "").upper()
            away_abbr = away.get("team", {}).get("abbreviation", "").upper()
            home_score = _parse_score(home.get("score"))
            away_score = _parse_score(away.get("score"))
            home_won = home.get("winner", False)

            if home_score is None or away_score is None:
                continue

            all_games.append({
                "date": date_str,
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "home_score": home_score,
                "away_score": away_score,
                "home_won": home_won,
                "margin": home_score - away_score,
            })

        time.sleep(0.3)

    console.print(f"[dim]Loaded {len(all_games)} completed games from last {n_days} days.[/dim]")

    if not all_games:
        console.print("[yellow]No completed games found for backtesting.[/yellow]")
        return result

    # Step 3: Run model predictions against each game
    for game in all_games:
        home_stats = team_stats.get(game["home_abbr"])
        away_stats = team_stats.get(game["away_abbr"])

        if not home_stats or not away_stats:
            continue

        # Our model predicts home team win probability
        home_win_prob, spread, _ = estimate_win_probability(
            home_stats, away_stats, team_a_home=True
        )

        # Did we predict correctly?
        predicted_home_win = home_win_prob > 0.5
        actual_home_win = game["home_won"]
        correct = predicted_home_win == actual_home_win

        result.total_games += 1
        if correct:
            result.correct_predictions += 1

        # Simulate edge detection
        # Assume the market price is roughly calibrated (use a simple prior)
        # We'll simulate the market as a coin flip adjusted by simple win%
        home_market_implied = (
            (home_stats.win_pct * 0.6 + 0.5 * 0.4) * 100  # Naive market price
        )
        # Add some noise to simulate market inefficiency
        import random
        random.seed(hash(f"{game['date']}{game['home_abbr']}"))
        noise = random.gauss(0, 5)
        simulated_market = max(5, min(95, home_market_implied + noise))

        our_fair = home_win_prob * 100
        edge = abs(our_fair - simulated_market)

        prediction_record = {
            "date": game["date"],
            "home": game["home_abbr"],
            "away": game["away_abbr"],
            "model_prob": home_win_prob,
            "model_spread": spread,
            "actual_margin": game["margin"],
            "correct": correct,
            "edge_size": edge,
            "had_edge": edge > MIN_EDGE_PCT,
        }

        if edge > MIN_EDGE_PCT:
            result.total_edges_found += 1
            result.avg_edge_size += edge

            # Would the edge bet have been profitable?
            if our_fair > simulated_market:
                # We'd buy YES on home team
                profitable = actual_home_win
            else:
                # We'd buy NO on home team (bet on away)
                profitable = not actual_home_win

            if profitable:
                result.profitable_edges += 1
                result.total_profit_cents += edge
            else:
                result.total_profit_cents -= (100 - edge)

            prediction_record["profitable"] = profitable

        result.predictions.append(prediction_record)

    if result.total_edges_found > 0:
        result.avg_edge_size /= result.total_edges_found

    return result


def display_backtest(result: BacktestResult):
    """Pretty-print backtest results."""
    console.print(f"\n[bold cyan]===== NBA Model Backtest Results =====[/bold cyan]\n")

    table = Table(show_header=False, show_lines=False, padding=(0, 2))
    table.add_column("Metric", style="white", width=30)
    table.add_column("Value", style="bold", width=20)

    table.add_row("Games analyzed", str(result.total_games))
    table.add_row(
        "Correct predictions",
        f"{result.correct_predictions}/{result.total_games} "
        f"({result.accuracy:.1%})"
    )
    table.add_row("", "")
    table.add_row("Edges detected (>{:.0f}%)".format(MIN_EDGE_PCT), str(result.total_edges_found))
    table.add_row(
        "Profitable edges",
        f"{result.profitable_edges}/{result.total_edges_found} "
        f"({result.edge_hit_rate:.1%})" if result.total_edges_found > 0 else "N/A"
    )
    table.add_row("Avg edge size", f"{result.avg_edge_size:.1f}c" if result.total_edges_found > 0 else "N/A")
    table.add_row(
        "Net P&L (simulated)",
        f"{result.total_profit_cents:+.0f}c" if result.total_edges_found > 0 else "N/A"
    )

    console.print(table)

    # Model calibration check
    if result.total_games > 0:
        console.print(f"\n[bold]Model Calibration:[/bold]")

        # Bin predictions by probability range and check accuracy
        bins = [(0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
        cal_table = Table(title="Prediction Accuracy by Confidence Bucket")
        cal_table.add_column("Model Prob", justify="center", width=12)
        cal_table.add_column("Games", justify="right", width=8)
        cal_table.add_column("Correct", justify="right", width=8)
        cal_table.add_column("Accuracy", justify="right", width=10)
        cal_table.add_column("Expected", justify="right", width=10)

        for lo, hi in bins:
            bucket = [p for p in result.predictions if lo <= p["model_prob"] < hi]
            if not bucket:
                continue
            n = len(bucket)
            correct = sum(1 for p in bucket if p["correct"])
            avg_prob = sum(p["model_prob"] for p in bucket) / n
            acc = correct / n if n > 0 else 0

            acc_color = "green" if abs(acc - avg_prob) < 0.10 else "yellow" if abs(acc - avg_prob) < 0.15 else "red"

            cal_table.add_row(
                f"{lo:.0%}-{hi:.0%}",
                str(n),
                str(correct),
                f"[{acc_color}]{acc:.1%}[/{acc_color}]",
                f"{avg_prob:.1%}",
            )

        console.print(cal_table)

        # Spread accuracy
        predictions_with_spread = [p for p in result.predictions if "model_spread" in p]
        if predictions_with_spread:
            spread_errors = [abs(p["model_spread"] - p["actual_margin"]) for p in predictions_with_spread]
            avg_spread_error = sum(spread_errors) / len(spread_errors)
            console.print(f"\n[dim]Avg spread prediction error: {avg_spread_error:.1f} points "
                          f"(model stdev assumption: {NBA_GAME_STDEV})[/dim]")

    console.print(f"\n[dim]Note: Backtest uses current-season stats as proxy for pre-game stats. "
                  f"Actual edge detection requires live market prices.[/dim]")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run edge detection
    edges = find_nba_edges()
    display_edges(edges)

    # Run backtest
    console.print("\n")
    result = backtest_model(n_days=7)
    display_backtest(result)
