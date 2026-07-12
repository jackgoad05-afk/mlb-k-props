"""
Shared The Odds API access: API key loading + thin HTTP helpers, used by both
daily_ks.py (today's live odds) and reconcile_ks.py (historical closing odds).

API key setup: set an ODDS_API_KEY environment variable, or create a `.env` file in
the project root containing:

    ODDS_API_KEY=your_key_here

Free signup (500 requests/month): https://the-odds-api.com/
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
USAGE_LOG_PATH = ROOT / "output" / "odds_api_usage.csv"
LOW_QUOTA_WARNING_THRESHOLD = 30  # units remaining -- print a loud warning below this


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key
    key = _read_dotenv(ENV_PATH).get("ODDS_API_KEY")
    if key:
        return key
    raise RuntimeError(
        f"ODDS_API_KEY not found.\n\n"
        f"Get a free key (500 requests/month) at https://the-odds-api.com/ , then either:\n"
        f"  1. Create/edit {ENV_PATH} and add this line:\n"
        f"       ODDS_API_KEY=your_key_here\n"
        f"  2. Or export it in your shell:\n"
        f"       export ODDS_API_KEY=your_key_here\n"
    )


class OddsAPIPlanError(RuntimeError):
    """Raised when an endpoint (e.g. /historical/) returns 401/403/422 -- usually a
    paid-tier-only feature the current API key's plan doesn't include."""


def _log_usage(url: str, used: int | None, remaining: int | None, last_cost: int | None) -> None:
    USAGE_LOG_PATH.parent.mkdir(exist_ok=True)
    row = pd.DataFrame([{
        "logged_at": datetime.now().isoformat(timespec="seconds"), "endpoint": url,
        "requests_used": used, "requests_remaining": remaining, "last_call_cost": last_cost,
    }])
    row.to_csv(USAGE_LOG_PATH, mode="a", header=not USAGE_LOG_PATH.exists(), index=False)
    if remaining is not None and remaining < LOW_QUOTA_WARNING_THRESHOLD:
        print(f"[WARNING] Odds API quota low: {remaining} requests remaining this month "
              f"(warning threshold: {LOW_QUOTA_WARNING_THRESHOLD}).")


def remaining_quota() -> int | None:
    """Last known remaining quota from the usage log, or None if nothing logged yet."""
    if not USAGE_LOG_PATH.exists():
        return None
    df = pd.read_csv(USAGE_LOG_PATH)
    if df.empty:
        return None
    return int(df["requests_remaining"].iloc[-1])


def _request(url: str, params: dict) -> dict | list:
    r = requests.get(url, params=params, timeout=30)
    used = int(r.headers["x-requests-used"]) if "x-requests-used" in r.headers else None
    remaining = int(r.headers["x-requests-remaining"]) if "x-requests-remaining" in r.headers else None
    last_cost = int(r.headers["x-requests-last"]) if "x-requests-last" in r.headers else None
    if used is not None or remaining is not None:
        _log_usage(url, used, remaining, last_cost)
    if r.status_code in (401, 403, 422):
        raise OddsAPIPlanError(
            f"{url} returned {r.status_code}: {r.text[:300]} -- this is usually a plan/quota "
            f"restriction (e.g. the /historical/ endpoints are paid-tier only)."
        )
    r.raise_for_status()
    return r.json()


def get_events(api_key: str, sport: str = "baseball_mlb") -> list[dict]:
    return _request(f"{ODDS_API_BASE}/sports/{sport}/events", {"apiKey": api_key})


def get_event_odds(api_key: str, event_id: str, markets: str, sport: str = "baseball_mlb",
                    regions: str = "us", odds_format: str = "american") -> dict:
    return _request(f"{ODDS_API_BASE}/sports/{sport}/events/{event_id}/odds", {
        "apiKey": api_key, "regions": regions, "markets": markets, "oddsFormat": odds_format,
    })


def get_bulk_odds(api_key: str, markets: str = "h2h", sport: str = "baseball_mlb",
                   regions: str = "us", odds_format: str = "american") -> list[dict]:
    """Featured markets (h2h/spreads/totals) for ALL of today's games in one call.
    Cost = markets x regions = 1 unit total regardless of how many games are returned
    -- unlike player props, which need the per-event endpoint (get_event_odds) at
    1 unit/event. Do not use this for pitcher_strikeouts or other additional markets;
    they aren't included in the bulk response."""
    return _request(f"{ODDS_API_BASE}/sports/{sport}/odds", {
        "apiKey": api_key, "regions": regions, "markets": markets, "oddsFormat": odds_format,
    })


def h2h_consensus_favorite(game_odds: dict) -> dict | None:
    """Best-effort no-vig consensus favorite for one bulk-odds game entry. Returns
    {favorite_team, favorite_fair_prob, n_books} or None if no h2h data present."""
    from fetch_odds import american_to_prob

    home, away = game_odds.get("home_team"), game_odds.get("away_team")
    home_probs, away_probs = [], []
    for book in game_odds.get("bookmakers", []):
        for market in book.get("markets", []):
            if market["key"] != "h2h":
                continue
            prices = {o["name"]: o["price"] for o in market["outcomes"]}
            if home in prices and away in prices:
                h_raw, a_raw = american_to_prob(prices[home]), american_to_prob(prices[away])
                vig = h_raw + a_raw
                home_probs.append(h_raw / vig)
                away_probs.append(a_raw / vig)
    if not home_probs:
        return None
    home_fair = sum(home_probs) / len(home_probs)
    away_fair = sum(away_probs) / len(away_probs)
    if home_fair >= away_fair:
        return {"favorite_team": home, "favorite_fair_prob": home_fair, "n_books": len(home_probs)}
    return {"favorite_team": away, "favorite_fair_prob": away_fair, "n_books": len(away_probs)}


def parse_pitcher_strikeouts_market(event_id: str, event_odds_json: dict) -> list[dict]:
    """Flatten one event's /odds response into per-book (player, line, over/under) rows."""
    rows = []
    for book in event_odds_json.get("bookmakers", []):
        for market in book.get("markets", []):
            if market["key"] != "pitcher_strikeouts":
                continue
            by_player: dict[str, dict] = {}
            for outcome in market["outcomes"]:
                by_player.setdefault(outcome["description"], {})[outcome["name"]] = outcome
            for player, sides in by_player.items():
                if "Over" not in sides or "Under" not in sides:
                    continue
                over, under = sides["Over"], sides["Under"]
                rows.append({
                    "event_id": event_id, "player_name": player, "line": over["point"],
                    "book": book["key"], "over_odds": over["price"], "under_odds": under["price"],
                })
    return rows


def consensus_over_under(odds_rows: pd.DataFrame) -> pd.DataFrame:
    """Per (player_name, line): avg no-vig fair prob across books, and a representative
    American price aggregated in DECIMAL space (see fetch_odds.american_to_decimal --
    averaging/medianing raw American odds directly is wrong, they jump from -100 to
    +100 with nothing between)."""
    from fetch_odds import american_to_decimal, american_to_prob, decimal_to_american

    df = odds_rows.copy()
    df["over_prob_raw"] = df["over_odds"].map(american_to_prob)
    df["under_prob_raw"] = df["under_odds"].map(american_to_prob)
    vig = df["over_prob_raw"] + df["under_prob_raw"]
    df["over_prob_fair"] = df["over_prob_raw"] / vig
    df["over_decimal"] = df["over_odds"].map(american_to_decimal)
    df["under_decimal"] = df["under_odds"].map(american_to_decimal)

    consensus = df.groupby(["event_id", "player_name", "line"]).agg(
        over_prob_fair=("over_prob_fair", "mean"), n_books=("book", "nunique"),
        over_decimal=("over_decimal", "median"), under_decimal=("under_decimal", "median"),
    ).reset_index()
    consensus["over_odds"] = consensus["over_decimal"].map(decimal_to_american)
    consensus["under_odds"] = consensus["under_decimal"].map(decimal_to_american)
    return consensus


def get_historical_events(api_key: str, iso_timestamp: str, sport: str = "baseball_mlb") -> list[dict]:
    d = _request(f"{ODDS_API_BASE}/historical/sports/{sport}/events", {
        "apiKey": api_key, "date": iso_timestamp,
    })
    return d.get("data", []) if isinstance(d, dict) else d


def get_historical_event_odds(api_key: str, event_id: str, iso_timestamp: str, markets: str,
                               sport: str = "baseball_mlb", regions: str = "us",
                               odds_format: str = "american") -> dict:
    d = _request(f"{ODDS_API_BASE}/historical/sports/{sport}/events/{event_id}/odds", {
        "apiKey": api_key, "regions": regions, "markets": markets, "oddsFormat": odds_format,
        "date": iso_timestamp,
    })
    return d.get("data", d) if isinstance(d, dict) else d
