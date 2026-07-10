"""
Track 1: real historical MLB closing moneyline odds, 2021-04-01 through 2025-08-16.

Source: a public GitHub release (ArnavSaraogi/mlb-odds-scraper), itself scraped from
SportsBookReview -- per-game opening and "current" (effectively closing, since these
are historical/completed games) American moneyline odds across 4-6 sportsbooks.
SportsBookReviewsOnline's own official archive only goes through 2021, so this
GitHub dataset is what actually gets us 2022-2025 without a paid API. No API key,
no scraping infrastructure of our own needed. Repo disclaims "educational/
demonstrational purposes" -- used here for backtest research only, not resale.

This REPLACES the log5 proxy market from backtest.py wherever a game has real odds.
It does not cover late-Aug/Sept/Oct 2025, so the 2025 holdout backtest against real
odds is on a subset of the full holdout (see coverage report printed by main()).

Output: data/processed/real_odds.parquet -- one row per (game_pk, sportsbook) with
opening + closing American odds and no-vig fair probabilities, plus a same-shaped
consensus row (sportsbook="consensus") averaging the no-vig fair probability across
all books for that game.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ODDS_JSON = RAW / "external_odds" / "mlb_odds_dataset.json"

# odds-dataset shortName -> MLB Stats API abbreviation (only where they differ)
SHORTNAME_FIX = {"ARI": "AZ", "CHW": "CWS", "WAS": "WSH"}


def american_to_prob(odds: float) -> float:
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def american_to_decimal(odds: float) -> float:
    """Payout multiple including stake. Decimal odds are continuous (always > 1.0)
    and safe to average/median -- American odds are NOT: they jump from -100 to
    +100 with nothing in between, so a median taken directly in American-odds space
    can land in that gap (e.g. median([-105,-102,100]) = -102... but even-count
    medians can average a negative and a positive leg into a nonsense value like -1,
    which decodes as a 100x payout. Always convert to decimal before aggregating.
    """
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def decimal_to_american(decimal_odds: float) -> float:
    """Inverse of american_to_decimal, for displaying a representative consensus
    price after aggregating in decimal space."""
    profit = decimal_odds - 1
    return round(profit * 100, 1) if profit >= 1.0 else round(-100 / profit, 1)


def _team_id_crosswalk() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW.glob("teams_*.csv")):
        frames.append(pd.read_csv(path))
    teams = pd.concat(frames, ignore_index=True)
    return teams[["season", "abbreviation", "team_id"]].drop_duplicates()


def parse_odds_json() -> pd.DataFrame:
    with open(ODDS_JSON) as f:
        data = json.load(f)

    rows = []
    for date_key, games in data.items():
        for g in games:
            gv = g.get("gameView", {})
            if gv.get("gameType") != "R":
                continue
            home_short = SHORTNAME_FIX.get(gv["homeTeam"]["shortName"], gv["homeTeam"]["shortName"])
            away_short = SHORTNAME_FIX.get(gv["awayTeam"]["shortName"], gv["awayTeam"]["shortName"])
            # Use the scraper's own date bucket, not startDate's UTC date component --
            # UTC crosses midnight mid-game for US West Coast night games, which would
            # silently shift those games to the wrong calendar day and fail the join.
            official_date = date_key
            season = int(official_date[:4])
            for ml in g.get("odds", {}).get("moneyline", []):
                op, cl = ml.get("openingLine", {}), ml.get("currentLine", {})
                if cl.get("homeOdds") is None or cl.get("awayOdds") is None:
                    continue
                rows.append({
                    "official_date": official_date, "season": season,
                    "home_short": home_short, "away_short": away_short,
                    "home_score": gv.get("homeTeamScore"), "away_score": gv.get("awayTeamScore"),
                    "sportsbook": ml["sportsbook"],
                    "home_open_odds": op.get("homeOdds"), "away_open_odds": op.get("awayOdds"),
                    "home_close_odds": cl.get("homeOdds"), "away_close_odds": cl.get("awayOdds"),
                })
    return pd.DataFrame(rows)


def build_real_odds() -> pd.DataFrame:
    odds = parse_odds_json()

    for col in ["home_open_odds", "away_open_odds", "home_close_odds", "away_close_odds"]:
        odds[col] = pd.to_numeric(odds[col], errors="coerce")
    odds = odds.dropna(subset=["home_close_odds", "away_close_odds"])
    odds = odds[(odds["home_close_odds"] != 0) & (odds["away_close_odds"] != 0)]

    odds["home_close_raw_prob"] = odds["home_close_odds"].map(american_to_prob)
    odds["away_close_raw_prob"] = odds["away_close_odds"].map(american_to_prob)
    vig_sum = odds["home_close_raw_prob"] + odds["away_close_raw_prob"]
    odds["home_close_fair_prob"] = odds["home_close_raw_prob"] / vig_sum
    odds["away_close_fair_prob"] = odds["away_close_raw_prob"] / vig_sum

    has_open = odds["home_open_odds"].notna() & odds["away_open_odds"].notna()
    odds.loc[has_open, "home_open_raw_prob"] = odds.loc[has_open, "home_open_odds"].map(american_to_prob)
    odds.loc[has_open, "away_open_raw_prob"] = odds.loc[has_open, "away_open_odds"].map(american_to_prob)
    open_vig_sum = odds["home_open_raw_prob"] + odds["away_open_raw_prob"]
    odds["home_open_fair_prob"] = odds["home_open_raw_prob"] / open_vig_sum
    odds["away_open_fair_prob"] = odds["away_open_raw_prob"] / open_vig_sum

    odds["home_close_decimal"] = odds["home_close_odds"].map(american_to_decimal)
    odds["away_close_decimal"] = odds["away_close_odds"].map(american_to_decimal)

    crosswalk = _team_id_crosswalk()
    odds = odds.merge(crosswalk.rename(columns={"abbreviation": "home_short", "team_id": "home_team_id"}),
                       on=["season", "home_short"], how="left")
    odds = odds.merge(crosswalk.rename(columns={"abbreviation": "away_short", "team_id": "away_team_id"}),
                       on=["season", "away_short"], how="left")

    games = pd.read_parquet(PROCESSED / "games.parquet")[["game_pk", "season", "official_date", "home_team_id", "away_team_id"]]
    games["official_date"] = games["official_date"].dt.strftime("%Y-%m-%d")
    odds = odds.merge(games, on=["season", "official_date", "home_team_id", "away_team_id"], how="inner")

    # per-book rows
    per_book = odds.copy()
    per_book["is_consensus"] = False

    # consensus row per game: average no-vig fair probability across books, re-add a
    # standard -110/-110-equivalent vig for a representative consensus "price"
    consensus = odds.groupby("game_pk").agg(
        season=("season", "first"), official_date=("official_date", "first"),
        home_team_id=("home_team_id", "first"), away_team_id=("away_team_id", "first"),
        home_score=("home_score", "first"), away_score=("away_score", "first"),
        home_close_fair_prob=("home_close_fair_prob", "mean"),
        home_open_fair_prob=("home_open_fair_prob", "mean"),
        home_close_decimal=("home_close_decimal", "median"), away_close_decimal=("away_close_decimal", "median"),
        n_books=("sportsbook", "nunique"), n_books_open=("home_open_fair_prob", "count"),
    ).reset_index()
    consensus["away_close_fair_prob"] = 1 - consensus["home_close_fair_prob"]  # renormalize after averaging
    consensus["away_open_fair_prob"] = 1 - consensus["home_open_fair_prob"]
    consensus["sportsbook"] = "consensus"
    consensus["is_consensus"] = True

    out = pd.concat([per_book, consensus], ignore_index=True)
    out.to_parquet(PROCESSED / "real_odds.parquet", index=False)
    return out


if __name__ == "__main__":
    out = build_real_odds()
    consensus = out[out["is_consensus"]]
    games = pd.read_parquet(PROCESSED / "games.parquet")
    print(f"parsed odds rows: {len(out):,}  (consensus games: {len(consensus):,})")
    print(f"matched vs. total games.parquet: {consensus['game_pk'].nunique():,} / {games['game_pk'].nunique():,}")
    print("\nconsensus games by season:")
    print(consensus.groupby("season").size())
    print("\nsample:")
    with pd.option_context("display.width", 160):
        print(consensus.sample(min(5, len(consensus)), random_state=0)[
            ["game_pk", "season", "official_date", "home_team_id", "away_team_id",
             "home_close_fair_prob", "away_close_fair_prob", "n_books"]].to_string())
