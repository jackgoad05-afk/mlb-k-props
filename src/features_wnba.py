"""
Builds a leak-free, game-level feature matrix for both WNBA models (moneyline
classifier + totals regressor) from a single shared feature set -- team rolling
form captures both "who wins" and "how many combined points," so there's no
need for separate feature-engineering paths the way MLB needed pitcher-specific
K features on top of team features. Every diff is oriented so a MORE POSITIVE
value favors the home team (same convention as features.py).

No separate season-lag-only feature set here (unlike features.py's MLB feature
set, which deliberately kept season-lagged pitcher/team stats alongside the new
rolling ones in Session 2.1 specifically to prove rolling form was the fix for
a flat backtest). That comparison was already made and settled for MLB -- this
build starts from the rolling-form lesson already learned, not re-litigating it.

Output: data/processed/model_features_wnba.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rolling_wnba import build_team_rolling_form

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

FEATURE_COLS = [
    "home_rs_form", "away_rs_form", "rs_form_diff",
    "home_ra_form", "away_ra_form", "ra_form_diff",
    "home_win_form", "away_win_form", "win_form_diff",
    "home_trail_win_pct", "away_trail_win_pct", "trail_win_pct_diff",
    "home_games_played", "away_games_played",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "total_form_avg",
]

ID_COLS = ["game_id", "season", "official_date", "home_team_id", "away_team_id",
           "home_team_name", "away_team_name", "home_score", "away_score"]

TARGET_ML = "home_win"
TARGET_TOTALS = "total_points"

DEFAULT_REST_DAYS = 2.0
MAX_REST_DAYS = 10  # cap so a rare long All-Star-break-adjacent gap doesn't dominate the feature


def add_rolling_features(games: pd.DataFrame) -> pd.DataFrame:
    form = build_team_rolling_form(games)
    home_form = form[form["side"] == "home"].drop(columns=["side"]).rename(
        columns={c: f"home_{c}" for c in ["team_id", "games_played", "rs_form", "ra_form",
                                            "win_form", "trail_win_pct", "trail_n", "rest_days"]})
    away_form = form[form["side"] == "away"].drop(columns=["side"]).rename(
        columns={c: f"away_{c}" for c in ["team_id", "games_played", "rs_form", "ra_form",
                                            "win_form", "trail_win_pct", "trail_n", "rest_days"]})

    out = games.merge(home_form.drop(columns=["home_team_id"]), on="game_id", how="left")
    out = out.merge(away_form.drop(columns=["away_team_id"]), on="game_id", how="left")
    out["home_trail_win_pct"] = out["home_trail_win_pct"].fillna(out["home_win_form"])
    out["away_trail_win_pct"] = out["away_trail_win_pct"].fillna(out["away_win_form"])
    out["home_rest_days"] = out["home_rest_days"].fillna(DEFAULT_REST_DAYS).clip(upper=MAX_REST_DAYS)
    out["away_rest_days"] = out["away_rest_days"].fillna(DEFAULT_REST_DAYS).clip(upper=MAX_REST_DAYS)
    return out


def add_diffs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rs_form_diff"] = out["home_rs_form"] - out["away_rs_form"]        # more scoring is better
    out["ra_form_diff"] = out["away_ra_form"] - out["home_ra_form"]        # fewer points allowed is better
    out["win_form_diff"] = out["home_win_form"] - out["away_win_form"]
    out["trail_win_pct_diff"] = out["home_trail_win_pct"] - out["away_trail_win_pct"]
    out["rest_days_diff"] = out["home_rest_days"] - out["away_rest_days"]
    # Explicit scoring-environment feature for the totals model: each side's own
    # rolling (points scored + points allowed) averaged together. Weak but real
    # signal on its own (see model_wnba_totals.py's docstring) -- HGB with limited
    # 2021-2024 training data (~900 games) didn't reliably rediscover this sum from
    # the 4 separate rolling components, so it's engineered directly rather than
    # left for the model to find.
    out["total_form_avg"] = (out["home_rs_form"] + out["home_ra_form"]
                              + out["away_rs_form"] + out["away_ra_form"]) / 2
    return out


def build_features() -> pd.DataFrame:
    games = pd.read_parquet(PROCESSED / "games_wnba.parquet")
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games["total_points"] = games["home_score"] + games["away_score"]

    df = add_rolling_features(games)
    df = add_diffs(df)

    out = df[ID_COLS + [TARGET_ML, TARGET_TOTALS] + FEATURE_COLS].reset_index(drop=True)
    out.to_parquet(PROCESSED / "model_features_wnba.parquet", index=False)
    return out


if __name__ == "__main__":
    df = build_features()
    print(f"model_features_wnba: {len(df):,} rows, {len(FEATURE_COLS)} features")
    print(df.groupby("season").size())
    print(f"\nhome win rate: {df['home_win'].mean():.3f}")
    print(f"total_points: mean={df['total_points'].mean():.1f}  std={df['total_points'].std():.1f}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.sample(min(5, len(df)), random_state=0).to_string())
