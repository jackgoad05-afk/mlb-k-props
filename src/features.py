"""
Session 2 (+ rolling-features revision): build a leak-free, game-level feature
matrix for the moneyline model.

Session 2 v1 lagged every pitcher/team stat a full season behind the game being
predicted (season-Y games used season-(Y-1) stats) to stay leak-free, and it
backtested flat: Brier score tied a naive baseline exactly, permutation importance
was near zero everywhere. The diagnosis was that season-level lag throws away
current-form signal that matters most for a single game.

This revision adds AS-OF-DATE ROLLING features (see rolling.py) built from real
per-start pitcher game logs and the team's own game-by-game results, both computed
strictly from games before the one being predicted -- still leak-free, but now
actually current. The old season-lagged pitcher FIP/K-BB% and team off_index/k_pct
stay in the feature set alongside the new rolling ones (not replaced) so the
backtest can show directly, via permutation importance, whether "current form"
was in fact the missing signal.

Statcast-derived pitcher quality (xwOBA against, barrel% allowed, whiff%, fastball
velo) is still season-lagged, not rolling -- a real per-start rolling version needs
either ~22 hours of per-pitcher Statcast pulls or ~5+ hours of chunked full-league
pulls (measured directly, not estimated), which wasn't run this session. See
CLAUDE.md. Park factor is the one feature that was never lagged: it's a property
of the physical ballpark, not that season's performance.

Output: data/processed/model_features_ml.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rolling import build_starter_rolling_form, build_team_rolling_form

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

PITCHER_STAT_COLS = ["fip", "k_bb_pct", "xwoba_against", "barrel_pct_allowed", "whiff_pct", "fastball_velo"]
MIN_IP_FOR_LEAGUE_AVG = 20


def _season_league_avgs(pitchers: pd.DataFrame) -> pd.DataFrame:
    qualified = pitchers[pitchers["IP"] >= MIN_IP_FOR_LEAGUE_AVG]
    return qualified.groupby("season")[PITCHER_STAT_COLS].mean()


def build_pitcher_features(games: pd.DataFrame, pitchers: pd.DataFrame, hands: pd.DataFrame) -> pd.DataFrame:
    pitchers = pitchers.merge(hands, on="mlbID", how="left")
    season_avgs = _season_league_avgs(pitchers)

    out = games.copy()
    out["lag_season"] = out["season"] - 1

    for side, id_col in [("home", "home_probable_pitcher_id"), ("away", "away_probable_pitcher_id")]:
        key = out[["lag_season", id_col]].rename(columns={"lag_season": "season", id_col: "mlbID"})
        merged = key.merge(pitchers, on=["season", "mlbID"], how="left")

        avg_lookup = season_avgs.reindex(key["season"]).reset_index(drop=True)
        for c in PITCHER_STAT_COLS:
            missing = merged[c].isna()
            merged.loc[missing, c] = avg_lookup.loc[missing, c].values
            out[f"{side}_{c}"] = merged[c].values

        out[f"{side}_starter_ip_lag"] = merged["IP"].fillna(0).values
        out[f"{side}_starter_missing_history"] = merged["Name"].isna().astype(int).values
        out[f"{side}_starter_throws"] = merged["throws"].values

    return out


def build_team_offense_features(df: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    league_avgs = teams.groupby(["season", "split"])[["off_index", "k_pct"]].mean()
    out = df.copy()

    pairs = [
        ("home", "home_team_id", "away_starter_throws"),
        ("away", "away_team_id", "home_starter_throws"),
    ]
    for off_side, team_id_col, opp_hand_col in pairs:
        split = out[opp_hand_col].map({"L": "vs_lhp", "R": "vs_rhp"}).fillna("overall")
        key = pd.DataFrame({
            "season": out["lag_season"], "team_id": out[team_id_col], "split": split,
        })
        merged = key.merge(teams, on=["season", "team_id", "split"], how="left")

        avg_lookup = league_avgs.reindex(pd.MultiIndex.from_frame(key[["season", "split"]])).reset_index(drop=True)
        for c in ["off_index", "k_pct"]:
            missing = merged[c].isna()
            merged.loc[missing, c] = avg_lookup.loc[missing, c].values

        out[f"{off_side}_off_index"] = merged["off_index"].values
        out[f"{off_side}_off_kpct"] = merged["k_pct"].values
        out[f"{off_side}_off_pa_lag"] = merged["plateAppearances"].fillna(0).values

    return out


def add_starter_rolling_features(df: pd.DataFrame, game_logs: pd.DataFrame, pitchers: pd.DataFrame) -> pd.DataFrame:
    form = build_starter_rolling_form(game_logs, pitchers)
    out = df.copy()
    for side, id_col in [("home", "home_probable_pitcher_id"), ("away", "away_probable_pitcher_id")]:
        key = out[["game_pk", id_col]].rename(columns={id_col: "mlbID"})
        merged = key.merge(form, on=["game_pk", "mlbID"], how="left")
        out[f"{side}_trail_fip"] = merged["trail_fip"].fillna(out[f"{side}_fip"]).values
        out[f"{side}_trail_k_bb_pct"] = merged["trail_k_bb_pct"].fillna(out[f"{side}_k_bb_pct"]).values
        out[f"{side}_starts_this_season"] = merged["starts_this_season_so_far"].fillna(0).values
        out[f"{side}_days_rest"] = merged["days_rest"].fillna(5.0).clip(upper=20).values
    return out


def add_team_rolling_features(df: pd.DataFrame, games_all_seasons: pd.DataFrame) -> pd.DataFrame:
    form = build_team_rolling_form(games_all_seasons)
    home_form = form[form["side"] == "home"].drop(columns=["side"]).rename(
        columns={c: f"home_{c}" for c in ["team_id", "games_played", "rs_form", "ra_form", "win_form", "trail_win_pct", "trail_n"]})
    away_form = form[form["side"] == "away"].drop(columns=["side"]).rename(
        columns={c: f"away_{c}" for c in ["team_id", "games_played", "rs_form", "ra_form", "win_form", "trail_win_pct", "trail_n"]})

    out = df.merge(home_form.drop(columns=["home_team_id"]), on="game_pk", how="left")
    out = out.merge(away_form.drop(columns=["away_team_id"]), on="game_pk", how="left")
    out["home_trail_win_pct"] = out["home_trail_win_pct"].fillna(out["home_win_form"])
    out["away_trail_win_pct"] = out["away_trail_win_pct"].fillna(out["away_win_form"])
    return out


def add_park_factor(df: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    park = teams[teams["split"] == "overall"][["season", "team_id", "park_factor_runs"]]
    out = df.merge(park, left_on=["season", "home_team_id"], right_on=["season", "team_id"], how="left")
    out = out.drop(columns=["team_id"])
    out["park_factor_runs"] = out["park_factor_runs"].fillna(100.0)
    return out


def add_diffs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Every diff is oriented so that a MORE POSITIVE value favors the home team.
    out["fip_diff"] = out["away_fip"] - out["home_fip"]                                   # lower FIP is better
    out["k_bb_pct_diff"] = out["home_k_bb_pct"] - out["away_k_bb_pct"]                     # higher is better
    out["xwoba_against_diff"] = out["away_xwoba_against"] - out["home_xwoba_against"]      # lower is better
    out["barrel_pct_allowed_diff"] = out["away_barrel_pct_allowed"] - out["home_barrel_pct_allowed"]  # lower is better
    out["whiff_pct_diff"] = out["home_whiff_pct"] - out["away_whiff_pct"]                  # higher is better
    out["fastball_velo_diff"] = out["home_fastball_velo"] - out["away_fastball_velo"]      # higher is better
    out["off_index_diff"] = out["home_off_index"] - out["away_off_index"]                  # higher is better
    out["off_kpct_diff"] = out["away_off_kpct"] - out["home_off_kpct"]                     # lower K% is better for offense

    # rolling / current-form diffs (same "positive favors home" orientation)
    out["trail_fip_diff"] = out["away_trail_fip"] - out["home_trail_fip"]
    out["trail_k_bb_pct_diff"] = out["home_trail_k_bb_pct"] - out["away_trail_k_bb_pct"]
    out["days_rest_diff"] = out["home_days_rest"] - out["away_days_rest"]
    out["rs_form_diff"] = out["home_rs_form"] - out["away_rs_form"]                        # more runs scored is better
    out["ra_form_diff"] = out["away_ra_form"] - out["home_ra_form"]                        # fewer runs allowed is better
    out["win_form_diff"] = out["home_win_form"] - out["away_win_form"]
    out["trail_win_pct_diff"] = out["home_trail_win_pct"] - out["away_trail_win_pct"]
    return out


FEATURE_COLS = [
    # season-lagged pitcher quality (Session 2 v1 -- kept for comparison against rolling)
    "home_fip", "away_fip", "fip_diff",
    "home_k_bb_pct", "away_k_bb_pct", "k_bb_pct_diff",
    "home_xwoba_against", "away_xwoba_against", "xwoba_against_diff",
    "home_barrel_pct_allowed", "away_barrel_pct_allowed", "barrel_pct_allowed_diff",
    "home_whiff_pct", "away_whiff_pct", "whiff_pct_diff",
    "home_fastball_velo", "away_fastball_velo", "fastball_velo_diff",
    # season-lagged team offense splits (Session 2 v1)
    "home_off_index", "away_off_index", "off_index_diff",
    "home_off_kpct", "away_off_kpct", "off_kpct_diff",
    "home_starter_ip_lag", "away_starter_ip_lag",
    "home_starter_missing_history", "away_starter_missing_history",
    "home_off_pa_lag", "away_off_pa_lag",
    "park_factor_runs",
    # rolling / current-form pitcher features (new this revision)
    "home_trail_fip", "away_trail_fip", "trail_fip_diff",
    "home_trail_k_bb_pct", "away_trail_k_bb_pct", "trail_k_bb_pct_diff",
    "home_starts_this_season", "away_starts_this_season",
    "home_days_rest", "away_days_rest", "days_rest_diff",
    # rolling / current-form team features (new this revision)
    "home_rs_form", "away_rs_form", "rs_form_diff",
    "home_ra_form", "away_ra_form", "ra_form_diff",
    "home_win_form", "away_win_form", "win_form_diff",
    "home_trail_win_pct", "away_trail_win_pct", "trail_win_pct_diff",
    "home_games_played", "away_games_played",
]

ID_COLS = ["game_pk", "season", "official_date", "home_team_id", "away_team_id",
           "home_team_name", "away_team_name", "home_score", "away_score", "total_runs"]

TARGET_COL = "home_win"


def _load_all_game_logs() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW.glob("pitcher_gamelogs_*.csv")):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def build_features() -> pd.DataFrame:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    pitchers = pd.read_parquet(PROCESSED / "pitcher_season.parquet")
    teams = pd.read_parquet(PROCESSED / "team_season.parquet")
    hands = pd.read_csv(RAW / "pitcher_handedness.csv")
    game_logs = _load_all_game_logs()

    df = build_pitcher_features(games, pitchers, hands)
    df = build_team_offense_features(df, teams)
    df = add_starter_rolling_features(df, game_logs, pitchers)
    df = add_team_rolling_features(df, games)
    df = add_park_factor(df, teams)
    df = add_diffs(df)

    n_before = len(df)
    df = df.dropna(subset=["home_probable_pitcher_id", "away_probable_pitcher_id"])
    dropped = n_before - len(df)
    if dropped:
        print(f"[features] dropped {dropped} games with no probable-pitcher record (no starter -> no pitcher features)")

    out = df[ID_COLS + [TARGET_COL] + FEATURE_COLS].reset_index(drop=True)
    out.to_parquet(PROCESSED / "model_features_ml.parquet", index=False)
    return out


if __name__ == "__main__":
    df = build_features()
    print(f"model_features_ml: {len(df):,} rows, {len(FEATURE_COLS)} features")
    print(df.groupby("season").size())
    print(f"\nmissing starter history rate: home={df['home_starter_missing_history'].mean():.3f} "
          f"away={df['away_starter_missing_history'].mean():.3f}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.sample(min(5, len(df)), random_state=0).to_string())
