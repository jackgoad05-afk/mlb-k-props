"""
Track 2: build a leak-free, per-start feature matrix for the pitcher strikeout model.

One row per start (not per game -- a pitcher makes ~30 starts/season, so this is a
different population and grain than the moneyline model's per-game table). Every
feature is computed strictly from information available BEFORE that start: rolling
form from real per-start game logs (rolling.build_starter_k_form), season-lagged
Statcast/bref pitcher quality (reused from the moneyline pipeline's pitcher_season
table), and the opposing lineup's season-lagged K% split by the starter's own
throwing hand.

Deliberately NOT included, and why:
  - Opposing lineup K% is season-lagged, not rolling. A rolling version needs
    per-game team boxscores (~12,000 additional API calls) not pulled this session.
  - Umpire K tendencies: skipped. Same per-game-boxscore cost as the item above,
    and the spec only asked for it "if cheap" -- it isn't, at this data source.
  - Whiff%, exact same-game xwOBA-quality Statcast features: still season-lagged
    (see CLAUDE.md's Statcast-rolling cost estimate from the moneyline work --
    ~5+ hours, not run this session).

Output: data/processed/model_features_ks.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rolling import build_starter_k_form

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

TARGET_COL = "so"

FEATURE_COLS = [
    "trail_k_per9_3s", "trail_bb_per9_3s", "trail_k_per9_30d",
    "trail_ip_per_start", "trail_pitch_count_avg",
    "starts_this_season_so_far", "days_rest",
    "season_lag_whiff_pct", "season_lag_k_bb_pct", "season_lag_fastball_velo",
    "opp_off_kpct",
]
# used as a GLM offset (log-exposure), not a regular covariate -- see model_ks.py
EXPOSURE_COL = "trail_ip_per_start"

ID_COLS = ["mlbID", "game_pk", "season", "official_date", "Name", "is_home", "opponent_team_id"]


def _load_all_game_logs() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(RAW.glob("pitcher_gamelogs_*.csv"))]
    return pd.concat(frames, ignore_index=True)


def build_features() -> pd.DataFrame:
    game_logs = _load_all_game_logs()
    pitchers = pd.read_parquet(PROCESSED / "pitcher_season.parquet")
    teams = pd.read_parquet(PROCESSED / "team_season.parquet")
    hands = pd.read_csv(RAW / "pitcher_handedness.csv")

    form = build_starter_k_form(game_logs, pitchers)

    df = game_logs.merge(form, on=["mlbID", "game_pk", "season"], how="inner")
    df = df.merge(hands, on="mlbID", how="left")
    df = df.merge(pitchers[["season", "mlbID", "Name"]], on=["season", "mlbID"], how="left")

    # season-lagged pitcher quality (Statcast + bref), same lag convention as the
    # moneyline model: season-Y start uses season-(Y-1) aggregates.
    lag_pitchers = pitchers.rename(columns={"season": "lag_season"})
    df["lag_season"] = df["season"] - 1
    p_merge = df[["lag_season", "mlbID"]].merge(
        lag_pitchers[["lag_season", "mlbID", "whiff_pct", "k_bb_pct", "fastball_velo"]],
        on=["lag_season", "mlbID"], how="left")
    league_avg = pitchers[pitchers["IP"] >= 20].groupby("season")[["whiff_pct", "k_bb_pct", "fastball_velo"]].mean()
    avg_lookup = league_avg.reindex(df["lag_season"]).reset_index(drop=True)
    for c, out_c in [("whiff_pct", "season_lag_whiff_pct"), ("k_bb_pct", "season_lag_k_bb_pct"),
                      ("fastball_velo", "season_lag_fastball_velo")]:
        missing = p_merge[c].isna()
        p_merge.loc[missing, c] = avg_lookup.loc[missing, c].values
        df[out_c] = p_merge[c].values

    # opposing lineup's K% specifically against pitchers who throw the same hand as this
    # starter (team_season's vs_lhp/vs_rhp split), season-lagged
    split = df["throws"].map({"L": "vs_lhp", "R": "vs_rhp"}).fillna("overall")
    team_key = pd.DataFrame({"season": df["lag_season"], "team_id": df["opponent_team_id"], "split": split})
    team_merge = team_key.merge(teams, on=["season", "team_id", "split"], how="left")
    team_league_avg = teams.groupby(["season", "split"])["k_pct"].mean()
    avg_lookup2 = team_league_avg.reindex(pd.MultiIndex.from_frame(team_key[["season", "split"]])).reset_index(drop=True)
    missing = team_merge["k_pct"].isna()
    team_merge.loc[missing, "k_pct"] = avg_lookup2[missing].values
    df["opp_off_kpct"] = team_merge["k_pct"].values

    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # naive baseline: trailing-3-start K rate x trailing IP/start, no other covariates
    df["naive_pred_k"] = df["trail_k_per9_3s"] * df["trail_ip_per_start"] / 9

    out = df[ID_COLS + [TARGET_COL, "naive_pred_k"] + FEATURE_COLS].reset_index(drop=True)
    out = out.dropna(subset=FEATURE_COLS)
    out.to_parquet(PROCESSED / "model_features_ks.parquet", index=False)
    return out


if __name__ == "__main__":
    df = build_features()
    print(f"model_features_ks: {len(df):,} starts, {len(FEATURE_COLS)} features")
    print(df.groupby("season").size())
    print(f"\nSO distribution: mean={df[TARGET_COL].mean():.2f} var={df[TARGET_COL].var():.2f} "
          f"(var/mean={df[TARGET_COL].var()/df[TARGET_COL].mean():.2f} -- >1 means overdispersed vs Poisson)")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.sample(min(5, len(df)), random_state=0).to_string())
