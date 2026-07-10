"""
Session 2 backtest: full 2025 holdout report for the moneyline model.

IMPORTANT -- read before trusting any dollar figure below:
src/fetch.py does not pull historical sportsbook odds (The Odds API's historical
endpoint is paid; Kaggle archives need a manual download -- see CLAUDE.md). So:
  - Brier score and the calibration curve are REAL: they only need model probabilities
    and actual outcomes, no market data.
  - ROI and CLV are computed against a PROXY MARKET (a log5 win-rate baseline with
    empirical home-field advantage, NOT real closing lines) and are labeled "(proxy)"
    everywhere. They answer "does the model beat a naive baseline," not "would this
    have shown a profit against a real book." Don't quote them as real backtested ROI.

Proxy market construction:
  opening_proxy = log5(prior-season win%, prior-season win%) + HFA
    -- a preseason-strength view, no in-season information at all.
  closing_proxy = log5(in-season win% as of the day before, shrunk toward the prior-
    season win% early in the year) + HFA
    -- closer to what a market has priced in once real results exist.
  CLV proxy = closing_proxy movement toward the side the model liked, between opening
    and closing. That's the actual mechanism real CLV measures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import PROCESSED, TARGET_COL
from model_ml import run as run_model, OUTPUT, HOLDOUT_SEASON, TRAIN_SEASONS

VIG = 0.0455                     # standard -110/-110 total vig (~104.55% implied prob)
EDGE_FLAG_THRESHOLD = 0.03       # spec: surface edges >= 3%
EDGE_TIER_BOUNDS = [(0.01, 0.02), (0.02, 0.04), (0.04, 1.01)]
PAYOUT_WIN = 100 / 110           # flat 1-unit stake at -110, per spec
SHRINKAGE_K = 15                 # in-season record shrunk toward prior-season prior w/ this many "prior games"


# --------------------------------------------------------------------------- #
# Real metrics: Brier score + calibration
# --------------------------------------------------------------------------- #

def brier_score(probs: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean((probs - actuals) ** 2))


def calibration_table(probs: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"prob": probs, "actual": actuals})
    df["bin"] = pd.qcut(df["prob"], n_bins, duplicates="drop")
    table = df.groupby("bin", observed=True).agg(
        n=("actual", "size"), predicted=("prob", "mean"), actual=("actual", "mean")
    )
    table["gap_pp"] = (table["actual"] - table["predicted"]) * 100
    return table.reset_index()


# --------------------------------------------------------------------------- #
# Proxy market
# --------------------------------------------------------------------------- #

def team_season_win_pct(games: pd.DataFrame) -> pd.Series:
    home = games[["home_team_id", "home_win"]].rename(columns={"home_team_id": "team_id", "home_win": "win"})
    away = games[["away_team_id", "home_win"]].rename(columns={"away_team_id": "team_id"})
    away["win"] = 1 - games["home_win"].values
    both = pd.concat([home, away], ignore_index=True)
    return both.groupby("team_id")["win"].mean()


def log5(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    return (p_a - p_a * p_b) / (p_a + p_b - 2 * p_a * p_b)


def empirical_hfa_odds_ratio(train_games: pd.DataFrame) -> float:
    home_wp = train_games["home_win"].mean()
    return home_wp / (1 - home_wp)


def apply_hfa(p_home_no_hfa: np.ndarray, odds_ratio_hfa: float) -> np.ndarray:
    odds = p_home_no_hfa / (1 - p_home_no_hfa)
    odds_adj = odds * odds_ratio_hfa
    return odds_adj / (1 + odds_adj)


def expanding_shrunk_wp(season_games: pd.DataFrame, prior_wp: pd.Series, k: float = SHRINKAGE_K) -> pd.DataFrame:
    """As-of-date-before in-season win%, Bayesian-shrunk toward each team's prior-season
    win% (so a 3-1 season-opening week doesn't swing the proxy line hard)."""
    g = season_games.sort_values(["official_date", "game_pk"]).reset_index(drop=True)
    wins: dict[int, float] = {}
    n_games: dict[int, int] = {}
    home_input = np.empty(len(g))
    away_input = np.empty(len(g))

    for date, day_idx in g.groupby("official_date", sort=True).groups.items():
        day_idx = list(day_idx)
        snapshot: dict[int, float] = {}
        for idx in day_idx:
            row = g.loc[idx]
            for tid in (row["home_team_id"], row["away_team_id"]):
                if tid not in snapshot:
                    w = wins.get(tid, 0.0)
                    n = n_games.get(tid, 0)
                    prior = prior_wp.get(tid, 0.5)
                    snapshot[tid] = (w + k * prior) / (n + k)
        for idx in day_idx:
            row = g.loc[idx]
            home_input[idx] = snapshot[row["home_team_id"]]
            away_input[idx] = snapshot[row["away_team_id"]]
        for idx in day_idx:
            row = g.loc[idx]
            wins[row["home_team_id"]] = wins.get(row["home_team_id"], 0.0) + row["home_win"]
            n_games[row["home_team_id"]] = n_games.get(row["home_team_id"], 0) + 1
            wins[row["away_team_id"]] = wins.get(row["away_team_id"], 0.0) + (1 - row["home_win"])
            n_games[row["away_team_id"]] = n_games.get(row["away_team_id"], 0) + 1

    g["home_wp_input"] = home_input
    g["away_wp_input"] = away_input
    return g


def build_proxy_market(games_holdout: pd.DataFrame, games_prior_season: pd.DataFrame, hfa_or: float) -> pd.DataFrame:
    prior_wp = team_season_win_pct(games_prior_season)
    g = expanding_shrunk_wp(games_holdout, prior_wp)

    p_home_open_raw = log5(
        g["home_team_id"].map(prior_wp).fillna(0.5).values,
        g["away_team_id"].map(prior_wp).fillna(0.5).values,
    )
    g["p_home_open"] = apply_hfa(p_home_open_raw, hfa_or)

    p_home_close_raw = log5(g["home_wp_input"].values, g["away_wp_input"].values)
    g["p_home_close"] = apply_hfa(p_home_close_raw, hfa_or)

    return g


def add_vig(p_home: np.ndarray, total_vig: float = VIG) -> tuple[np.ndarray, np.ndarray]:
    scale = 1 + total_vig
    return p_home * scale, (1 - p_home) * scale


# --------------------------------------------------------------------------- #
# Bet simulation (proxy)
# --------------------------------------------------------------------------- #

def simulate_bets(preds: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    df = preds.merge(
        proxy[["game_pk", "p_home_open", "p_home_close"]], on="game_pk", how="inner"
    )
    home_implied, away_implied = add_vig(df["p_home_close"].values)
    df["home_implied_vig"] = home_implied
    df["away_implied_vig"] = away_implied

    df["home_edge"] = df["model_prob"] - df["home_implied_vig"]
    df["away_edge"] = (1 - df["model_prob"]) - df["away_implied_vig"]

    df["bet_side"] = np.where(df["home_edge"] >= df["away_edge"], "home", "away")
    df["edge"] = np.where(df["bet_side"] == "home", df["home_edge"], df["away_edge"])
    df["bet_won"] = np.where(df["bet_side"] == "home", df[TARGET_COL] == 1, df[TARGET_COL] == 0)
    df["pnl"] = np.where(df["bet_won"], PAYOUT_WIN, -1.0)

    df["clv_proxy"] = np.where(
        df["bet_side"] == "home",
        df["p_home_close"] - df["p_home_open"],
        (1 - df["p_home_close"]) - (1 - df["p_home_open"]),
    )
    return df


def drawdown_stats(pnl_chrono: pd.Series) -> tuple[float, int, pd.Series]:
    equity = pnl_chrono.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    is_loss = pnl_chrono < 0
    if is_loss.any():
        streak_id = (is_loss != is_loss.shift()).cumsum()
        longest = int(is_loss.groupby(streak_id).sum().max())
    else:
        longest = 0
    return max_dd, longest, equity


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def run():
    preds, _model, _cal = run_model()

    all_games = pd.read_parquet(PROCESSED / "games.parquet")
    train_games = all_games[all_games["season"].isin(TRAIN_SEASONS)]
    holdout_games = all_games[all_games["season"] == HOLDOUT_SEASON]
    prior_season_games = all_games[all_games["season"] == HOLDOUT_SEASON - 1]

    hfa_or = empirical_hfa_odds_ratio(train_games)
    proxy = build_proxy_market(holdout_games, prior_season_games, hfa_or)

    bets = simulate_bets(preds, proxy)

    print("=" * 78)
    print(f"MONEYLINE BACKTEST -- {HOLDOUT_SEASON} holdout (n={len(preds):,} games)")
    print("=" * 78)

    # --- Brier + calibration (real) ---
    bs_model = brier_score(preds["model_prob"].values, preds[TARGET_COL].values)
    bs_raw = brier_score(preds["raw_prob"].values, preds[TARGET_COL].values)
    bs_proxy = brier_score(bets["p_home_close"].values, bets[TARGET_COL].values)
    print(f"\nBrier score (lower is better, 0.25 = coin flip):")
    print(f"  model, calibrated : {bs_model:.4f}")
    print(f"  model, raw (uncal): {bs_raw:.4f}")
    print(f"  proxy market      : {bs_proxy:.4f}")

    print("\nCalibration (10 bins, predicted vs actual home-win rate):")
    cal = calibration_table(preds["model_prob"].values, preds[TARGET_COL].values)
    print(cal.to_string(index=False))

    # --- Edge tiers (real edge vs proxy, all games with edge >= 1%) ---
    print(f"\nPerformance by edge size tier (vs proxy market, -110 flat, {VIG:.2%} vig):")
    tier_rows = []
    for lo, hi in EDGE_TIER_BOUNDS:
        tier = bets[(bets["edge"] >= lo) & (bets["edge"] < hi)]
        if len(tier) == 0:
            continue
        roi = tier["pnl"].sum() / len(tier)
        tier_rows.append({
            "edge_tier": f"{lo:.0%}-{hi:.0%}" if hi < 1 else f"{lo:.0%}+",
            "n_bets": len(tier), "win_rate": tier["bet_won"].mean(),
            "roi_proxy": roi, "avg_clv_proxy": tier["clv_proxy"].mean(),
        })
    tier_df = pd.DataFrame(tier_rows)
    print(tier_df.to_string(index=False))

    # --- Flagged bets (>= 3% edge, the spec's live threshold) ---
    flagged = bets[bets["edge"] >= EDGE_FLAG_THRESHOLD].copy()
    flagged = flagged.sort_values("official_date")
    n_flagged = len(flagged)
    win_rate = flagged["bet_won"].mean() if n_flagged else float("nan")
    roi = flagged["pnl"].sum() / n_flagged if n_flagged else float("nan")
    avg_clv = flagged["clv_proxy"].mean() if n_flagged else float("nan")
    clv_positive_rate = (flagged["clv_proxy"] > 0).mean() if n_flagged else float("nan")

    print(f"\nFlagged bets (edge >= {EDGE_FLAG_THRESHOLD:.0%}, -110 flat, 1u stakes):")
    print(f"  n flagged        : {n_flagged}  ({n_flagged/len(bets):.1%} of all {len(bets)} holdout games)")
    print(f"  win rate         : {win_rate:.1%}")
    print(f"  ROI (proxy)      : {roi:+.1%}")
    print(f"  units won/lost   : {flagged['pnl'].sum():+.2f}u")
    print(f"  CLV proxy, avg   : {avg_clv:+.2%}  |  positive-CLV rate: {clv_positive_rate:.1%}")

    max_dd, longest_streak, equity = drawdown_stats(flagged["pnl"].reset_index(drop=True))
    print(f"  max drawdown     : {max_dd:.2f}u")
    print(f"  longest losing streak: {longest_streak} bets")

    print("\n" + "-" * 78)
    if n_flagged >= 500 and (roi < 0 or avg_clv < 0):
        print("KILL CRITERIA MET (on proxy market): ROI or CLV negative on 500+ flagged bets.")
        print("  -> Model does not bet. Treat as a dashboard product until real odds validate it.")
    elif n_flagged < 500:
        print(f"Only {n_flagged} flagged bets on this holdout -- below the spec's 500-bet threshold")
        print("  for kill-criteria evaluation. Numbers above are directional, not yet decisive.")
    else:
        print("Kill criteria not triggered on the proxy market (ROI and CLV both positive).")
    print("  Remember: this is all vs. a log5 PROXY market, not real closing lines.")
    print("-" * 78)

    bets.to_csv(OUTPUT / "ml_backtest_bets.csv", index=False)
    cal.to_csv(OUTPUT / "ml_calibration.csv", index=False)
    tier_df.to_csv(OUTPUT / "ml_edge_tiers.csv", index=False)

    return {
        "preds": preds, "bets": bets, "flagged": flagged, "calibration": cal,
        "edge_tiers": tier_df, "brier_model": bs_model, "brier_proxy": bs_proxy,
        "equity": equity, "max_dd": max_dd, "longest_streak": longest_streak,
        "hfa_odds_ratio": hfa_or,
    }


if __name__ == "__main__":
    run()
