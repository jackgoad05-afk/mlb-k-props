"""
Track 1: re-run the moneyline backtest against REAL closing odds instead of the log5
proxy market. Closes the "is proxy-market ROI/CLV meaningful" question for good.

Methodology (mirrors backtest.py's proxy design, now with real numbers):
  - Edge is computed against the CLOSING consensus fair probability (avg no-vig
    probability across up to 6 books) -- the sharpest, most-informed line available.
  - ROI is simulated by betting AT THE CLOSING PRICE (median American odds across
    books), flat 1u stakes. This is the standard, conservative backtest convention:
    it assumes no ability to beat the market's own final number, so any positive
    ROI here is a real edge over a genuinely efficient closing line, not an
    artifact of vig assumptions or a crude baseline.
  - CLV = closing fair probability minus OPENING fair probability, in the
    direction of the bet. Positive CLV means the market moved toward the model's
    side between open and close -- the standard real-money signal that an edge is
    real even in samples too small for ROI alone to prove it.

Only games with real odds coverage are included (see fetch_odds.py for coverage;
2025 cuts off 2025-08-16, so this is a subset of the full 2025 holdout).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import drawdown_stats
from features import PROCESSED, TARGET_COL
from model_ml import OUTPUT

EDGE_FLAG_THRESHOLD = 0.03
EDGE_TIER_BOUNDS = [(0.01, 0.02), (0.02, 0.04), (0.04, 1.01)]
MIN_BOOKS = 2


def brier_score(probs: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean((probs - actuals) ** 2))


def run():
    preds = pd.read_parquet(OUTPUT / "ml_holdout_predictions.parquet")
    odds = pd.read_parquet(PROCESSED / "real_odds.parquet")
    odds = odds[odds["is_consensus"] & (odds["n_books"] >= MIN_BOOKS)].copy()

    df = preds.merge(odds, on="game_pk", how="inner", suffixes=("", "_odds"))
    bad_odds = (df["home_close_decimal"] <= 1.0) | (df["away_close_decimal"] <= 1.0)
    if bad_odds.any():
        print(f"[warn] dropping {bad_odds.sum()} games with an invalid closing decimal price")
        df = df[~bad_odds].copy()
    print(f"real-odds holdout coverage: {len(df):,} / {len(preds):,} games "
          f"({len(df)/len(preds):.1%}) -- odds data ends 2025-08-16, so late-season games are excluded")

    df["home_edge"] = df["model_prob"] - df["home_close_fair_prob"]
    df["away_edge"] = (1 - df["model_prob"]) - df["away_close_fair_prob"]
    df["bet_side"] = np.where(df["home_edge"] >= df["away_edge"], "home", "away")
    df["edge"] = np.where(df["bet_side"] == "home", df["home_edge"], df["away_edge"])
    df["bet_won"] = np.where(df["bet_side"] == "home", df[TARGET_COL] == 1, df[TARGET_COL] == 0)

    df["bet_decimal"] = np.where(df["bet_side"] == "home", df["home_close_decimal"], df["away_close_decimal"])
    df["payout_win"] = df["bet_decimal"] - 1.0
    df["pnl"] = np.where(df["bet_won"], df["payout_win"], -1.0)

    has_open = df["home_open_fair_prob"].notna()
    df["clv"] = np.nan
    df.loc[has_open, "clv"] = np.where(
        df.loc[has_open, "bet_side"] == "home",
        df.loc[has_open, "home_close_fair_prob"] - df.loc[has_open, "home_open_fair_prob"],
        df.loc[has_open, "away_close_fair_prob"] - df.loc[has_open, "away_open_fair_prob"],
    )

    bs_model = brier_score(df["model_prob"].values, df[TARGET_COL].values)
    bs_market = brier_score(df["home_close_fair_prob"].values, df[TARGET_COL].values)

    print("\n" + "=" * 78)
    print("MONEYLINE BACKTEST -- REAL CLOSING ODDS (not proxy)")
    print("=" * 78)
    print(f"\nBrier score (lower is better, 0.25 = coin flip):")
    print(f"  model, calibrated     : {bs_model:.4f}")
    print(f"  real closing market   : {bs_market:.4f}")
    print(f"  (Session 2.1 proxy backtest reported model 0.2461 vs. log5 proxy 0.2474 --")
    print(f"   compare the market number above to that proxy to see how much sharper a real book is)")

    print(f"\nPerformance by edge size tier (vs REAL closing line, real decimal payout, 1u stakes):")
    tier_rows = []
    for lo, hi in EDGE_TIER_BOUNDS:
        tier = df[(df["edge"] >= lo) & (df["edge"] < hi)]
        if len(tier) == 0:
            continue
        roi = tier["pnl"].sum() / len(tier)
        clv = tier["clv"].mean()
        tier_rows.append({"edge_tier": f"{lo:.0%}-{hi:.0%}" if hi < 1 else f"{lo:.0%}+",
                           "n_bets": len(tier), "win_rate": tier["bet_won"].mean(),
                           "roi_real": roi, "avg_clv_real": clv})
    tier_df = pd.DataFrame(tier_rows)
    print(tier_df.to_string(index=False))

    flagged = df[df["edge"] >= EDGE_FLAG_THRESHOLD].sort_values("official_date").copy()
    n = len(flagged)
    print(f"\nFlagged bets (edge >= {EDGE_FLAG_THRESHOLD:.0%}, real closing decimal odds, 1u stakes):")
    if n:
        win_rate = flagged["bet_won"].mean()
        roi = flagged["pnl"].sum() / n
        avg_clv = flagged["clv"].mean()
        clv_pos_rate = (flagged["clv"] > 0).mean()
        max_dd, longest_streak, equity = drawdown_stats(flagged["pnl"].reset_index(drop=True))

        print(f"  n flagged        : {n}  ({n/len(df):.1%} of {len(df)} games with real odds)")
        print(f"  win rate         : {win_rate:.1%}")
        print(f"  ROI (real)       : {roi:+.1%}")
        print(f"  units won/lost   : {flagged['pnl'].sum():+.2f}u")
        print(f"  CLV, avg (real)  : {avg_clv:+.2%}  |  positive-CLV rate: {clv_pos_rate:.1%}")
        print(f"  max drawdown     : {max_dd:.2f}u")
        print(f"  longest losing streak: {int(longest_streak)} bets")

        print("\n" + "-" * 78)
        if n >= 500 and (roi < 0 or avg_clv < 0):
            print("KILL CRITERIA MET on REAL odds: this model does not bet, full stop.")
        elif n < 500:
            print(f"Only {n} flagged bets vs. real odds -- below the spec's 500-bet threshold,")
            print("  directional but not decisive on its own.")
        else:
            print("Kill criteria NOT triggered on real odds.")
        print("-" * 78)
    else:
        print("  no games cleared the edge threshold vs. real closing lines.")

    df.to_csv(OUTPUT / "ml_real_odds_bets.csv", index=False)
    tier_df.to_csv(OUTPUT / "ml_real_odds_tiers.csv", index=False)
    return {"df": df, "tiers": tier_df, "flagged": flagged, "brier_model": bs_model, "brier_market": bs_market}


if __name__ == "__main__":
    run()
