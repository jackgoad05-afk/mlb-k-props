"""
Reconciliation for the pure-prediction moneyline ledger (output/ml_predictions_ledger.csv).
Straight-up accuracy only -- no CLV/pnl, since this module never bets (see daily_ml.py).

Run after games finish:

    python src/reconcile_ml.py reconcile           # fill in actual_winner/correct
    python src/reconcile_ml.py summary              # overall record
    python src/reconcile_ml.py summary --vs-market  # ALSO break out by whether the model
                                                      # agreed with the market favorite --
                                                      # terminal-only, this never touches
                                                      # the dashboard (see daily_ml.py's
                                                      # module docstring on why).
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

import fetch
from daily_ml import MARKET_COMPARISON_PATH, PREDICTIONS_LEDGER_PATH


def fetch_actual_winners(pending: pd.DataFrame) -> dict[int, str]:
    """One schedule pull per season actually needed (cached after the first), not per row."""
    out: dict[int, str] = {}
    pending = pending.copy()
    pending["season"] = pd.to_datetime(pending["date"]).dt.year
    for season in pending["season"].unique():
        games = fetch.fetch_schedule(int(season), refresh=True)
        game_pks = set(pending.loc[pending["season"] == season, "game_pk"].astype(int))
        for _, g in games[games["game_pk"].isin(game_pks)].iterrows():
            if pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
                continue
            winner = g["home_team_name"] if g["home_score"] > g["away_score"] else g["away_team_name"]
            out[int(g["game_pk"])] = winner
    return out


def reconcile():
    if not PREDICTIONS_LEDGER_PATH.exists():
        print(f"no ledger found at {PREDICTIONS_LEDGER_PATH} -- nothing to reconcile yet.")
        return

    ledger = pd.read_csv(PREDICTIONS_LEDGER_PATH)
    ledger["correct"] = ledger["correct"].astype("object")
    pending = ledger[ledger["actual_winner"].isna()].copy()
    if pending.empty:
        print("nothing pending -- every ledger row already has a result.")
        return

    today = datetime.now().date()
    pending["game_date"] = pd.to_datetime(pending["date"]).dt.date
    playable = pending[pending["game_date"] < today]
    n_too_soon = len(pending) - len(playable)
    if playable.empty:
        print(f"{n_too_soon} pending row(s), none playable yet (game date >= today).")
        return

    winners = fetch_actual_winners(playable)

    n_updated, n_no_result_yet = 0, 0
    for idx, row in playable.iterrows():
        winner = winners.get(int(row["game_pk"]))
        if winner is None:
            n_no_result_yet += 1
            continue  # postponed/suspended/not final upstream yet -- retry next run

        ledger.loc[idx, "actual_winner"] = winner
        ledger.loc[idx, "correct"] = bool(winner == row["predicted_winner"])
        n_updated += 1

    ledger.to_csv(PREDICTIONS_LEDGER_PATH, index=False)
    print(f"reconciled {n_updated} row(s)  |  {n_too_soon} not yet playable  |  "
          f"{n_no_result_yet} playable but no result upstream yet (retry later)")


def summarize(vs_market: bool = False):
    if not PREDICTIONS_LEDGER_PATH.exists():
        print(f"no ledger found at {PREDICTIONS_LEDGER_PATH}.")
        return
    ledger = pd.read_csv(PREDICTIONS_LEDGER_PATH)
    done = ledger[ledger["correct"].notna()].copy()
    done["correct"] = done["correct"].astype(bool)
    pending_n = ledger["correct"].isna().sum()

    print("=== moneyline predictions leaderboard (straight-up) ===")
    print(f"total predictions: {len(ledger)}  |  reconciled: {len(done)}  |  pending: {pending_n}")
    if done.empty:
        print("no reconciled predictions yet.")
        return

    def report(df: pd.DataFrame, label: str):
        n = len(df)
        w = df["correct"].sum()
        acc = w / n if n else float("nan")
        print(f"{label:20s} n={n:4d}  record={w}-{n - w}  accuracy={acc:.1%}")

    report(done, "ALL")

    if not vs_market:
        return
    if not MARKET_COMPARISON_PATH.exists():
        print("\nno market-comparison file yet -- run daily_ml.py (live, not --dry-run) first.")
        return

    market = pd.read_csv(MARKET_COMPARISON_PATH)[["date", "game_pk", "model_agrees_with_favorite"]]
    merged = done.merge(market, on=["date", "game_pk"], how="inner")
    if merged.empty:
        print("\nno overlap yet between reconciled predictions and logged market-favorite rows.")
        return

    print("\nvs. market favorite (private -- not shown on dashboard):")
    report(merged[merged["model_agrees_with_favorite"]], "agreed w/ favorite")
    report(merged[~merged["model_agrees_with_favorite"]], "picked underdog")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("reconcile")
    summary_p = sub.add_parser("summary")
    summary_p.add_argument("--vs-market", action="store_true")
    args = ap.parse_args()

    if args.command == "reconcile":
        reconcile()
    elif args.command == "summary":
        summarize(vs_market=args.vs_market)
