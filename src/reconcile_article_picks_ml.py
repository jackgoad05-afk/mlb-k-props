"""
Reconciliation for the article-based moneyline ledger (output/article_picks_ml_ledger.csv,
see daily_article_picks.py). Unlike the stats-only moneyline model (daily_ml.py,
deliberately no betting logic -- see its docstring), this ledger tracks real
PnL and CLV, same 1-unit-stake convention as the K-props ledgers, since these
picks are article-driven judgment calls meant to be compared head-to-head
against the stats-only and research-model K-props ledgers on the dashboard.

Winner-determination logic (fetch.fetch_schedule keyed by game_pk, home_score
vs. away_score) is the same pattern reconcile_pm.py already uses successfully
for prediction-market moneyline reconciliation.

    python src/reconcile_article_picks_ml.py reconcile   # fill in actual_winner/result/pnl/clv
    python src/reconcile_article_picks_ml.py summary      # running totals
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

import fetch
from daily_article_picks import ML_LEDGER_PATH as LEDGER_PATH
from fetch_odds import american_to_decimal


def _payout(decimal_odds: float) -> float:
    return decimal_odds - 1.0


def _devig_pair(home_odds: float, away_odds: float) -> tuple[float, float]:
    home_raw = american_to_decimal(home_odds)
    away_raw = american_to_decimal(away_odds)
    home_p, away_p = 1 / home_raw, 1 / away_raw
    vig = home_p + away_p
    return home_p / vig, away_p / vig


def fetch_actual_winners(pending: pd.DataFrame) -> dict[int, str]:
    """One schedule pull per season actually needed (cached after the first)."""
    out: dict[int, str] = {}
    pending = pending.copy()
    pending["season"] = pd.to_datetime(pending["date"]).dt.year
    for season in pending["season"].unique():
        games = fetch.fetch_schedule(int(season), refresh=True)
        game_pks = set(pending.loc[pending["season"] == season, "game_pk"].astype(int))
        for _, g in games[games["game_pk"].isin(game_pks)].iterrows():
            if pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
                continue
            winner = "home" if g["home_score"] > g["away_score"] else "away"
            out[int(g["game_pk"])] = winner
    return out


def reconcile():
    if not LEDGER_PATH.exists():
        print(f"no ledger found at {LEDGER_PATH} -- nothing to reconcile yet.")
        return

    ledger = pd.read_csv(LEDGER_PATH)
    ledger["result"] = ledger["result"].astype("object")
    ledger["actual_winner"] = ledger["actual_winner"].astype("object")
    pending = ledger[ledger["result"].isna()].copy()
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

    n_updated, n_no_result_yet, n_no_closing, n_has_closing = 0, 0, 0, 0
    for idx, row in playable.iterrows():
        winner = winners.get(int(row["game_pk"]))
        if winner is None:
            n_no_result_yet += 1
            continue

        won = winner == row["bet_side"]
        picked_odds = row["home_odds"] if row["bet_side"] == "home" else row["away_odds"]
        pnl = _payout(american_to_decimal(picked_odds)) if won else -1.0

        ledger.loc[idx, "actual_winner"] = winner
        ledger.loc[idx, "result"] = winner
        ledger.loc[idx, "pnl"] = pnl
        n_updated += 1

        closing_home, closing_away = row.get("closing_home_odds"), row.get("closing_away_odds")
        if pd.notna(closing_home) and pd.notna(closing_away) and pd.notna(row.get("home_odds")) and pd.notna(row.get("away_odds")):
            closing_home_fair, closing_away_fair = _devig_pair(closing_home, closing_away)
            picked_home_fair, picked_away_fair = _devig_pair(row["home_odds"], row["away_odds"])
            closing_fair = closing_home_fair if row["bet_side"] == "home" else closing_away_fair
            picked_fair = picked_home_fair if row["bet_side"] == "home" else picked_away_fair
            ledger.loc[idx, "clv"] = closing_fair - picked_fair
            n_has_closing += 1
        else:
            n_no_closing += 1

    ledger.to_csv(LEDGER_PATH, index=False)
    print(f"reconciled {n_updated} row(s)  |  {n_too_soon} not yet playable  |  "
          f"{n_no_result_yet} playable but no result upstream yet (retry later)")
    print(f"CLV: {n_has_closing} row(s) had a captured closing price, {n_no_closing} did not")


def summarize():
    if not LEDGER_PATH.exists():
        print(f"no ledger found at {LEDGER_PATH}.")
        return
    ledger = pd.read_csv(LEDGER_PATH)
    done = ledger[ledger["result"].notna()].copy()
    pending_n = ledger["result"].isna().sum()

    print("=== article-based moneyline ledger ===")
    print(f"total picks: {len(ledger)}  |  reconciled: {len(done)}  |  pending: {pending_n}")
    if done.empty:
        print("no reconciled picks yet.")
        return

    n = len(done)
    w = (done["result"] == done["bet_side"]).sum()
    roi = done["pnl"].sum() / n
    clv = done["clv"].dropna()
    line = f"ALL          n={n:4d}  record={w}-{n - w}  ROI={roi:+.1%}  units={done['pnl'].sum():+.2f}u  "
    if len(clv):
        line += f"avg_CLV={clv.mean():+.2%} (n={len(clv)})  beat_close={(clv > 0).mean():.1%}"
    else:
        line += "avg_CLV=n/a (no closing lines yet)"
    print(line)

    if "confidence" in done.columns:
        print("\nby confidence:")
        for level in ["high", "medium", "low"]:
            tier = done[done["confidence"] == level]
            if tier.empty:
                continue
            n = len(tier)
            w = (tier["result"] == tier["bet_side"]).sum()
            roi = tier["pnl"].sum() / n
            print(f"{level:12s} n={n:4d}  record={w}-{n - w}  ROI={roi:+.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("reconcile")
    sub.add_parser("summary")
    args = ap.parse_args()

    if args.command == "reconcile":
        reconcile()
    elif args.command == "summary":
        summarize()
