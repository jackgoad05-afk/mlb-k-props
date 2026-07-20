"""
Reconciliation for the WNBA paper ledger (output/wnba_paper_ledger.csv). Both
market types (moneyline, totals) share one ledger and one reconciliation pass.

`actual_result`/`correct` are filled in for EVERY logged row (not just flagged
ones) -- daily_wnba.py logs every game's model prediction unconditionally, so
straight-up accuracy is trackable across the full distribution, not just the
tail that happened to clear the 3% edge bar. `pnl` is only computed for
`flagged` rows, since that's the only column with a real "would have bet this"
meaning -- 1-unit stake at the logged `price`, same payout convention as
reconcile_ks.py.

    python src/reconcile_wnba.py reconcile              # fill in actual_result/correct/pnl
    python src/reconcile_wnba.py summary                 # overall record/ROI (flagged only)
    python src/reconcile_wnba.py summary --by-tier        # broken out by edge-size tier
    python src/reconcile_wnba.py summary --by-market-type  # moneyline vs. totals
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

import fetch_wnba
from daily_wnba import LEDGER_PATH
from fetch_odds import american_to_decimal

EDGE_TIER_BOUNDS = [(0.03, 0.05), (0.05, 0.10), (0.10, 1.01)]


def _payout(decimal_odds: float) -> float:
    return decimal_odds - 1.0


def fetch_actual_results(pending: pd.DataFrame) -> dict[str, dict]:
    """game_id -> {home_score, away_score}. One schedule pull per season
    actually needed (cached after the first, refreshed for the current one)."""
    out: dict[str, dict] = {}
    pending = pending.copy()
    pending["season"] = pd.to_datetime(pending["date"]).dt.year
    for season in pending["season"].unique():
        games = fetch_wnba.fetch_schedule(int(season), refresh=True)
        game_ids = set(pending.loc[pending["season"] == season, "game_id"].astype(str))
        for _, g in games[games["game_id"].astype(str).isin(game_ids)].iterrows():
            if pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
                continue
            out[str(g["game_id"])] = {"home_score": g["home_score"], "away_score": g["away_score"]}
    return out


def reconcile():
    if not LEDGER_PATH.exists():
        print(f"no ledger found at {LEDGER_PATH} -- nothing to reconcile yet.")
        return

    ledger = pd.read_csv(LEDGER_PATH)
    ledger["correct"] = ledger["correct"].astype("object")
    ledger["actual_result"] = ledger["actual_result"].astype("object")
    pending = ledger[ledger["actual_result"].isna()].copy()
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

    results = fetch_actual_results(playable)

    n_updated, n_no_result_yet, n_flagged_pnl = 0, 0, 0
    for idx, row in playable.iterrows():
        res = results.get(str(row["game_id"]))
        if res is None:
            n_no_result_yet += 1
            continue  # postponed/not final upstream yet -- retry next run

        if row["market_type"] == "moneyline":
            actual = "home" if res["home_score"] > res["away_score"] else "away"
        else:
            total = res["home_score"] + res["away_score"]
            actual = "push" if total == row["line"] else ("over" if total > row["line"] else "under")

        correct = actual == row["side"]
        ledger.loc[idx, "actual_result"] = actual
        ledger.loc[idx, "correct"] = bool(correct) if actual != "push" else pd.NA
        n_updated += 1

        if bool(row["flagged"]) and pd.notna(row.get("price")):
            if actual == "push":
                pnl = 0.0
            else:
                pnl = _payout(american_to_decimal(row["price"])) if correct else -1.0
            ledger.loc[idx, "pnl"] = pnl
            n_flagged_pnl += 1

    ledger.to_csv(LEDGER_PATH, index=False)
    print(f"reconciled {n_updated} row(s)  |  {n_too_soon} not yet playable  |  "
          f"{n_no_result_yet} playable but no result upstream yet (retry later)  |  "
          f"{n_flagged_pnl} flagged row(s) got a pnl")


def summarize(by_tier: bool = False, by_market_type: bool = False):
    if not LEDGER_PATH.exists():
        print(f"no ledger found at {LEDGER_PATH}.")
        return
    ledger = pd.read_csv(LEDGER_PATH)
    flagged = ledger[ledger["flagged"] == True].copy()  # noqa: E712 (CSV round-trip bool)
    done = flagged[flagged["pnl"].notna()].copy()
    pending_n = len(flagged) - len(done)

    print("=== WNBA paper ledger (flagged bets only) ===")
    print(f"total flagged: {len(flagged)}  |  reconciled: {len(done)}  |  pending: {pending_n}")
    if done.empty:
        print("no reconciled flagged bets yet.")
        return

    def report(df: pd.DataFrame, label: str):
        n = len(df)
        if n == 0:
            print(f"{label:20s} n=   0")
            return
        w = int((df["correct"] == True).sum())  # noqa: E712
        l_ = int((df["pnl"] < 0).sum())
        p = n - w - l_
        roi = df["pnl"].sum() / n
        print(f"{label:20s} n={n:4d}  record={w}-{l_}-{p}  ROI={roi:+.1%}  units={df['pnl'].sum():+.2f}u")

    report(done, "ALL")

    if by_market_type:
        print("\nby market type:")
        for mt in sorted(done["market_type"].unique()):
            report(done[done["market_type"] == mt], mt)

    if by_tier:
        print("\nby edge tier:")
        for lo, hi in EDGE_TIER_BOUNDS:
            tier = done[(done["edge"] >= lo) & (done["edge"] < hi)]
            if tier.empty:
                continue
            label = f"{lo:.0%}-{hi:.0%}" if hi < 1 else f"{lo:.0%}+"
            report(tier, label)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("reconcile")
    summary_p = sub.add_parser("summary")
    summary_p.add_argument("--by-tier", action="store_true")
    summary_p.add_argument("--by-market-type", action="store_true")
    args = ap.parse_args()

    if args.command == "reconcile":
        reconcile()
    elif args.command == "summary":
        summarize(by_tier=args.by_tier, by_market_type=args.by_market_type)
