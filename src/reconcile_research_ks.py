"""
Reconciliation for the research-model K-props ledger (output/research_ks_ledger.csv).
Identical logic to reconcile_ks.py (same ledger schema, same reconciliation math),
just pointed at the research model's own ledger so its record/ROI/CLV can be
compared head-to-head against the stats-only model on the dashboard.

    python src/reconcile_research_ks.py reconcile          # fill in actual_so/result/pnl/clv
    python src/reconcile_research_ks.py summary             # running totals
    python src/reconcile_research_ks.py summary --by-tier   # broken out by edge-size tier
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

import fetch
from daily_research_ks import LEDGER_PATH
from fetch_odds import american_to_decimal

EDGE_TIER_BOUNDS = [(0.03, 0.05), (0.05, 0.10), (0.10, 1.01)]


def _payout(decimal_odds: float) -> float:
    return decimal_odds - 1.0


def _devig_pair(over_odds: float, under_odds: float) -> tuple[float, float]:
    over_raw = american_to_decimal(over_odds)
    under_raw = american_to_decimal(under_odds)
    over_p, under_p = 1 / over_raw, 1 / under_raw
    vig = over_p + under_p
    return over_p / vig, under_p / vig


def fetch_actual_scores(pending: pd.DataFrame) -> dict[tuple[int, str], float]:
    """One gameLog pull per (mlbID, season) actually needed, not per ledger row."""
    out: dict[tuple[int, str], float] = {}
    pending = pending.copy()
    pending["season"] = pd.to_datetime(pending["date"]).dt.year
    for (season, mlbID), _ in pending.groupby(["season", "mlbID"]):
        df = fetch.fetch_starter_game_logs(int(season), [int(mlbID)], refresh=True, sleep=0.0)
        for _, r in df[df["mlbID"] == mlbID].iterrows():
            out[(int(mlbID), r["official_date"])] = r["so"]
    return out


def reconcile():
    if not LEDGER_PATH.exists():
        print(f"no ledger found at {LEDGER_PATH} -- nothing to reconcile yet.")
        return

    ledger = pd.read_csv(LEDGER_PATH)
    ledger["result"] = ledger["result"].astype("object")
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

    actual_scores = fetch_actual_scores(playable)

    n_updated, n_no_result_yet, n_no_closing, n_has_closing = 0, 0, 0, 0
    for idx, row in playable.iterrows():
        actual_so = actual_scores.get((int(row["mlbID"]), row["date"]))
        if actual_so is None or pd.isna(actual_so):
            n_no_result_yet += 1
            continue

        line = row["line"]
        result = "over" if actual_so > line else ("under" if actual_so < line else "push")
        won = result == row["bet_side"]
        if result == "push":
            pnl = 0.0
        else:
            flagged_odds = row["over_odds"] if row["bet_side"] == "over" else row["under_odds"]
            pnl = _payout(american_to_decimal(flagged_odds)) if won else -1.0

        ledger.loc[idx, "actual_so"] = actual_so
        ledger.loc[idx, "result"] = result
        ledger.loc[idx, "pnl"] = pnl
        n_updated += 1

        closing_over, closing_under = row.get("closing_over_odds"), row.get("closing_under_odds")
        if pd.notna(closing_over) and pd.notna(closing_under):
            over_fair, under_fair = _devig_pair(closing_over, closing_under)
            closing_fair = over_fair if row["bet_side"] == "over" else under_fair
            flagged_fair = row["over_prob_fair"] if row["bet_side"] == "over" else 1 - row["over_prob_fair"]
            ledger.loc[idx, "clv"] = closing_fair - flagged_fair
            n_has_closing += 1
        else:
            n_no_closing += 1

    ledger.to_csv(LEDGER_PATH, index=False)
    print(f"reconciled {n_updated} row(s)  |  {n_too_soon} not yet playable  |  "
          f"{n_no_result_yet} playable but no result upstream yet (retry later)")
    print(f"CLV: {n_has_closing} row(s) had a captured closing price, {n_no_closing} did not "
          f"(run capture_closing_ks.py before this to fill those in -- it captures for both ledgers)")


def summarize(by_tier: bool = False):
    if not LEDGER_PATH.exists():
        print(f"no ledger found at {LEDGER_PATH}.")
        return
    ledger = pd.read_csv(LEDGER_PATH)
    done = ledger[ledger["result"].notna()].copy()
    pending_n = ledger["result"].isna().sum()

    print("=== research-model K-props ledger ===")
    print(f"total flagged: {len(ledger)}  |  reconciled: {len(done)}  |  pending: {pending_n}")
    if done.empty:
        print("no reconciled bets yet.")
        return

    def report(df: pd.DataFrame, label: str):
        n = len(df)
        w = (df["result"] == df["bet_side"]).sum()
        l_ = (df["pnl"] < 0).sum()
        p = (df["result"] == "push").sum()
        roi = df["pnl"].sum() / n if n else float("nan")
        clv = df["clv"].dropna()

        line = (f"{label:12s} n={n:4d}  record={w}-{l_}-{p}  ROI={roi:+.1%}  "
                f"units={df['pnl'].sum():+.2f}u  ")
        if len(clv):
            line += f"avg_CLV={clv.mean():+.2%} (n={len(clv)})  beat_close={(clv > 0).mean():.1%}"
        else:
            line += "avg_CLV=n/a (no closing lines yet)"
        print(line)

    report(done, "ALL")

    if by_tier:
        print("\nby edge tier:")
        for lo, hi in EDGE_TIER_BOUNDS:
            tier = done[(done["bet_edge"] >= lo) & (done["bet_edge"] < hi)]
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
    args = ap.parse_args()

    if args.command == "reconcile":
        reconcile()
    elif args.command == "summary":
        summarize(by_tier=args.by_tier)
