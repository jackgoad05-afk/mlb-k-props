"""
Reconciliation for the article-based K-props ledger (output/article_picks_ks_ledger.csv,
see daily_article_picks.py). Identical logic to reconcile_ks.py (same ledger
columns for everything reconciliation touches: mlbID, date, line, bet_side,
over_odds/under_odds, over_prob_fair, closing_over_odds/closing_under_odds),
just pointed at the article-picks ledger so its record/ROI/CLV can be compared
against the stats model and the research model on the dashboard.

    python src/reconcile_article_picks_ks.py reconcile          # fill in actual_so/result/pnl/clv
    python src/reconcile_article_picks_ks.py summary             # running totals
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

import fetch
from daily_article_picks import KS_LEDGER_PATH as LEDGER_PATH
from fetch_odds import american_to_decimal


def _payout(decimal_odds: float) -> float:
    return decimal_odds - 1.0


def _devig_pair(over_odds: float, under_odds: float) -> tuple[float, float]:
    over_raw = american_to_decimal(over_odds)
    under_raw = american_to_decimal(under_odds)
    over_p, under_p = 1 / over_raw, 1 / under_raw
    vig = over_p + under_p
    return over_p / vig, under_p / vig


def fetch_actual_scores(pending: pd.DataFrame) -> dict[tuple[int, str], float]:
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
          f"(run capture_closing_ks.py before this -- it captures for all three K-props ledgers)")


def summarize():
    if not LEDGER_PATH.exists():
        print(f"no ledger found at {LEDGER_PATH}.")
        return
    ledger = pd.read_csv(LEDGER_PATH)
    done = ledger[ledger["result"].notna()].copy()
    pending_n = ledger["result"].isna().sum()

    print("=== article-based K-props ledger ===")
    print(f"total picks: {len(ledger)}  |  reconciled: {len(done)}  |  pending: {pending_n}")
    if done.empty:
        print("no reconciled bets yet.")
        return

    n = len(done)
    w = (done["result"] == done["bet_side"]).sum()
    l_ = (done["pnl"] < 0).sum()
    p = (done["result"] == "push").sum()
    roi = done["pnl"].sum() / n
    clv = done["clv"].dropna()
    line = f"ALL          n={n:4d}  record={w}-{l_}-{p}  ROI={roi:+.1%}  units={done['pnl'].sum():+.2f}u  "
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

    # Aligned (articles + stats model agree) vs contrarian (they disagree) -- the
    # actual question this pipeline exists to answer.
    if "alignment" in done.columns:
        print("\nby alignment with stats model:")
        for level in ["aligned", "contrarian"]:
            tier = done[done["alignment"] == level]
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
