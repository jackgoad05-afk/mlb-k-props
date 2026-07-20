"""
Closing-line capture for the WNBA paper ledger. Same role as
capture_closing_ks.py: the free Odds API tier has no /historical/ access, so
instead of asking "what was the price at time T" after the fact, this
snapshots the CURRENT price ourselves, close to tipoff, while the game is
still upcoming -- writes into the ledger's closing_price column, so CLV at
reconciliation is just arithmetic on numbers already sitting there.

Simpler than the K-props version: moneyline/totals are both in the FREE bulk
endpoint (odds_api.get_bulk_odds), unlike player props, which need the paid-
quota per-event endpoint. One bulk call covers every flagged game's closing
price, no per-event cost.

Idempotent: any flagged row that already has a closing_price is skipped, so
re-running later just catches games that weren't in the capture window yet.

Usage:
    python src/capture_closing_wnba.py                  # today's flagged, not-yet-captured rows
    python src/capture_closing_wnba.py --date 2026-07-18 # override "today"
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import odds_api
from daily_wnba import LEDGER_PATH, _representative_price

CAPTURE_WINDOW_BEFORE = timedelta(hours=3)   # don't bother this far ahead of tipoff
CAPTURE_WINDOW_AFTER = timedelta(minutes=20)  # small grace period after commence_time


def run(target_date: date):
    if not LEDGER_PATH.exists():
        print(f"no ledger found at {LEDGER_PATH} -- nothing to capture yet.")
        return

    ledger = pd.read_csv(LEDGER_PATH)
    todays_flagged = ledger[(ledger["date"] == target_date.isoformat()) & ledger["flagged"]]
    pending = todays_flagged[todays_flagged["closing_price"].isna()]
    if pending.empty:
        print(f"no pending closing-line captures for {target_date.isoformat()} "
              f"(either nothing flagged today, or already captured).")
        return

    api_key = odds_api.load_api_key()
    bulk = odds_api.get_bulk_odds(api_key, markets="h2h,totals", sport="basketball_wnba")
    now = datetime.now(timezone.utc)

    due_games = {}  # (home_team_name, away_team_name) -> raw game odds
    n_too_early, n_too_late = 0, 0
    for g_odds in bulk:
        commence = datetime.fromisoformat(g_odds["commence_time"].replace("Z", "+00:00"))
        match = pending[(pending["home_team_name"] == g_odds["home_team"])
                         & (pending["away_team_name"] == g_odds["away_team"])]
        if match.empty:
            continue
        if now < commence - CAPTURE_WINDOW_BEFORE:
            n_too_early += len(match)
            continue
        if now > commence + CAPTURE_WINDOW_AFTER:
            n_too_late += len(match)
            continue
        due_games[(g_odds["home_team"], g_odds["away_team"])] = g_odds

    print(f"pending captures: {len(pending)}  |  due now: {len(due_games)} game(s)  |  "
          f"too early: {n_too_early}  |  too late (already underway): {n_too_late}")
    if not due_games:
        print("nothing in the capture window right now -- run again closer to tipoff.")
        return

    n_captured = 0
    for idx, row in pending.iterrows():
        g_odds = due_games.get((row["home_team_name"], row["away_team_name"]))
        if g_odds is None:
            continue

        if row["market_type"] == "moneyline":
            outcome_name = row["home_team_name"] if row["side"] == "home" else row["away_team_name"]
            price = _representative_price(g_odds, "h2h", outcome_name)
        else:
            outcome_name = "Over" if row["side"] == "over" else "Under"
            price = _representative_price(g_odds, "totals", outcome_name, row["line"])

        if price is None:
            continue
        ledger.loc[idx, "closing_price"] = price
        n_captured += 1

    ledger.to_csv(LEDGER_PATH, index=False)
    print(f"captured closing lines for {n_captured} of {len(pending)} pending row(s).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    args = ap.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    run(target)
