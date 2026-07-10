"""
Closing-line capture for the K-props ledger. Free-tier substitute for The Odds API's
paid /historical/ endpoint: instead of asking the API "what was the price at time T"
after the fact, snapshot the CURRENT price ourselves, close to first pitch, while the
game is still upcoming. Writes straight into the ledger's closing_over_odds/
closing_under_odds columns -- by the time reconcile_ks.py runs overnight, CLV is just
arithmetic on numbers already sitting in the ledger, no further API calls needed.

Idempotent and safe to run more than once a day: any ledger row that already has a
closing price is skipped, so re-running later just catches games that weren't in the
capture window yet. Session 5's default schedule runs this once, ~6pm ET -- late
enough that most evening games haven't started, early enough to be genuinely close to
first pitch for the bulk of the slate. Day games that already finished by then are
reported as missed, not silently dropped: actual_so/result/pnl still fill in fine at
reconciliation (those only need the free MLB Stats API), just without a CLV number for
that specific bet. Run this script again earlier in the day (e.g. ~1pm ET) if you want
day-game coverage too -- it costs nothing extra, since already-captured/not-yet-
flagged rows are skipped either way.

Usage:
    python src/capture_closing_ks.py                  # today's flagged, not-yet-captured rows
    python src/capture_closing_ks.py --date 2026-07-10 # override "today"
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import odds_api
from daily_ks import LEDGER_PATH

CAPTURE_WINDOW_BEFORE = timedelta(hours=4)   # don't bother this far ahead of first pitch
CAPTURE_WINDOW_AFTER = timedelta(minutes=20)  # small grace period after commence_time


def run(target_date: date):
    if not LEDGER_PATH.exists():
        print(f"no ledger found at {LEDGER_PATH} -- nothing to capture yet.")
        return

    ledger = pd.read_csv(LEDGER_PATH)
    todays = ledger[ledger["date"] == target_date.isoformat()]
    pending = todays[todays["closing_over_odds"].isna()]
    if pending.empty:
        print(f"no pending closing-line captures for {target_date.isoformat()} "
              f"(either nothing flagged today, or already captured).")
        return

    api_key = odds_api.load_api_key()
    events = {ev["id"]: ev for ev in odds_api.get_events(api_key)}
    now = datetime.now(timezone.utc)

    due_event_ids = set()
    n_no_event, n_too_early, n_too_late = 0, 0, 0
    for event_id, group in pending.groupby("event_id"):
        if pd.isna(event_id) or str(event_id) not in events:
            n_no_event += len(group)
            continue
        commence = datetime.fromisoformat(events[str(event_id)]["commence_time"].replace("Z", "+00:00"))
        if now < commence - CAPTURE_WINDOW_BEFORE:
            n_too_early += len(group)
            continue
        if now > commence + CAPTURE_WINDOW_AFTER:
            n_too_late += len(group)
            continue
        due_event_ids.add(str(event_id))

    print(f"pending captures: {len(pending)}  |  due now: {len(due_event_ids)} event(s)  |  "
          f"too early: {n_too_early}  |  too late (game already underway): {n_too_late}  |  "
          f"no event match: {n_no_event}")
    if not due_event_ids:
        print("nothing in the capture window right now -- run again closer to first pitch.")
        return

    all_rows = []
    for event_id in due_event_ids:
        d = odds_api.get_event_odds(api_key, event_id, markets="pitcher_strikeouts")
        all_rows.extend(odds_api.parse_pitcher_strikeouts_market(event_id, d))

    remaining = odds_api.remaining_quota()
    print(f"Odds API quota remaining: {remaining if remaining is not None else 'unknown'}")

    if not all_rows:
        print("no pitcher_strikeouts odds returned for the due event(s) (book coverage varies close to game time).")
        return

    odds_df = pd.DataFrame(all_rows)
    consensus = odds_api.consensus_over_under(odds_df)
    consensus["name_norm"] = consensus["player_name"].str.lower().str.strip()

    ledger["name_norm"] = ledger["name"].str.lower().str.strip()
    n_captured = 0
    for idx, row in pending.iterrows():
        if str(row.get("event_id")) not in due_event_ids:
            continue
        name_norm = str(row["name"]).lower().strip()
        match = consensus[(consensus["name_norm"] == name_norm) &
                           (abs(consensus["line"] - row["line"]) < 1e-9)]
        if match.empty:
            continue
        m = match.iloc[0]
        ledger.loc[idx, "closing_over_odds"] = m["over_odds"]
        ledger.loc[idx, "closing_under_odds"] = m["under_odds"]
        n_captured += 1

    ledger = ledger.drop(columns=["name_norm"])
    ledger.to_csv(LEDGER_PATH, index=False)
    print(f"captured closing lines for {n_captured} of {len(pending)} pending row(s).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    args = ap.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    run(target)
