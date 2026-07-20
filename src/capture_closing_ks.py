"""
Closing-line capture for the K-props ledgers -- both the stats-only model
(output/ks_paper_ledger.csv) and the research model (output/research_ks_ledger.csv,
see daily_research_ks.py). Free-tier substitute for The Odds API's paid /historical/
endpoint: instead of asking the API "what was the price at time T" after the fact,
snapshot the CURRENT price ourselves, close to first pitch, while the game is still
upcoming. Writes straight into each ledger's closing_over_odds/closing_under_odds
columns -- by the time the reconcile scripts run overnight, CLV is just arithmetic
on numbers already sitting in the ledger, no further API calls needed.

Capturing for both ledgers costs NOTHING extra over capturing for one: the two
ledgers' pending rows are usually the same games (same slate), so the due-event-id
set is computed as the UNION across both, and the single odds pull that follows is
checked against both ledgers' pending rows. Only genuinely new events (a game the
stats model didn't flag but the research model did, or vice versa) add real
incremental cost, and even that is capped at 1 unit/event, same as any other pull.

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
from daily_ks import LEDGER_PATH as STATS_LEDGER_PATH
from daily_research_ks import LEDGER_PATH as RESEARCH_LEDGER_PATH

CAPTURE_WINDOW_BEFORE = timedelta(hours=4)   # don't bother this far ahead of first pitch
CAPTURE_WINDOW_AFTER = timedelta(minutes=20)  # small grace period after commence_time


def _load_pending(ledger_path, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    if not ledger_path.exists():
        return None, None
    ledger = pd.read_csv(ledger_path)
    todays = ledger[ledger["date"] == target_date.isoformat()]
    pending = todays[todays["closing_over_odds"].isna()]
    return ledger, pending


def _capture_into(ledger: pd.DataFrame, pending: pd.DataFrame, due_event_ids: set,
                   consensus: pd.DataFrame, ledger_path) -> int:
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
    ledger.to_csv(ledger_path, index=False)
    return n_captured


def run(target_date: date):
    stats_ledger, stats_pending = _load_pending(STATS_LEDGER_PATH, target_date)
    research_ledger, research_pending = _load_pending(RESEARCH_LEDGER_PATH, target_date)

    if stats_ledger is None and research_ledger is None:
        print("no ledgers found -- nothing to capture yet.")
        return

    total_pending = (len(stats_pending) if stats_pending is not None else 0) + \
                    (len(research_pending) if research_pending is not None else 0)
    if total_pending == 0:
        print(f"no pending closing-line captures for {target_date.isoformat()} "
              f"(either nothing flagged today, or already captured).")
        return

    api_key = odds_api.load_api_key()
    events = {ev["id"]: ev for ev in odds_api.get_events(api_key)}
    now = datetime.now(timezone.utc)

    def due_events_for(pending: pd.DataFrame | None) -> tuple[set, int, int, int]:
        due, n_no_event, n_too_early, n_too_late = set(), 0, 0, 0
        if pending is None or pending.empty:
            return due, 0, 0, 0
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
            due.add(str(event_id))
        return due, n_no_event, n_too_early, n_too_late

    stats_due, s_no_event, s_early, s_late = due_events_for(stats_pending)
    research_due, r_no_event, r_early, r_late = due_events_for(research_pending)
    due_event_ids = stats_due | research_due  # union -- one odds pull covers both ledgers

    print(f"pending captures: {total_pending} ({len(stats_pending) if stats_pending is not None else 0} stats, "
          f"{len(research_pending) if research_pending is not None else 0} research)  |  "
          f"due now: {len(due_event_ids)} event(s)  |  "
          f"too early: {s_early + r_early}  |  too late: {s_late + r_late}  |  no event match: {s_no_event + r_no_event}")
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

    if stats_ledger is not None and not stats_pending.empty:
        n = _capture_into(stats_ledger, stats_pending, due_event_ids, consensus, STATS_LEDGER_PATH)
        print(f"stats ledger: captured closing lines for {n} of {len(stats_pending)} pending row(s).")
    if research_ledger is not None and not research_pending.empty:
        n = _capture_into(research_ledger, research_pending, due_event_ids, consensus, RESEARCH_LEDGER_PATH)
        print(f"research ledger: captured closing lines for {n} of {len(research_pending)} pending row(s).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    args = ap.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    run(target)
