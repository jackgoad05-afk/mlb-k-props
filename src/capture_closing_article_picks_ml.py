"""
Closing-line capture for the article-based moneyline ledger
(output/article_picks_ml_ledger.csv, see daily_article_picks.py). Separate
from capture_closing_ks.py because this is an h2h moneyline market, not
pitcher_strikeouts -- different Odds API endpoint shape, so it doesn't fit
that script's per-event pitcher-props pull. Uses the free bulk h2h endpoint
(1 unit total regardless of how many games), same as daily_article_picks.py's
own pick-time pull, so this costs nothing beyond the single bulk call.

Idempotent: any row that already has a closing price is skipped.

Usage:
    python src/capture_closing_article_picks_ml.py                  # today's not-yet-captured rows
    python src/capture_closing_article_picks_ml.py --date 2026-07-20 # override "today"
"""
from __future__ import annotations

import argparse
from datetime import date, datetime

import pandas as pd

import odds_api
from daily_article_picks import ML_LEDGER_PATH, _representative_h2h_price


def run(target_date: date):
    if not ML_LEDGER_PATH.exists():
        print(f"no ledger found at {ML_LEDGER_PATH} -- nothing to capture yet.")
        return

    ledger = pd.read_csv(ML_LEDGER_PATH)
    todays = ledger[ledger["date"] == target_date.isoformat()]
    pending = todays[todays["closing_home_odds"].isna()]
    if pending.empty:
        print(f"no pending closing-line captures for {target_date.isoformat()} "
              f"(either nothing picked today, or already captured).")
        return

    api_key = odds_api.load_api_key()
    bulk_h2h = odds_api.get_bulk_odds(api_key, markets="h2h")  # free, 1 unit total
    remaining = odds_api.remaining_quota()
    print(f"Odds API quota remaining: {remaining if remaining is not None else 'unknown'}")

    n_captured = 0
    for idx, row in pending.iterrows():
        home_price = _representative_h2h_price(bulk_h2h, row["home_team_name"], row["away_team_name"], "home")
        away_price = _representative_h2h_price(bulk_h2h, row["home_team_name"], row["away_team_name"], "away")
        if home_price is None or away_price is None:
            continue
        ledger.loc[idx, "closing_home_odds"] = home_price
        ledger.loc[idx, "closing_away_odds"] = away_price
        n_captured += 1

    ledger.to_csv(ML_LEDGER_PATH, index=False)
    print(f"captured closing lines for {n_captured} of {len(pending)} pending row(s).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    args = ap.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    run(target)
