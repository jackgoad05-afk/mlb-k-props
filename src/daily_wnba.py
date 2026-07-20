"""
Daily WNBA moneyline + totals pipeline. Paper/tracking only -- see
model_wnba_ml.py / model_wnba_totals.py for the honest backtest picture
(moneyline beats its proxy market on the 2025 holdout; totals is roughly at
parity with the simplest possible heuristic, not a confirmed edge). Not gated
on either backtest result -- everything gets logged, real reconciled paper
results decide what's worth trusting later, same convention as
ml_predictions_ledger.csv for MLB.

Run each morning (the WNBA is in-season roughly May-October, games most days):

    python src/daily_wnba.py                  # live
    python src/daily_wnba.py --dry-run         # model scoring only, no odds pull
    python src/daily_wnba.py --date 2026-07-18 # override "today"

Logs EVERY scheduled game for both market types to output/wnba_paper_ledger.csv
(model prob, market fair prob, edge, price, flagged bool) -- not just flagged
edges -- so the full distribution is there to judge later, not just the tail.
Idempotent per date (re-running today replaces today's rows).
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import fetch_wnba
import odds_api
from features_wnba import FEATURE_COLS, PROCESSED, add_diffs, add_rolling_features
from fetch_odds import american_to_decimal, decimal_to_american
from model_wnba_totals import prob_over

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

LEDGER_PATH = OUTPUT / "wnba_paper_ledger.csv"
EDGE_FLAG_THRESHOLD = 0.03

# Raw feature values persisted into every ledger row for a game, so the dashboard's
# "why" text can be built straight from what the model actually scored with --
# same convention as daily_ks.py's WHY_COLS (recomputing as-of-date rolling features
# after the fact would risk leakage, so these are captured once, at scoring time).
WHY_COLS = ["home_rs_form", "away_rs_form", "home_ra_form", "away_ra_form",
            "home_win_form", "away_win_form", "home_trail_win_pct", "away_trail_win_pct",
            "home_rest_days", "away_rest_days", "total_form_avg", "mu", "sigma"]


def build_todays_features(target_date: date, todays_games: pd.DataFrame) -> pd.DataFrame:
    """Same as-of-date rolling mechanism as daily_ml.py's synthetic-"today"-row
    pattern: real history (prior season cached, current season refreshed) plus
    today's games as NaN-result placeholders, so build_team_rolling_form's
    as-of-date-before form is computed from real games only, never today's own
    (not-yet-known) outcome."""
    season = target_date.year
    prior = fetch_wnba.fetch_schedule(season - 1, refresh=False)
    current = fetch_wnba.fetch_schedule(season, refresh=True)
    history = pd.concat([prior, current], ignore_index=True)

    synth = todays_games[["game_id", "home_team_id", "away_team_id"]].copy()
    synth["season"] = season
    synth["official_date"] = target_date.isoformat()
    synth["home_team_name"] = todays_games["home_team_name"]
    synth["away_team_name"] = todays_games["away_team_name"]
    synth["home_score"] = np.nan
    synth["away_score"] = np.nan
    synth["completed"] = False

    # If an early game today already finished by the time this runs, `current`
    # (completed games only) already carries its real result under the same
    # game_id as the synthetic placeholder -- keep the real row, drop the
    # placeholder, so the rolling merge doesn't see a duplicate key.
    combined = pd.concat([history, synth], ignore_index=True).drop_duplicates(subset=["game_id"], keep="first")
    combined["home_win"] = (combined["home_score"] > combined["away_score"]).astype("Int64")
    combined["total_points"] = combined["home_score"] + combined["away_score"]

    feats = add_rolling_features(combined)
    feats = add_diffs(feats)
    return feats[feats["game_id"].isin(todays_games["game_id"])].reset_index(drop=True)


def score_ml(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(OUTPUT / "model_wnba_ml.joblib")
    calibrator = joblib.load(OUTPUT / "calibrator_wnba_ml.joblib")
    raw_prob = model.predict_proba(df[FEATURE_COLS])[:, 1]
    out = df.copy()
    out["home_win_prob"] = calibrator.transform(raw_prob)
    return out


def score_totals(df: pd.DataFrame) -> pd.DataFrame:
    saved = joblib.load(OUTPUT / "model_wnba_totals.joblib")
    out = df.copy()
    out["mu"] = saved["model"].predict(df[FEATURE_COLS])
    out["sigma"] = saved["sigma"]
    return out


def _representative_price(game_odds: dict, market_key: str, outcome_name: str, point: float | None = None) -> float | None:
    """Median American price for one outcome across books, aggregated in decimal
    space (median of raw American odds is wrong -- see fetch_odds.american_to_decimal).
    Used so reconcile_wnba.py can compute real paper P&L, not just fair-prob edges."""
    decimals = []
    for book in game_odds.get("bookmakers", []):
        for market in book.get("markets", []):
            if market["key"] != market_key:
                continue
            for o in market["outcomes"]:
                if o["name"] != outcome_name:
                    continue
                if point is not None and o.get("point") != point:
                    continue
                decimals.append(american_to_decimal(o["price"]))
    if not decimals:
        return None
    decimals.sort()
    n = len(decimals)
    median_decimal = decimals[n // 2] if n % 2 else (decimals[n // 2 - 1] + decimals[n // 2]) / 2
    return decimal_to_american(median_decimal)


def sportsbook_odds_by_game(scored: pd.DataFrame) -> dict[str, dict]:
    """game_id -> {home_fair_prob, away_fair_prob, ml_n_books, totals_line,
    totals_over_fair_prob, totals_n_books}. Reuses odds_api's existing h2h and
    totals consensus helpers (see odds_api.py's team_totals_consensus, added
    for this build) -- zero new odds-fetching code, just a new sport param."""
    out: dict[str, dict] = {}
    try:
        api_key = odds_api.load_api_key()
        bulk = odds_api.get_bulk_odds(api_key, markets="h2h,totals", sport="basketball_wnba")
    except Exception as e:
        print(f"[warn] sportsbook odds pull failed, proceeding without it: {e}")
        return out

    for g_odds in bulk:
        match = scored[(scored["home_team_name"] == g_odds["home_team"])
                        & (scored["away_team_name"] == g_odds["away_team"])]
        if match.empty:
            continue
        game_id = match.iloc[0]["game_id"]
        entry: dict = {}

        fav = odds_api.h2h_consensus_favorite(g_odds)
        if fav is not None:
            home_fair = fav["favorite_fair_prob"] if fav["favorite_team"] == g_odds["home_team"] else 1 - fav["favorite_fair_prob"]
            entry.update(home_fair_prob=home_fair, away_fair_prob=1 - home_fair, ml_n_books=fav["n_books"],
                         home_price=_representative_price(g_odds, "h2h", g_odds["home_team"]),
                         away_price=_representative_price(g_odds, "h2h", g_odds["away_team"]))

        tot = odds_api.team_totals_consensus(g_odds)
        if tot is not None:
            entry.update(totals_line=tot["line"], totals_over_fair_prob=tot["over_fair_prob"],
                         totals_n_books=tot["n_books"],
                         over_price=_representative_price(g_odds, "totals", "Over", tot["line"]),
                         under_price=_representative_price(g_odds, "totals", "Under", tot["line"]))

        if entry:
            out[game_id] = entry
    return out


def run(target_date: date, dry_run: bool):
    print(f"=== daily WNBA predictions: {target_date.isoformat()} ===")
    todays_games = fetch_wnba.fetch_games_on_date(target_date.isoformat())
    print(f"games today: {len(todays_games)}")
    if todays_games.empty:
        print("no WNBA games found for this date.")
        return

    feats = build_todays_features(target_date, todays_games)
    scored = score_ml(feats)
    scored = score_totals(scored)

    print("\nmodel predictions:")
    with pd.option_context("display.width", 160):
        print(scored[["home_team_name", "away_team_name", "home_win_prob", "mu"]]
              .sort_values("home_win_prob", ascending=False).to_string(index=False))

    ledger_rows = []
    if dry_run:
        print("\n--dry-run: skipping sportsbook odds pull; logging model-only rows.")
    odds_by_game = {} if dry_run else sportsbook_odds_by_game(scored)

    for _, row in scored.iterrows():
        odds = odds_by_game.get(row["game_id"], {})
        base = {"date": target_date.isoformat(), "game_id": row["game_id"],
                "home_team_name": row["home_team_name"], "away_team_name": row["away_team_name"],
                "logged_at": datetime.now().isoformat(timespec="seconds"),
                **{c: row[c] for c in WHY_COLS}}

        for side, model_prob in [("home", row["home_win_prob"]), ("away", 1 - row["home_win_prob"])]:
            market_prob = odds.get(f"{side}_fair_prob")
            edge = model_prob - market_prob if market_prob is not None else np.nan
            ledger_rows.append({**base, "market_type": "moneyline", "side": side, "line": np.nan,
                                 "model_prob": model_prob, "market_prob": market_prob, "edge": edge,
                                 "price": odds.get(f"{side}_price"), "n_books": odds.get("ml_n_books"),
                                 "flagged": bool(pd.notna(edge) and edge >= EDGE_FLAG_THRESHOLD)})

        line = odds.get("totals_line")
        market_over_prob = odds.get("totals_over_fair_prob")
        for side in ["over", "under"]:
            if line is not None:
                model_p = prob_over(np.array([row["mu"]]), row["sigma"], line)[0]
                model_prob = model_p if side == "over" else 1 - model_p
                market_prob = market_over_prob if side == "over" else (1 - market_over_prob if market_over_prob is not None else None)
                edge = model_prob - market_prob if market_prob is not None else np.nan
            else:
                model_prob = market_prob = edge = np.nan
            ledger_rows.append({**base, "market_type": "totals", "side": side, "line": line,
                                 "model_prob": model_prob, "market_prob": market_prob, "edge": edge,
                                 "price": odds.get(f"{side}_price"), "n_books": odds.get("totals_n_books"),
                                 "flagged": bool(pd.notna(edge) and edge >= EDGE_FLAG_THRESHOLD)})

    ledger_rows = pd.DataFrame(ledger_rows)
    for c in ["closing_price", "actual_result", "correct", "pnl"]:
        ledger_rows[c] = np.nan
    ledger_rows["correct"] = ledger_rows["correct"].astype("object")
    ledger_rows["actual_result"] = ledger_rows["actual_result"].astype("object")

    n_flagged = int(ledger_rows["flagged"].sum())
    print(f"\nlogged {len(ledger_rows)} rows ({len(todays_games)} games x moneyline home/away + totals over/under) "
          f"-- {n_flagged} flagged (edge >= {EDGE_FLAG_THRESHOLD:.0%})")
    if n_flagged:
        with pd.option_context("display.width", 160):
            print(ledger_rows[ledger_rows["flagged"]][
                ["home_team_name", "away_team_name", "market_type", "side", "line",
                 "model_prob", "market_prob", "edge"]].to_string(index=False))

    if LEDGER_PATH.exists():
        existing = pd.read_csv(LEDGER_PATH)
        existing["correct"] = existing["correct"].astype("object")
        existing["actual_result"] = existing["actual_result"].astype("object")
        existing = existing[existing["date"] != target_date.isoformat()]
        ledger_rows = pd.concat([existing, ledger_rows], ignore_index=True)
    ledger_rows.to_csv(LEDGER_PATH, index=False)
    print(f"ledger updated: {LEDGER_PATH} ({len(ledger_rows)} total rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    ap.add_argument("--dry-run", action="store_true", help="skip the sportsbook odds pull")
    args = ap.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    run(target, args.dry_run)
