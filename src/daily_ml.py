"""
Pure moneyline prediction pipeline -- NO odds as a model input. Team form, matchups,
pitcher/bullpen quality, and park factor only, same feature set model_ml.py was
trained on (see features.py). Tracking/paper only: the moneyline model already failed
the spec's kill criteria against real odds (CLAUDE.md, Track 1) and does not bet. This
answers a different, narrower question -- can it pick winners straight-up better than
chance, and is it finding non-obvious winners or just parroting the market favorite.

Run each morning (piggybacks on the same schedule as the K-props pull):
    python src/daily_ml.py                  # live
    python src/daily_ml.py --dry-run         # skip the (cheap) market-favorite pull
    python src/daily_ml.py --date 2026-07-12 # override "today"

Two outputs, kept deliberately separate:
  - output/ml_predictions_ledger.csv: predicted winner, win probability, and a plain-
    language "why" per game. This is what the dashboard shows.
  - output/ml_market_comparison.csv: today's market favorite per game (from the cheap
    bulk h2h endpoint, ~1 unit total for the whole slate) and whether the model agreed
    with it. NOT surfaced in streamlit_app.py -- for direct inspection only, so a look
    at "today's predictions" isn't anchored by knowing who the market likes.

Both are idempotent per date: re-running today just replaces today's rows rather than
duplicating them (ks_paper_ledger.csv does NOT have this guard yet -- worth porting
back if daily_ks.py ever needs to be safely re-run mid-day).
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import fetch
import odds_api
from features import (FEATURE_COLS, add_diffs, add_park_factor, add_starter_rolling_features,
                       add_team_rolling_features, build_pitcher_features, build_team_offense_features)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

PREDICTIONS_LEDGER_PATH = OUTPUT / "ml_predictions_ledger.csv"
MARKET_COMPARISON_PATH = OUTPUT / "ml_market_comparison.csv"  # not read by streamlit_app.py, by design

# Same priority order permutation importance found in the Session 2.1 backtest
# (win_form_diff ~3x everything else) -- used only to pick which factors to surface
# in the "why," not to change how the model itself weighs them.
FACTOR_INFO = {
    "win_form_diff":          ("recent win form", "home_win_form", "away_win_form", "pct"),
    "off_index_diff":         ("offense quality (park-adj.)", "home_off_index", "away_off_index", "num1"),
    "ra_form_diff":           ("recent runs allowed", "home_ra_form", "away_ra_form", "num2"),
    "k_bb_pct_diff":          ("starter K-BB% (season)", "home_k_bb_pct", "away_k_bb_pct", "pct"),
    "rs_form_diff":           ("recent runs scored", "home_rs_form", "away_rs_form", "num2"),
    "trail_fip_diff":         ("starter's current-form FIP", "home_trail_fip", "away_trail_fip", "num2"),
    "trail_win_pct_diff":     ("trailing win% (last 10)", "home_trail_win_pct", "away_trail_win_pct", "pct"),
    "trail_k_bb_pct_diff":    ("starter's current-form K-BB%", "home_trail_k_bb_pct", "away_trail_k_bb_pct", "pct"),
    "fip_diff":               ("starter FIP (season)", "home_fip", "away_fip", "num2"),
    "xwoba_against_diff":     ("starter xwOBA against", "home_xwoba_against", "away_xwoba_against", "num3"),
    "whiff_pct_diff":         ("starter whiff%", "home_whiff_pct", "away_whiff_pct", "pctraw"),
    "days_rest_diff":         ("rest", "home_days_rest", "away_days_rest", "days"),
    "off_kpct_diff":          ("offense strikeout rate", "home_off_kpct", "away_off_kpct", "pct"),
    "barrel_pct_allowed_diff": ("starter barrel% allowed", "home_barrel_pct_allowed", "away_barrel_pct_allowed", "pctraw"),
    "fastball_velo_diff":     ("starter fastball velocity", "home_fastball_velo", "away_fastball_velo", "mph"),
}
FACTOR_PRIORITY = list(FACTOR_INFO.keys())


def _fmt(val, kind: str) -> str:
    if pd.isna(val):
        return "n/a"
    if kind == "pct":
        return f"{val:.1%}"
    if kind == "pctraw":
        return f"{val:.1f}%"
    if kind == "num1":
        return f"{val:.1f}"
    if kind == "num2":
        return f"{val:.2f}"
    if kind == "num3":
        return f"{val:.3f}"
    if kind == "days":
        return f"{val:.0f}d"
    if kind == "mph":
        return f"{val:.1f} mph"
    return str(val)


def fetch_todays_games(target_date: date) -> pd.DataFrame:
    d = fetch._get(f"{fetch.STATSAPI_BASE}/schedule", params={
        "sportId": 1, "date": target_date.isoformat(), "gameType": "R", "hydrate": "probablePitcher",
    })
    rows = []
    for date_block in d.get("dates", []):
        for g in date_block.get("games", []):
            home, away = g["teams"]["home"], g["teams"]["away"]
            home_pp, away_pp = home.get("probablePitcher"), away.get("probablePitcher")
            if not home_pp or not away_pp:
                continue
            rows.append({
                "game_pk": g["gamePk"], "season": target_date.year, "official_date": target_date.isoformat(),
                "home_team_id": home["team"]["id"], "home_team_name": home["team"]["name"],
                "away_team_id": away["team"]["id"], "away_team_name": away["team"]["name"],
                "home_probable_pitcher_id": home_pp["id"], "home_probable_pitcher_name": home_pp["fullName"],
                "away_probable_pitcher_id": away_pp["id"], "away_probable_pitcher_name": away_pp["fullName"],
            })
    return pd.DataFrame(rows)


def _rolling_team_games(target_date: date, todays_games: pd.DataFrame) -> pd.DataFrame:
    """Prior season (cached, stable) + current season to date (refreshed) + today's
    games as synthetic NaN-result rows, for add_team_rolling_features's as-of-date
    rolling win%/runs-scored/allowed. Real results only, never today's own outcome."""
    prior = fetch.fetch_schedule(target_date.year - 1, refresh=False)
    current = fetch.fetch_schedule(target_date.year, refresh=True)
    games = pd.concat([prior, current], ignore_index=True)
    games = games.dropna(subset=["home_score", "away_score"])
    games["home_score"] = games["home_score"].astype(int)
    games["away_score"] = games["away_score"].astype(int)
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games["official_date"] = pd.to_datetime(games["official_date"])

    synth = todays_games[["game_pk", "season", "official_date", "home_team_id", "away_team_id"]].copy()
    synth["official_date"] = pd.to_datetime(synth["official_date"])
    for c in ["home_score", "away_score", "home_win"]:
        synth[c] = np.nan

    # If an early game in today's slate has already finished by the time this runs,
    # fetch_schedule (completed games only) already carries its real result under the
    # same game_pk as the synthetic placeholder below -- keep the real row, drop the
    # placeholder, so build_team_rolling_form's merge on game_pk doesn't see a dup key.
    combined = pd.concat([games[["game_pk", "season", "official_date", "home_team_id", "away_team_id",
                                  "home_score", "away_score", "home_win"]], synth], ignore_index=True)
    return combined.drop_duplicates(subset=["game_pk"], keep="first")


def _rolling_starter_games(target_date: date, todays_games: pd.DataFrame) -> pd.DataFrame:
    ids = pd.concat([todays_games["home_probable_pitcher_id"], todays_games["away_probable_pitcher_id"]]).unique().tolist()
    game_logs = fetch.fetch_starter_game_logs(target_date.year, ids, refresh=True, sleep=0.1)

    synth_rows = []
    for _, row in todays_games.iterrows():
        for id_col in ["home_probable_pitcher_id", "away_probable_pitcher_id"]:
            synth_rows.append({"mlbID": row[id_col], "season": target_date.year,
                                "official_date": target_date.isoformat(), "game_pk": row["game_pk"]})
    synth = pd.DataFrame(synth_rows)
    for c in ["ip", "bf", "so", "bb", "hr", "er", "pitches"]:
        synth[c] = np.nan
    synth["is_home"] = np.nan
    synth["opponent_team_id"] = np.nan

    # Same collision as _rolling_team_games above: a pitcher whose early game already
    # finished today has a real completed-start row under this game_pk already -- keep
    # it, drop the synthetic placeholder, so add_starter_rolling_features's merge on
    # (game_pk, mlbID) doesn't see a dup key.
    combined = pd.concat([game_logs, synth], ignore_index=True)
    return combined.drop_duplicates(subset=["mlbID", "game_pk"], keep="first")


def build_todays_ml_features(target_date: date, todays_games: pd.DataFrame) -> pd.DataFrame:
    pitchers = pd.read_parquet(PROCESSED / "pitcher_season.parquet")
    teams = pd.read_parquet(PROCESSED / "team_season.parquet")
    hands = pd.read_csv(RAW / "pitcher_handedness.csv")

    df = build_pitcher_features(todays_games, pitchers, hands)
    df = build_team_offense_features(df, teams)

    starter_logs = _rolling_starter_games(target_date, todays_games)
    df = add_starter_rolling_features(df, starter_logs, pitchers)

    team_games = _rolling_team_games(target_date, todays_games)
    df = add_team_rolling_features(df, team_games)

    df = add_park_factor(df, teams)
    df = add_diffs(df)
    return df


def score_ml(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(OUTPUT / "model_ml.joblib")
    calibrator = joblib.load(OUTPUT / "calibrator_ml.joblib")

    raw_prob = model.predict_proba(df[FEATURE_COLS])[:, 1]
    home_win_prob = calibrator.transform(raw_prob)

    out = df.copy()
    out["home_win_prob"] = home_win_prob
    out["predicted_winner"] = np.where(home_win_prob >= 0.5, out["home_team_name"], out["away_team_name"])
    out["predicted_win_prob"] = np.where(home_win_prob >= 0.5, home_win_prob, 1 - home_win_prob)
    return out


def why_prediction(row: pd.Series) -> tuple[list[str], str]:
    home_favored = row["home_win_prob"] >= 0.5
    sign = 1 if home_favored else -1
    winner = row["home_team_name"] if home_favored else row["away_team_name"]

    stats, supporting_labels = [], []
    for col in FACTOR_PRIORITY:
        label, home_col, away_col, kind = FACTOR_INFO[col]
        d = row.get(col)
        if pd.isna(d):
            continue
        h_val, a_val = row.get(home_col), row.get(away_col)
        stats.append(f"{label}: home {_fmt(h_val, kind)} / away {_fmt(a_val, kind)}")
        if d * sign > 0 and len(supporting_labels) < 3:
            supporting_labels.append(label)

    if supporting_labels:
        if len(supporting_labels) == 1:
            body = supporting_labels[0]
        else:
            body = ", ".join(supporting_labels[:-1]) + " and " + supporting_labels[-1]
        summary = f"{winner} favored mainly on {body}."
    else:
        summary = f"{winner} favored by the model's overall read; no single factor dominates."

    return stats[:6], summary


def log_market_comparison(target_date: date, scored: pd.DataFrame) -> None:
    try:
        api_key = odds_api.load_api_key()
        bulk = odds_api.get_bulk_odds(api_key, markets="h2h")
    except Exception as e:
        print(f"[warn] market-comparison pull failed, skipping (predictions ledger unaffected): {e}")
        return

    todays = [g for g in bulk if g["commence_time"][:10] == target_date.isoformat()]
    rows = []
    for g in todays:
        fav = odds_api.h2h_consensus_favorite(g)
        if fav is None:
            continue
        match = scored[(scored["home_team_name"] == g["home_team"]) & (scored["away_team_name"] == g["away_team"])]
        if match.empty:
            continue
        m = match.iloc[0]
        rows.append({
            "date": target_date.isoformat(), "game_pk": int(m["game_pk"]),
            "home_team_name": g["home_team"], "away_team_name": g["away_team"],
            "market_favorite": fav["favorite_team"], "market_favorite_fair_prob": fav["favorite_fair_prob"],
            "n_books": fav["n_books"], "model_predicted_winner": m["predicted_winner"],
            "model_agrees_with_favorite": m["predicted_winner"] == fav["favorite_team"],
        })
    if not rows:
        print("no market-comparison rows built (no h2h odds matched today's games yet).")
        return

    new_rows = pd.DataFrame(rows)
    if MARKET_COMPARISON_PATH.exists():
        existing = pd.read_csv(MARKET_COMPARISON_PATH)
        existing = existing[existing["date"] != target_date.isoformat()]
        new_rows = pd.concat([existing, new_rows], ignore_index=True)
    new_rows.to_csv(MARKET_COMPARISON_PATH, index=False)

    agree_rate = pd.DataFrame(rows)["model_agrees_with_favorite"].mean()
    print(f"market comparison logged: {len(rows)} games, model agreed with favorite {agree_rate:.0%} of the time "
          f"(private file, not shown on dashboard)")


def run(target_date: date, dry_run: bool):
    print(f"=== daily moneyline predictions: {target_date.isoformat()} ===")
    todays_games = fetch_todays_games(target_date)
    print(f"games with both probable starters announced: {len(todays_games)}")
    if todays_games.empty:
        print("no games with announced starters yet for this date.")
        return

    feats = build_todays_ml_features(target_date, todays_games)
    scored = score_ml(feats)

    cols = ["home_team_name", "away_team_name", "predicted_winner", "predicted_win_prob"]
    print("\npredictions:")
    with pd.option_context("display.width", 160):
        print(scored[cols].sort_values("predicted_win_prob", ascending=False).to_string(index=False))

    ledger_rows = []
    for _, row in scored.iterrows():
        stats, summary = why_prediction(row)
        ledger_rows.append({
            "date": target_date.isoformat(), "game_pk": row["game_pk"],
            "home_team_name": row["home_team_name"], "away_team_name": row["away_team_name"],
            "predicted_winner": row["predicted_winner"], "home_win_prob": row["home_win_prob"],
            "predicted_win_prob": row["predicted_win_prob"], "why_stats": " | ".join(stats),
            "why_summary": summary, "logged_at": datetime.now().isoformat(timespec="seconds"),
            "actual_winner": np.nan, "correct": pd.NA,
        })
    ledger_rows = pd.DataFrame(ledger_rows)
    ledger_rows["actual_winner"] = ledger_rows["actual_winner"].astype("object")
    ledger_rows["correct"] = ledger_rows["correct"].astype("object")

    if PREDICTIONS_LEDGER_PATH.exists():
        existing = pd.read_csv(PREDICTIONS_LEDGER_PATH)
        existing["correct"] = existing["correct"].astype("object")
        existing = existing[existing["date"] != target_date.isoformat()]
        ledger_rows = pd.concat([existing, ledger_rows], ignore_index=True)
    ledger_rows.to_csv(PREDICTIONS_LEDGER_PATH, index=False)
    print(f"\npredictions ledger updated: {PREDICTIONS_LEDGER_PATH} ({len(ledger_rows)} total rows)")

    if dry_run:
        print("\n--dry-run: skipping market-favorite comparison pull.")
        return
    log_market_comparison(target_date, scored)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    ap.add_argument("--dry-run", action="store_true", help="skip the market-favorite comparison pull")
    args = ap.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    run(target, args.dry_run)
