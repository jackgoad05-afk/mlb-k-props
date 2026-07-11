"""
Daily strikeout-prop pipeline. Run each morning:

    python src/daily_ks.py                  # live: pulls today's odds, needs ODDS_API_KEY
    python src/daily_ks.py --dry-run         # model predictions only, no odds API call
    python src/daily_ks.py --date 2026-07-10 # override "today" (testing / backfill)

Pulls today's probable starters (MLB Stats API, free), builds each one's as-of-date
rolling K form using the same leak-free logic as the backtest (rolling.py), scores
them with the fitted NB2 model (model_ks.py's saved params), pulls today's
pitcher_strikeouts market from The Odds API, flags edges >= 3%, and appends every
flagged edge to a paper-trading ledger (output/ks_paper_ledger.csv) with a blank
closing_line/actual_so/result to be filled in by a later reconciliation pass once
games finish -- that's what makes CLV tracking possible.

Requires an ODDS_API_KEY -- see odds_api.py for setup (env var or .env file). Without
it, everything except the live odds pull and edge-flagging works -- use --dry-run to
sanity-check the model side on its own.
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
from model_ks import EXPOSURE_COL, REGRESSOR_COLS, nb2_np, prob_over
from rolling import build_starter_k_form

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
LEDGER_PATH = OUTPUT / "ks_paper_ledger.csv"
DAILY_SCORES_PATH = OUTPUT / "ks_daily_scores.csv"    # every probable starter, overwritten each run
DAILY_MATCHED_PATH = OUTPUT / "ks_daily_matched.csv"  # every starter matched to real odds (not just flagged)

EDGE_FLAG_THRESHOLD = 0.03
PROP_LINES = [4.5, 5.5, 6.5]


def fetch_todays_probables(target_date: date) -> pd.DataFrame:
    d = fetch._get(f"{fetch.STATSAPI_BASE}/schedule", params={
        "sportId": 1, "date": target_date.isoformat(), "gameType": "R", "hydrate": "probablePitcher",
    })
    rows = []
    for date_block in d.get("dates", []):
        for g in date_block.get("games", []):
            home, away = g["teams"]["home"], g["teams"]["away"]
            for side, team, opp in [("home", home, away), ("away", away, home)]:
                pp = team.get("probablePitcher")
                if not pp:
                    continue
                rows.append({
                    "game_pk": g["gamePk"], "official_date": target_date.isoformat(),
                    "side": side, "mlbID": pp["id"], "name": pp["fullName"],
                    "team_id": team["team"]["id"], "opponent_team_id": opp["team"]["id"],
                    "opponent_name": opp["team"]["name"],
                })
    return pd.DataFrame(rows)


def build_todays_features(probables: pd.DataFrame, target_date: date) -> pd.DataFrame:
    season = target_date.year
    lag_season = season - 1

    ids = probables["mlbID"].unique().tolist()
    game_logs = fetch.fetch_starter_game_logs(season, ids, refresh=True, sleep=0.1)

    # Append a synthetic "today" row per starter so build_starter_k_form computes
    # each one's as-of-date-BEFORE-today rolling state from his real starts so far
    # this season -- exactly the same mechanism used for every historical start in
    # the backtest, just with today's date instead of a past game's date.
    synth = probables[["mlbID"]].copy()
    synth["season"] = season
    synth["official_date"] = target_date.isoformat()
    synth["game_pk"] = -1  # sentinel: never a real historical game_pk
    for c in ["ip", "bf", "so", "bb", "hr", "er", "pitches"]:
        synth[c] = np.nan
    synth["is_home"] = np.nan
    synth["opponent_team_id"] = np.nan

    combined = pd.concat([game_logs, synth], ignore_index=True)
    hands = fetch.fetch_pitcher_handedness(ids)

    pitchers = pd.read_parquet(PROCESSED / "pitcher_season.parquet")
    teams = pd.read_parquet(PROCESSED / "team_season.parquet")

    form = build_starter_k_form(combined, pitchers)
    today_form = form[form["game_pk"] == -1].drop(columns=["game_pk", "season"])

    df = probables.merge(today_form, on="mlbID", how="left")
    df = df.merge(hands, on="mlbID", how="left")

    p_merge = df[["mlbID"]].merge(
        pitchers[pitchers["season"] == lag_season][["mlbID", "whiff_pct", "k_bb_pct", "fastball_velo"]],
        on="mlbID", how="left")
    league_avg = pitchers[(pitchers["season"] == lag_season) & (pitchers["IP"] >= 20)][
        ["whiff_pct", "k_bb_pct", "fastball_velo"]].mean()
    for c, out_c in [("whiff_pct", "season_lag_whiff_pct"), ("k_bb_pct", "season_lag_k_bb_pct"),
                      ("fastball_velo", "season_lag_fastball_velo")]:
        df[out_c] = p_merge[c].fillna(league_avg[c]).values

    split = df["throws"].map({"L": "vs_lhp", "R": "vs_rhp"}).fillna("overall")
    team_key = pd.DataFrame({"team_id": df["opponent_team_id"], "split": split})
    team_lag = teams[teams["season"] == lag_season]
    team_merge = team_key.merge(team_lag, on=["team_id", "split"], how="left")
    team_league_avg = team_lag.groupby("split")["k_pct"].mean()
    missing = team_merge["k_pct"].isna()
    if missing.any():
        team_merge.loc[missing, "k_pct"] = team_key.loc[missing, "split"].map(team_league_avg).values
    df["opp_off_kpct"] = team_merge["k_pct"].values

    return df


def score(df: pd.DataFrame) -> pd.DataFrame:
    saved = joblib.load(OUTPUT / "model_ks.joblib")
    beta, alpha = saved["beta"], saved["alpha"]
    mu_s, sd_s = saved["mu_scale"], saved["sd_scale"]

    X = df[REGRESSOR_COLS]
    X_std = (X - mu_s) / sd_s
    X_std.insert(0, "const", 1.0)
    X_std = X_std[beta.index]

    mu = np.exp(X_std.values @ beta.values) * df[EXPOSURE_COL].values
    out = df.copy()
    out["mu"] = mu
    for line in PROP_LINES:
        out[f"model_p_over_{line}"] = prob_over(mu, alpha, line)
    return out


def fetch_odds_props(target_date: date, api_key: str) -> pd.DataFrame:
    events = odds_api.get_events(api_key)
    todays_events = [ev for ev in events if ev["commence_time"][:10] == target_date.isoformat()]
    rows = []
    for ev in todays_events:
        d = odds_api.get_event_odds(api_key, ev["id"], markets="pitcher_strikeouts")
        rows.extend(odds_api.parse_pitcher_strikeouts_market(ev["id"], d))
    return pd.DataFrame(rows)


def run(target_date: date, dry_run: bool):
    print(f"=== daily K props: {target_date.isoformat()} ===")
    probables = fetch_todays_probables(target_date)
    print(f"probable starters: {len(probables)}")
    if probables.empty:
        print("no probable starters found for this date (too early in the day, off-day, or postseason).")
        return

    feats = build_todays_features(probables, target_date)
    scored = score(feats)

    cols = ["mlbID", "name", "team_id", "opponent_name", "mu"] + [f"model_p_over_{l}" for l in PROP_LINES]
    print("\nmodel predictions:")
    with pd.option_context("display.width", 160):
        print(scored[cols].sort_values("mu", ascending=False).to_string(index=False))

    daily_scores = scored[cols].copy()
    daily_scores.insert(0, "date", target_date.isoformat())
    daily_scores.to_csv(DAILY_SCORES_PATH, index=False)

    if dry_run:
        print("\n--dry-run: skipping odds pull. Set ODDS_API_KEY and drop --dry-run to flag edges.")
        return

    api_key = odds_api.load_api_key()
    odds = fetch_odds_props(target_date, api_key)
    if odds.empty:
        print("\nno pitcher_strikeouts odds returned for today (book coverage varies by day/game time).")
        return
    remaining = odds_api.remaining_quota()
    print(f"\nOdds API quota remaining: {remaining if remaining is not None else 'unknown'}")

    consensus = odds_api.consensus_over_under(odds)

    scored["name_norm"] = scored["name"].str.lower().str.strip()
    consensus["name_norm"] = consensus["player_name"].str.lower().str.strip()
    matched = consensus.merge(scored, on="name_norm", how="inner")

    def model_p_for_line(row):
        nearest = min(PROP_LINES, key=lambda l: abs(l - row["line"]))
        return row[f"model_p_over_{nearest}"]

    matched["model_p_over"] = matched.apply(model_p_for_line, axis=1)
    matched["edge"] = matched["model_p_over"] - matched["over_prob_fair"]
    matched["under_edge"] = (1 - matched["model_p_over"]) - (1 - matched["over_prob_fair"])
    matched["bet_side"] = np.where(matched["edge"] >= matched["under_edge"], "over", "under")
    matched["bet_edge"] = np.where(matched["bet_side"] == "over", matched["edge"], matched["under_edge"])

    daily_matched = matched[["mlbID", "name", "opponent_name", "line", "model_p_over", "over_prob_fair",
                              "over_odds", "under_odds", "n_books"]].copy()
    daily_matched.insert(0, "date", target_date.isoformat())
    daily_matched.to_csv(DAILY_MATCHED_PATH, index=False)

    flagged = matched[matched["bet_edge"] >= EDGE_FLAG_THRESHOLD].copy()
    print(f"\nmatched to odds: {len(matched)}  |  flagged (edge >= {EDGE_FLAG_THRESHOLD:.0%}): {len(flagged)}")
    if len(flagged):
        with pd.option_context("display.width", 160):
            print(flagged[["name", "line", "bet_side", "bet_edge", "model_p_over", "over_prob_fair",
                            "over_odds", "under_odds", "n_books"]].to_string(index=False))

    WHY_COLS = ["trail_k_per9_3s", "trail_k_per9_30d", "season_lag_whiff_pct", "opp_off_kpct",
                "days_rest", "mu"]
    ledger_rows = flagged[["mlbID", "game_pk", "event_id", "name", "opponent_name", "line", "bet_side",
                            "bet_edge", "model_p_over", "over_prob_fair", "over_odds", "under_odds",
                            "n_books"] + WHY_COLS].copy()
    ledger_rows.insert(0, "date", target_date.isoformat())
    ledger_rows["logged_at"] = datetime.now().isoformat(timespec="seconds")
    for c in ["closing_over_odds", "closing_under_odds", "actual_so", "pnl", "clv"]:
        ledger_rows[c] = np.nan
    ledger_rows["result"] = pd.array([None] * len(ledger_rows), dtype="object")

    if LEDGER_PATH.exists():
        existing = pd.read_csv(LEDGER_PATH)
        ledger_rows = pd.concat([existing, ledger_rows], ignore_index=True)
    ledger_rows.to_csv(LEDGER_PATH, index=False)
    print(f"\nledger updated: {LEDGER_PATH} ({len(ledger_rows)} total rows)")
    print("NOTE: closing_over_odds/closing_under_odds/actual_so/result/clv are blank --")
    print("      fill these in with a reconciliation pass after games finish to track real CLV.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    ap.add_argument("--dry-run", action="store_true", help="skip the odds API call, show model predictions only")
    args = ap.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    run(target, args.dry_run)
