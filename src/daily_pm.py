"""
Prediction-market comparison pipeline (paper only): compares model_ml's real,
calibrated moneyline probabilities against live Polymarket + Kalshi prices, with
the sportsbook consensus line logged alongside for the same game. Both platforms
are free reads, no API key.

**Why this exists**: model_ml already failed its kill criteria against real SHARP
sportsbook closing lines (CLAUDE.md, Track 1 -- ROI -5.2%, CLV -1.42%, negative in
every edge tier). This is a separate, narrower measurement: does the model find
real edges against Polymarket/Kalshi's thinner, less efficient pricing that it
couldn't find against sharp books? Paper only, reported honestly either way.

**Scope: moneyline only gets a real model edge.** model_ml.py is the only fitted
model in this repo -- there's no runs-total model, so totals rows are logged with
model_prob left blank and are never flagged. A totals model is a separate, future,
multi-session build (like model_ks.py was), not a quick add here.

**No vig adjustment on the PM side, by design**: `pm_implied_prob` is the actual
best-ask price you'd transact at, not a synthetic no-vig-devigged number. That IS
what "no vig adjustment" means concretely for a market quoted as a single 0-1
price rather than two-sided American odds. The sportsbook comparison column
still uses the existing no-vig consensus (odds_api.h2h_consensus_favorite) since
that's the established convention daily_ml.py already uses for that side.

**Kalshi team matching is unverified against a real game as of this writing**
(2026-07-13, MLB All-Star break -- 0 real games today/tomorrow per the Stats API;
the only open KXMLBGAME market is the All-Star Game itself, whose sub-titles are
league codes "AL"/"NL", not team codes). Matching here is built against
team_season.parquet's team_id -> abbreviation table (the same MLB Stats API
abbreviations used elsewhere in this repo) as a first guess. Any Kalshi game
where neither side's sub_title matches a known abbreviation is skipped and
counted, not guessed at -- validate the real convention once the season resumes
(~July 17) before trusting Kalshi rows in the ledger.

Run each morning (or any time -- both platforms are live order books, not a daily
snapshot API):

    python src/daily_pm.py                  # live
    python src/daily_pm.py --dry-run         # model scoring only, no PM/sportsbook pulls
    python src/daily_pm.py --date 2026-07-18 # override "today"
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import fetch_prediction_markets as pmkt
import odds_api
from daily_ml import build_todays_ml_features, fetch_todays_games, score_ml

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
PROCESSED = ROOT / "data" / "processed"

LEDGER_PATH = OUTPUT / "pm_paper_ledger.csv"
DAILY_MATCHED_PATH = OUTPUT / "pm_daily_matched.csv"  # every matched row, not just flagged

EDGE_FLAG_THRESHOLD = 0.03
MIN_DEPTH_USD = 50.0
SLIPPAGE = 0.01  # $ tolerance around the touch price when summing order-book depth


def _team_abbrev_map(season: int) -> dict[int, str]:
    teams = pd.read_parquet(PROCESSED / "team_season.parquet")
    row = teams[teams["season"] == season][["team_id", "abbreviation"]].drop_duplicates()
    return dict(zip(row["team_id"], row["abbreviation"]))


def _game_date_matches(iso_ts: str, target_date: date) -> bool:
    """Kalshi/Polymarket game timestamps are UTC; a late-evening ET first pitch can
    already be past midnight UTC. Approximate ET with a fixed -5h offset (not DST-
    exact, but good enough to disambiguate same-matchup games on consecutive days --
    see fetch_odds.py's near-identical comment on this exact problem)."""
    ts = pd.to_datetime(iso_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    et_date = (ts - timedelta(hours=5)).date()
    return et_date == target_date or ts.date() == target_date


# ---------------------------------------------------------------------------
# Polymarket matching
# ---------------------------------------------------------------------------

def match_polymarket_moneyline(games: pd.DataFrame, target_date: date) -> list[dict]:
    events = pmkt.get_polymarket_mlb_events()
    rows = []
    for ev in events:
        parsed = pmkt.parse_polymarket_moneyline(ev)
        if parsed is None or not parsed.get("game_start_time"):
            continue
        if not _game_date_matches(parsed["game_start_time"], target_date):
            continue
        pm_team_names = set(parsed["teams"].keys())
        match = games[(games["home_team_name"].isin(pm_team_names)) & (games["away_team_name"].isin(pm_team_names))]
        if match.empty:
            continue
        g = match.iloc[0]
        for side, team_name in [("home", g["home_team_name"]), ("away", g["away_team_name"])]:
            info = parsed["teams"].get(team_name)
            if info is None or info["best_ask"] is None:
                continue
            rows.append({
                "game_pk": g["game_pk"], "market": "polymarket", "market_type": "moneyline",
                "side": side, "line": np.nan, "pm_price": info["best_ask"],
                "pm_implied_prob": info["best_ask"], "_token_id": info["token_id"],
            })
    return rows


def match_polymarket_totals(games: pd.DataFrame, target_date: date) -> list[dict]:
    events = pmkt.get_polymarket_mlb_events()
    rows = []
    for ev in events:
        for parsed in pmkt.parse_polymarket_totals(ev):
            if parsed["line"] is None:
                continue
            gst = None
            for m in ev.get("markets", []):
                if m.get("id") == parsed["market_id"]:
                    gst = m.get("gameStartTime")
            if not gst or not _game_date_matches(gst, target_date):
                continue
            title = ev.get("title", "")
            match = games[games.apply(lambda g: g["home_team_name"] in title and g["away_team_name"] in title, axis=1)]
            if match.empty:
                continue
            g = match.iloc[0]
            for side in ["Over", "Under"]:
                info = parsed["teams"].get(side)
                if info is None or info["best_ask"] is None:
                    continue
                rows.append({
                    "game_pk": g["game_pk"], "market": "polymarket", "market_type": "totals",
                    "side": side.lower(), "line": parsed["line"], "pm_price": info["best_ask"],
                    "pm_implied_prob": info["best_ask"], "_token_id": info["token_id"],
                })
    return rows


def add_polymarket_depth(rows: list[dict]) -> None:
    pm_rows = [r for r in rows if r["market"] == "polymarket"]
    for i, r in enumerate(pm_rows):
        try:
            book = pmkt.get_polymarket_orderbook(r["_token_id"])
            r["depth_usd"] = pmkt.usd_depth_polymarket(book, "asks", r["pm_price"], SLIPPAGE)
        except Exception as e:
            print(f"[warn] Polymarket orderbook fetch failed for token {r['_token_id']}: {e}")
            r["depth_usd"] = np.nan
        if (i + 1) % 10 == 0 or (i + 1) == len(pm_rows):
            print(f"  ... Polymarket depth {i + 1}/{len(pm_rows)}")


# ---------------------------------------------------------------------------
# Kalshi matching
# ---------------------------------------------------------------------------

def _kalshi_side_matches(sub_title: str, abbrev: str, team_name: str) -> bool:
    sub = (sub_title or "").strip().upper()
    if not sub:
        return False
    if sub == abbrev.upper():
        return True
    return sub in team_name.upper() or team_name.upper().replace(" ", "").endswith(sub)


def _match_kalshi_series(games: pd.DataFrame, target_date: date, abbrev_map: dict[int, str],
                          series_markets: list[dict], market_type: str) -> list[dict]:
    rows = []
    by_event: dict[str, list[dict]] = {}
    for m in series_markets:
        by_event.setdefault(m["event_ticker"], []).append(m)

    for event_ticker, markets in by_event.items():
        occ = markets[0].get("occurrence_datetime")
        if not occ or not _game_date_matches(occ, target_date):
            continue
        for _, g in games.iterrows():
            home_abbr = abbrev_map.get(g["home_team_id"])
            away_abbr = abbrev_map.get(g["away_team_id"])
            if not home_abbr or not away_abbr:
                continue
            side_map = {}
            for m in markets:
                if _kalshi_side_matches(m.get("yes_sub_title"), home_abbr, g["home_team_name"]):
                    side_map["home"] = m
                elif _kalshi_side_matches(m.get("yes_sub_title"), away_abbr, g["away_team_name"]):
                    side_map["away"] = m
            if len(side_map) != 2:
                continue  # couldn't confidently match both sides -- skip, don't guess
            for side, m in side_map.items():
                yes_ask = m.get("yes_ask_dollars")
                if yes_ask is None or pd.isna(yes_ask):
                    continue
                rows.append({
                    "game_pk": g["game_pk"], "market": "kalshi", "market_type": market_type,
                    "side": side, "line": np.nan, "pm_price": float(yes_ask),
                    "pm_implied_prob": float(yes_ask), "_ticker": m["ticker"],
                })
            break  # this event_ticker's teams are claimed; move to the next event
    return rows


def match_kalshi_moneyline(games: pd.DataFrame, target_date: date, abbrev_map: dict[int, str]) -> list[dict]:
    return _match_kalshi_series(games, target_date, abbrev_map,
                                 pmkt.get_kalshi_mlb_moneyline_markets(), "moneyline")


def match_kalshi_totals(games: pd.DataFrame, target_date: date, abbrev_map: dict[int, str]) -> list[dict]:
    return _match_kalshi_series(games, target_date, abbrev_map,
                                 pmkt.get_kalshi_mlb_totals_markets(), "totals")


def add_kalshi_depth(rows: list[dict]) -> None:
    kalshi_rows = [r for r in rows if r["market"] == "kalshi"]
    for i, r in enumerate(kalshi_rows):
        try:
            ob = pmkt.get_kalshi_orderbook(r["_ticker"])["orderbook_fp"]
            opposite = ob.get("no_dollars") or []  # buying YES draws on resting NO bids
            r["depth_usd"] = pmkt.usd_depth_kalshi_buy(opposite, r["pm_price"], SLIPPAGE)
        except Exception as e:
            print(f"[warn] Kalshi orderbook fetch failed for {r['_ticker']}: {e}")
            r["depth_usd"] = np.nan
        if (i + 1) % 10 == 0 or (i + 1) == len(kalshi_rows):
            print(f"  ... Kalshi depth {i + 1}/{len(kalshi_rows)}")


# ---------------------------------------------------------------------------
# Sportsbook consensus (reuses odds_api, same convention as daily_ml.py)
# ---------------------------------------------------------------------------

def sportsbook_fair_probs(games: pd.DataFrame) -> dict[int, dict]:
    """game_pk -> {home_fair_prob, away_fair_prob, n_books}. Away fair prob is
    exactly 1 - home fair prob by construction (see odds_api.h2h_consensus_favorite:
    each book's pair is devigged to sum to 1 before averaging, and averaging
    preserves that sum)."""
    out = {}
    try:
        api_key = odds_api.load_api_key()
        bulk = odds_api.get_bulk_odds(api_key, markets="h2h")
    except Exception as e:
        print(f"[warn] sportsbook consensus pull failed, proceeding without it: {e}")
        return out

    for g_odds in bulk:
        fav = odds_api.h2h_consensus_favorite(g_odds)
        if fav is None:
            continue
        match = games[(games["home_team_name"] == g_odds["home_team"]) & (games["away_team_name"] == g_odds["away_team"])]
        if match.empty:
            continue
        game_pk = int(match.iloc[0]["game_pk"])
        home_team = g_odds["home_team"]
        home_fair = fav["favorite_fair_prob"] if fav["favorite_team"] == home_team else 1 - fav["favorite_fair_prob"]
        out[game_pk] = {"home_fair_prob": home_fair, "away_fair_prob": 1 - home_fair, "n_books": fav["n_books"]}
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(target_date: date, dry_run: bool):
    print(f"=== prediction-market comparison: {target_date.isoformat()} ===")
    todays_games = fetch_todays_games(target_date)
    print(f"games with both probable starters announced: {len(todays_games)}")
    if todays_games.empty:
        print("no games with announced starters yet for this date.")
        return

    feats = build_todays_ml_features(target_date, todays_games)
    scored = score_ml(feats)
    scored["home_win_prob"] = scored["home_win_prob"].astype(float)

    print("\nmodel predictions:")
    with pd.option_context("display.width", 160):
        print(scored[["home_team_name", "away_team_name", "home_win_prob"]]
              .sort_values("home_win_prob", ascending=False).to_string(index=False))

    if dry_run:
        print("\n--dry-run: skipping Polymarket/Kalshi/sportsbook pulls.")
        return

    abbrev_map = _team_abbrev_map(target_date.year)

    rows = []
    rows += match_polymarket_moneyline(scored, target_date)
    rows += match_polymarket_totals(scored, target_date)
    n_polymarket = len(rows)
    rows += match_kalshi_moneyline(scored, target_date, abbrev_map)
    rows += match_kalshi_totals(scored, target_date, abbrev_map)
    print(f"\nmatched rows: {len(rows)} ({n_polymarket} from Polymarket, "
          f"{len(rows) - n_polymarket} from Kalshi)")
    if not rows:
        print("no PM rows matched today's games -- nothing further to do.")
        return

    add_polymarket_depth(rows)
    add_kalshi_depth(rows)

    sb_probs = sportsbook_fair_probs(scored)

    matched = pd.DataFrame(rows)
    game_lookup = scored.set_index("game_pk")[["home_team_name", "away_team_name", "home_win_prob"]]
    matched = matched.join(game_lookup, on="game_pk")

    def model_prob_for_row(r):
        if r["market_type"] != "moneyline":
            return np.nan
        return r["home_win_prob"] if r["side"] == "home" else 1 - r["home_win_prob"]

    matched["model_prob"] = matched.apply(model_prob_for_row, axis=1)
    matched["edge"] = matched["model_prob"] - matched["pm_implied_prob"]

    def sb_prob_for_row(r):
        sb = sb_probs.get(int(r["game_pk"]))
        if sb is None or r["market_type"] != "moneyline":
            return np.nan
        return sb["home_fair_prob"] if r["side"] == "home" else sb["away_fair_prob"]

    matched["sportsbook_prob"] = matched.apply(sb_prob_for_row, axis=1)
    matched["n_books"] = matched["game_pk"].map(lambda pk: sb_probs.get(int(pk), {}).get("n_books"))

    out_cols = ["game_pk", "home_team_name", "away_team_name", "market", "market_type", "side", "line",
                "model_prob", "pm_price", "pm_implied_prob", "edge", "depth_usd", "sportsbook_prob", "n_books"]
    matched[out_cols].to_csv(DAILY_MATCHED_PATH, index=False)
    print(f"daily matched rows written: {DAILY_MATCHED_PATH} ({len(matched)} rows)")

    flagged = matched[(matched["market_type"] == "moneyline") &
                       (matched["edge"] >= EDGE_FLAG_THRESHOLD) &
                       (matched["depth_usd"] >= MIN_DEPTH_USD)].copy()
    print(f"\nflagged (edge >= {EDGE_FLAG_THRESHOLD:.0%}, depth >= ${MIN_DEPTH_USD:.0f}): {len(flagged)}")
    if len(flagged):
        with pd.option_context("display.width", 160):
            print(flagged[["home_team_name", "away_team_name", "market", "side", "model_prob",
                            "pm_implied_prob", "edge", "depth_usd", "sportsbook_prob"]].to_string(index=False))

    ledger_rows = flagged[out_cols].copy()
    ledger_rows.insert(0, "date", target_date.isoformat())
    ledger_rows["logged_at"] = datetime.now().isoformat(timespec="seconds")
    for c in ["actual_result", "correct", "pnl"]:
        ledger_rows[c] = np.nan
    ledger_rows["correct"] = ledger_rows["correct"].astype("object")
    ledger_rows["actual_result"] = ledger_rows["actual_result"].astype("object")

    if LEDGER_PATH.exists():
        existing = pd.read_csv(LEDGER_PATH)
        existing["correct"] = existing["correct"].astype("object")
        existing["actual_result"] = existing["actual_result"].astype("object")
        existing = existing[existing["date"] != target_date.isoformat()]
        ledger_rows = pd.concat([existing, ledger_rows], ignore_index=True)
    ledger_rows.to_csv(LEDGER_PATH, index=False)
    print(f"\nledger updated: {LEDGER_PATH} ({len(ledger_rows)} total rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    ap.add_argument("--dry-run", action="store_true", help="skip Polymarket/Kalshi/sportsbook pulls")
    args = ap.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    run(target, args.dry_run)
