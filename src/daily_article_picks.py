"""
Pure article-based pick generation -- a third, distinct pipeline alongside
daily_ks.py (stats-only) and daily_research_ks.py (stats+research blended).
Where daily_research_ks.py explicitly shows Claude the stats model's numbers
and asks it to weigh research against them, THIS pipeline shows Claude
nothing but articles -- its picks come purely from what it reads, with no
stats-model number anywhere in the prompt. The two ledgers this produces are
a genuinely independent epistemic input, not a hybrid of the other two.

Two-stage design, built specifically to keep cost low -- most of the slate
gets ZERO research spend:

Stage 1 (free, no LLM, no new API calls): rank today's games by the size of
the K-props edge daily_ks.py already flagged (reusing ks_daily_matched.csv,
same as daily_research_ks.py does), take the top TOP_N_GAMES. This is a
cheap, already-computed signal -- the K-props model is the one component in
this repo with a real, backtested edge over naive baselines (CLAUDE.md,
Track 2), so it's the most defensible filter for "which games are worth
paying for research on."

Stage 2 (the only real spend): for each selected game, Claude searches up to
MAX_SEARCHES_PER_GAME articles/previews about both starting pitchers and
both teams (web_search tool, max_uses caps the search count so cost per game
has a hard ceiling), then produces ONE strikeout-prop pick (over/under on the
SAME specific market line Stage 1 flagged -- keeps the comparison to the
stats model apples-to-apples: same pitcher, same line, independent read) and
ONE moneyline pick (straight-up winner), each with a confidence level and a
reasoning summary that cites specific things it found. One Claude call per
game covers both picks (they're informed by the same body of research), not
two -- avoids redundant search spend.

Logs to two separate ledgers with full reconciliation (unlike the
stats-only moneyline model, which deliberately has no betting logic --
see daily_ml.py's docstring): output/article_picks_ks_ledger.csv and
output/article_picks_ml_ledger.csv. Both use the same 1-unit-stake,
CLV-from-captured-closing-price convention as the K-props ledger.

Usage:
    python src/daily_article_picks.py                  # live: needs ANTHROPIC_API_KEY + ODDS_API_KEY
    python src/daily_article_picks.py --dry-run         # stage 1 only, no Claude/odds calls
    python src/daily_article_picks.py --date 2026-07-20 # override "today"
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fetch
import odds_api
from research_agents import load_anthropic_api_key

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
DAILY_MATCHED_PATH = OUTPUT / "ks_daily_matched.csv"  # written by daily_ks.py -- read-only here
ML_PREDICTIONS_PATH = OUTPUT / "ml_predictions_ledger.csv"  # written by daily_ml.py -- read-only here
KS_LEDGER_PATH = OUTPUT / "article_picks_ks_ledger.csv"
ML_LEDGER_PATH = OUTPUT / "article_picks_ml_ledger.csv"

TOP_N_GAMES = 4
MAX_SEARCHES_PER_GAME = 5
RESEARCH_MODEL = "claude-sonnet-5"
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"


# --------------------------------------------------------------------------- #
# Stage 1: cheap selection, no LLM, no new API calls
# --------------------------------------------------------------------------- #

def select_top_games(target_date: date, top_n: int = TOP_N_GAMES) -> pd.DataFrame:
    """Rank today's games by the largest K-props edge daily_ks.py already
    computed (reusing ks_daily_matched.csv -- zero extra cost), return one row
    per selected game: the specific (pitcher, line) that drove the edge."""
    if not DAILY_MATCHED_PATH.exists():
        raise FileNotFoundError(f"{DAILY_MATCHED_PATH} not found -- daily_ks.py must run first.")

    matched = pd.read_csv(DAILY_MATCHED_PATH)
    todays = matched[matched["date"] == target_date.isoformat()].copy()
    if todays.empty:
        return todays

    todays["over_edge"] = todays["model_p_over"] - todays["over_prob_fair"]
    todays["under_edge"] = (1 - todays["model_p_over"]) - (1 - todays["over_prob_fair"])
    todays["edge"] = np.maximum(todays["over_edge"], todays["under_edge"])
    todays["side"] = np.where(todays["over_edge"] >= todays["under_edge"], "over", "under")

    # One row per game_pk: whichever matched pitcher/line had the biggest edge.
    top_per_game = todays.loc[todays.groupby("game_pk")["edge"].idxmax()]
    return top_per_game.sort_values("edge", ascending=False).head(top_n)


def _fetch_todays_team_names(target_date: date) -> dict[int, tuple[str, str]]:
    """game_pk -> (home_team_name, away_team_name) for today's slate, independent
    of whether probable pitchers are announced for both sides yet."""
    d = fetch._get(f"{fetch.STATSAPI_BASE}/schedule", params={
        "sportId": 1, "date": target_date.isoformat(), "gameType": "R",
    })
    out = {}
    for date_block in d.get("dates", []):
        for g in date_block.get("games", []):
            home, away = g["teams"]["home"], g["teams"]["away"]
            out[g["gamePk"]] = (home["team"]["name"], away["team"]["name"])
    return out


# --------------------------------------------------------------------------- #
# Stage 2: the only real spend -- one Claude call per selected game
# --------------------------------------------------------------------------- #

def build_prompt(row: pd.Series, home_team: str, away_team: str, target_date: date) -> str:
    return f"""You are handicapping tonight's MLB game between {away_team} and {home_team} ({target_date.isoformat()}), using ONLY what you find by researching -- do not rely on your own general knowledge of these teams' season-long tendencies unless you find it confirmed in what you read tonight.

Search for up to {MAX_SEARCHES_PER_GAME} recent articles or previews covering:
- {row['name']} (probable starter, one side of this game) -- recent form, health, role.
- The opposing starting pitcher for this game (search to confirm who it is if not already obvious).
- Both teams -- lineup news, injuries, any matchup-specific angles beat writers are flagging for tonight.

Then do two things:

First, summarize what the articles COLLECTIVELY say about this matchup -- the shared narrative across what you read, and which way it points for tonight's strikeout total specifically. For example: "Articles emphasize the Twins' recent contact-heavy approach and Bibee's dip in velocity -- consensus leans toward fewer strikeouts."

Then, based ONLY on what you found, make two independent picks:
1. Strikeout prop for {row['name']}: the market line tonight is {row['line']}. Pick over or under.
2. Moneyline: which team wins tonight, {home_team} (home) or {away_team} (away)?

Respond with ONLY a JSON object in exactly this shape, no other text before or after it:
{{"article_consensus": "<1-2 sentences on what the articles collectively say about this matchup and which way they lean on tonight's strikeout total>", "ks_pick": {{"side": "over" or "under", "confidence": "low" or "medium" or "high", "reasoning": "<2-4 sentences citing specific things you found -- name the source/finding, not just \\"reports suggest\\">"}}, "ml_pick": {{"team": "home" or "away", "confidence": "low" or "medium" or "high", "reasoning": "<2-4 sentences citing specific things you found>"}}}}"""


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def load_stats_ml_sides(target_date: date) -> dict[int, str]:
    """game_pk -> 'home'/'away', the stats moneyline model's lean for each of
    today's games (daily_ml.py's ml_predictions_ledger.csv). Empty dict if that
    file doesn't exist or has no rows for today -- the alignment comparison just
    degrades to 'no stats ML pick available' for those games, never crashes.

    Requires daily_ml.py to have run before this pipeline (see morning-pull.yml's
    step order); if it hasn't, moneyline alignment simply isn't computed."""
    if not ML_PREDICTIONS_PATH.exists():
        return {}
    ml = pd.read_csv(ML_PREDICTIONS_PATH)
    todays = ml[ml["date"] == target_date.isoformat()]
    out: dict[int, str] = {}
    for _, r in todays.iterrows():
        side = "home" if r["predicted_winner"] == r["home_team_name"] else "away"
        out[int(r["game_pk"])] = side
    return out


def research_game(client, row: pd.Series, home_team: str, away_team: str, target_date: date) -> dict | None:
    prompt = build_prompt(row, home_team, away_team, target_date)
    try:
        response = client.messages.create(
            model=RESEARCH_MODEL,
            max_tokens=1500,
            tools=[{"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": MAX_SEARCHES_PER_GAME}],
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"  [warn] research call failed for {away_team} @ {home_team}: {e}")
        return None

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        print(f"  [warn] no text response for {away_team} @ {home_team} (stop_reason={response.stop_reason})")
        return None

    parsed = _extract_json(text_blocks[-1])
    if parsed is None or "ks_pick" not in parsed or "ml_pick" not in parsed:
        print(f"  [warn] could not parse response for {away_team} @ {home_team}: {text_blocks[-1][:200]}")
        return None
    parsed.setdefault("article_consensus", "")  # tolerate an older-shaped response missing the field
    return parsed


# --------------------------------------------------------------------------- #
# Real current odds for both picks -- needed for PnL/CLV, pulled once for the
# whole selected slate (free bulk h2h call + the K-props line already have in
# hand from Stage 1).
# --------------------------------------------------------------------------- #

def _representative_h2h_price(bulk_odds: list[dict], home_team: str, away_team: str, side: str) -> float | None:
    """Median American price across books, aggregated in decimal-odds space --
    same fix as odds_api.consensus_over_under (raw American odds jump from -100
    to +100 with nothing between, so a naive median can land inside that gap)."""
    from fetch_odds import american_to_decimal, decimal_to_american

    team_name = home_team if side == "home" else away_team
    for g in bulk_odds:
        if g.get("home_team") != home_team or g.get("away_team") != away_team:
            continue
        prices = []
        for book in g.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market["outcomes"]:
                    if outcome["name"] == team_name:
                        prices.append(american_to_decimal(outcome["price"]))
        if prices:
            return decimal_to_american(float(np.median(prices)))
    return None


def run(target_date: date, dry_run: bool):
    print(f"=== article-based picks: {target_date.isoformat()} ===")

    try:
        selected = select_top_games(target_date, TOP_N_GAMES)
    except FileNotFoundError as e:
        print(str(e))
        return

    if selected.empty:
        print(f"No matched K-props edges for {target_date.isoformat()} to select games from "
              f"-- run daily_ks.py for this date first.")
        return

    team_names = _fetch_todays_team_names(target_date)
    print(f"selected {len(selected)} game(s) by K-props edge size:")
    for _, row in selected.iterrows():
        home, away = team_names.get(row["game_pk"], ("?", "?"))
        print(f"  {away} @ {home}  ({row['name']} {row['side']} {row['line']}, edge {row['edge']:+.1%})")

    if dry_run:
        print(f"\n--dry-run: would make {len(selected)} research call(s) to {RESEARCH_MODEL} "
              f"({MAX_SEARCHES_PER_GAME} searches each max), no API calls made.")
        return

    import anthropic
    try:
        anthropic_key = load_anthropic_api_key()
    except RuntimeError:
        # Clean, actionable one-liner instead of a raw traceback -- with
        # continue-on-error the workflow shows green either way, so this is what
        # makes "no picks produced" legible in the Actions log.
        print("ANTHROPIC_API_KEY not visible to this process (no env var, none in .env). "
              "In GitHub Actions this means the repo Actions secret 'ANTHROPIC_API_KEY' is unset, "
              "misnamed, or scoped to an environment this job doesn't use. Skipping article picks.")
        return
    client = anthropic.Anthropic(api_key=anthropic_key)

    api_key = odds_api.load_api_key()
    bulk_h2h = odds_api.get_bulk_odds(api_key, markets="h2h")  # free, 1 unit total for the whole slate
    stats_ml_sides = load_stats_ml_sides(target_date)  # game_pk -> 'home'/'away' from daily_ml.py

    ks_results, ml_results = [], []
    for i, (_, row) in enumerate(selected.iterrows()):
        home_team, away_team = team_names.get(row["game_pk"], (None, None))
        if home_team is None:
            print(f"  [warn] no team names found for game_pk {row['game_pk']}, skipping")
            continue

        picks = research_game(client, row, home_team, away_team, target_date)
        if picks is None:
            continue

        # --- K-props alignment: article's over/under vs the stats model's own lean
        # on the exact same (pitcher, line). stats_model_side is already in the row.
        # Stored as an explicit string ("aligned"/"contrarian") rather than a bool
        # -- CSV round-trips a bool column with any blanks into an object mix of
        # True/False/NaN where `"False"` is truthy, a well-known footgun. A string
        # is unambiguous on read and directly filterable (aligned vs contrarian ROI). ---
        ks_side = picks["ks_pick"]["side"]
        ks_stats_side = row["side"]
        ks_alignment = "aligned" if ks_side == ks_stats_side else "contrarian"

        # --- Moneyline alignment: article's home/away vs the stats moneyline model's
        # predicted winner for this game_pk. "" if daily_ml.py hasn't run / no row. ---
        ml_side = picks["ml_pick"]["team"]
        ml_stats_side = stats_ml_sides.get(int(row["game_pk"]))
        ml_alignment = "" if ml_stats_side is None else ("aligned" if ml_side == ml_stats_side else "contrarian")

        consensus = picks.get("article_consensus", "")
        print(f"  [{i + 1}/{len(selected)}] {away_team} @ {home_team}: "
              f"KS={ks_side} vs model {ks_stats_side} [{ks_alignment}]  "
              f"ML={ml_side} vs model {ml_stats_side or 'n/a'} [{ml_alignment or 'no model pick'}]")

        # Store BOTH sides' prices at pick time, not just the picked side -- CLV
        # needs a proper devigged fair probability at both bet-time and closing-
        # time (reconcile_article_picks_ks.py reuses reconcile_ks.py's
        # _devig_pair), which requires both prices, same convention as every
        # other ledger in this repo (see fetch_odds.american_to_decimal's
        # docstring on why a single side's raw price isn't enough).
        ks_results.append({
            "date": target_date.isoformat(), "mlbID": row["mlbID"], "game_pk": row["game_pk"],
            "event_id": row.get("event_id"), "name": row["name"], "opponent_name": row["opponent_name"],
            "line": row["line"], "bet_side": ks_side, "over_odds": row["over_odds"],
            "under_odds": row["under_odds"], "over_prob_fair": row["over_prob_fair"],
            "confidence": picks["ks_pick"]["confidence"], "reasoning": picks["ks_pick"]["reasoning"],
            "article_consensus": consensus, "stats_model_side": ks_stats_side,
            "alignment": ks_alignment, "stats_model_edge": row["edge"],
            "logged_at": datetime.now().isoformat(timespec="seconds"),
        })

        home_price = _representative_h2h_price(bulk_h2h, home_team, away_team, "home")
        away_price = _representative_h2h_price(bulk_h2h, home_team, away_team, "away")
        ml_results.append({
            "date": target_date.isoformat(), "game_pk": row["game_pk"], "home_team_name": home_team,
            "away_team_name": away_team, "bet_side": ml_side, "home_odds": home_price,
            "away_odds": away_price, "confidence": picks["ml_pick"]["confidence"],
            "reasoning": picks["ml_pick"]["reasoning"], "article_consensus": consensus,
            "stats_model_side": ml_stats_side if ml_stats_side is not None else "",
            "alignment": ml_alignment, "logged_at": datetime.now().isoformat(timespec="seconds"),
        })

    _write_ledger(KS_LEDGER_PATH, ks_results, target_date, extra_blank_cols=[
        "closing_over_odds", "closing_under_odds", "actual_so", "result", "pnl", "clv"])
    _write_ledger(ML_LEDGER_PATH, ml_results, target_date, extra_blank_cols=[
        "closing_home_odds", "closing_away_odds", "actual_winner", "result", "pnl", "clv"])


def _write_ledger(path: Path, rows: list[dict], target_date: date, extra_blank_cols: list[str]):
    if not rows:
        print(f"\nno rows produced for {path.name}.")
        return
    new_rows = pd.DataFrame(rows)
    for c in extra_blank_cols:
        new_rows[c] = np.nan
    new_rows["result"] = pd.array([None] * len(new_rows), dtype="object")

    if path.exists():
        existing = pd.read_csv(path)
        existing["result"] = existing["result"].astype("object")
        existing = existing[existing["date"] != target_date.isoformat()]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(path, index=False)
    print(f"ledger updated: {path} ({len(combined)} total rows, {len(new_rows)} new)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    args = ap.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    run(target, args.dry_run)
