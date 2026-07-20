"""
Research-model K-props pipeline -- runs PARALLEL to the stats-only model
(daily_ks.py), never modifies it. Where daily_ks.py projects strikeouts purely
from rolling/season-lag features via a fitted NB2 regression, this pipeline
asks Claude to review the SAME stats alongside live web-searched news (recent
articles on the starting pitcher and the opposing lineup) and produce its own
projection + reasoning. The point is a real head-to-head test: does folding in
qualitative research beat stats alone, tracked on identical terms (same
distributional math, same market lines, same reconciliation/CLV process) in a
separate paper ledger -- output/research_ks_ledger.csv.

Reuses daily_ks.py's already-fetched market data (output/ks_daily_matched.csv)
instead of re-pulling odds, so this costs ZERO extra Odds API quota. Requires
daily_ks.py to have already run today (its matched file must exist).

Methodology note: this pipeline's P(over) is computed by feeding Claude's
projected mu into the SAME NB2 distribution (model_ks.prob_over) using the
STATS model's own fitted dispersion parameter (alpha) -- not a separately
fitted distribution. This keeps the comparison to daily_ks.py apples-to-apples:
the only thing that differs between the two ledgers is where mu comes from
(regression vs. research-informed Claude call), not the probability math
layered on top of it.

Cost: one Claude Sonnet 5 call per matched pitcher (typically 15-20/day, same
count as daily_ks.py's odds-matched pitchers), each with the web_search tool
enabled. See the module's own cost estimate in the project chat history --
default model is Sonnet 5 for reasoning quality; swap RESEARCH_MODEL to
"claude-haiku-4-5" for a cheaper, lower-quality run.

Usage:
    python src/daily_research_ks.py                  # live: needs ANTHROPIC_API_KEY
    python src/daily_research_ks.py --dry-run         # skip Claude calls, print what would run
    python src/daily_research_ks.py --date 2026-07-20 # override "today"
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from model_ks import nb2_np
from research_agents import load_anthropic_api_key

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
DAILY_MATCHED_PATH = OUTPUT / "ks_daily_matched.csv"  # written by daily_ks.py -- read-only here
RESEARCH_DAILY_PATH = OUTPUT / "research_ks_daily.csv"  # every researched pitcher, overwritten each run
LEDGER_PATH = OUTPUT / "research_ks_ledger.csv"

EDGE_FLAG_THRESHOLD = 0.03
RESEARCH_MODEL = "claude-sonnet-5"  # swap to "claude-haiku-4-5" for a cheaper/lower-quality run
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"  # Sonnet 5 supports the dynamic-filtering variant


def build_prompt(row: pd.Series, target_date: date) -> str:
    return f"""You are evaluating tonight's MLB strikeout prop for starting pitcher {row['name']}, facing {row['opponent_name']}, on {target_date.isoformat()}.

Search for recent news and articles (the last few days, and any beat-writer or injury-report coverage from today) about:
1. {row['name']} -- his last couple starts, any injury/health notes, reported velocity or stuff changes, workload/pitch-count management, role changes.
2. {row['opponent_name']}'s lineup for tonight -- confirmed or expected lineup, notable absences or replacements, how this lineup has performed against pitchers with a similar profile recently.

For reference, here is what a stats-only model already computed for this matchup. Do not just repeat these numbers back -- weigh them against whatever you find in your research, and be explicit about agreement or disagreement:
- Trailing K/9 (last 3 starts): {row.get('trail_k_per9_3s', float('nan')):.1f}
- Trailing K/9 (last 30 days): {row.get('trail_k_per9_30d', float('nan')):.1f}
- Season whiff rate: {row.get('season_lag_whiff_pct', float('nan')):.1f}%
- Opponent strikeout rate vs. his throwing hand: {row.get('opp_off_kpct', float('nan')):.1%}
- Trailing innings per start: {row.get('trail_ip_per_start', float('nan')):.1f}
- Days rest: {row.get('days_rest', float('nan')):.0f}
- Stats-only model's own projection: {row.get('mu', float('nan')):.1f} strikeouts

Tonight's market strikeout prop line is {row['line']}.

After researching, respond with ONLY a JSON object in exactly this shape, no other text before or after it:
{{"projected_strikeouts": <number>, "reasoning": "<2-4 sentences citing specific things you found -- name the finding, not just \\"reports suggest\\" -- and stating explicitly how it squares with or contradicts the stats above. If you found nothing material, say so plainly instead of padding.>"}}"""


def prob_over_per_row(mu: np.ndarray, alpha: float, line: np.ndarray) -> np.ndarray:
    """Same NB2 math as model_ks.prob_over, reimplemented here only because that
    function's `int(np.floor(line))` assumes a single scalar line applied to every
    row at once (which is how daily_ks.py always calls it -- one of the fixed
    PROP_LINES constants for the whole array). This pipeline has a genuinely
    different line per pitcher, so `line` needs to vectorize -- np.floor(...).astype(int)
    does that correctly where int(np.floor(...)) raises on a multi-element array."""
    n, p = nb2_np(mu, alpha)
    threshold = np.floor(line).astype(int)  # P(K > 4.5) = P(K >= 5) = 1 - CDF(4)
    return 1 - scipy_stats.nbinom.cdf(threshold, n, p)


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def research_pitcher(client, row: pd.Series, target_date: date) -> dict | None:
    prompt = build_prompt(row, target_date)
    try:
        response = client.messages.create(
            model=RESEARCH_MODEL,
            max_tokens=1500,
            tools=[{"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"  [warn] research call failed for {row['name']}: {e}")
        return None

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        print(f"  [warn] no text response for {row['name']} (stop_reason={response.stop_reason})")
        return None

    parsed = _extract_json(text_blocks[-1])
    if parsed is None or "projected_strikeouts" not in parsed or "reasoning" not in parsed:
        print(f"  [warn] could not parse research response for {row['name']}: {text_blocks[-1][:200]}")
        return None

    return {"projected_strikeouts": float(parsed["projected_strikeouts"]), "reasoning": str(parsed["reasoning"])}


def run(target_date: date, dry_run: bool):
    print(f"=== research-model K props: {target_date.isoformat()} ===")

    if not DAILY_MATCHED_PATH.exists():
        print(f"No {DAILY_MATCHED_PATH} found -- daily_ks.py must run first (this pipeline reuses "
              f"its already-fetched market data instead of re-pulling odds).")
        return

    matched = pd.read_csv(DAILY_MATCHED_PATH)
    todays = matched[matched["date"] == target_date.isoformat()].copy()
    if todays.empty:
        print(f"No matched pitchers for {target_date.isoformat()} in {DAILY_MATCHED_PATH} "
              f"-- run daily_ks.py for this date first.")
        return
    print(f"pitchers with matched market lines: {len(todays)}")

    if dry_run:
        unique_pitchers = todays.drop_duplicates(subset=["mlbID"])
        print(f"\n--dry-run: would make {len(unique_pitchers)} research call(s) to {RESEARCH_MODEL} "
              f"(deduped from {len(todays)} matched lines), no API calls made.")
        for _, row in unique_pitchers.iterrows():
            lines_for_pitcher = todays[todays["mlbID"] == row["mlbID"]]["line"].tolist()
            print(f"  {row['name']:25s} vs {row['opponent_name']:25s} line(s) {lines_for_pitcher}")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=load_anthropic_api_key())

    saved = joblib.load(OUTPUT / "model_ks.joblib")
    alpha = saved["alpha"]  # reuse the stats model's fitted dispersion -- see module docstring

    # Research each unique pitcher ONCE, not once per matched line -- a pitcher can have
    # multiple lines offered (e.g. 4.5 and 5.5), and the research itself (pitcher/opponent
    # news) doesn't change with the line, so re-researching per line would waste real
    # Claude + web-search cost on near-identical calls for no benefit.
    unique_pitchers = todays.drop_duplicates(subset=["mlbID"])
    print(f"unique pitchers to research: {len(unique_pitchers)} (from {len(todays)} matched lines)")

    research_by_pitcher: dict = {}
    for i, (_, row) in enumerate(unique_pitchers.iterrows()):
        research = research_pitcher(client, row, target_date)
        if research is None:
            continue
        research_by_pitcher[row["mlbID"]] = research
        print(f"  [{i + 1}/{len(unique_pitchers)}] {row['name']:25s} research mu={research['projected_strikeouts']:.1f} "
              f"(stats mu={row.get('mu', float('nan')):.1f})")

    if not research_by_pitcher:
        print("\nNo research results produced -- nothing to score.")
        return

    # Apply each pitcher's single research result across all of that pitcher's matched lines.
    results = []
    for _, row in todays.iterrows():
        research = research_by_pitcher.get(row["mlbID"])
        if research is None:
            continue
        results.append({**row.to_dict(), **research})

    scored = pd.DataFrame(results)
    scored["research_p_over"] = prob_over_per_row(scored["projected_strikeouts"].values, alpha, scored["line"].values)
    scored["edge"] = scored["research_p_over"] - scored["over_prob_fair"]
    scored["under_edge"] = (1 - scored["research_p_over"]) - (1 - scored["over_prob_fair"])
    scored["bet_side"] = np.where(scored["edge"] >= scored["under_edge"], "over", "under")
    scored["bet_edge"] = np.where(scored["bet_side"] == "over", scored["edge"], scored["under_edge"])

    daily_cols = ["mlbID", "name", "opponent_name", "line", "projected_strikeouts", "research_p_over",
                  "over_prob_fair", "bet_side", "bet_edge", "reasoning"]
    daily_out = scored[daily_cols].copy()
    daily_out.insert(0, "date", target_date.isoformat())
    daily_out.to_csv(RESEARCH_DAILY_PATH, index=False)
    print(f"\nresearch predictions written: {RESEARCH_DAILY_PATH} ({len(daily_out)} rows)")

    flagged = scored[scored["bet_edge"] >= EDGE_FLAG_THRESHOLD].copy()
    print(f"flagged (edge >= {EDGE_FLAG_THRESHOLD:.0%}): {len(flagged)}")
    if flagged.empty:
        return

    ledger_rows = flagged[["mlbID", "game_pk", "event_id", "name", "opponent_name", "line", "bet_side",
                            "bet_edge", "research_p_over", "over_prob_fair", "over_odds", "under_odds",
                            "n_books", "projected_strikeouts", "reasoning"]].copy()
    ledger_rows = ledger_rows.rename(columns={"research_p_over": "model_p_over"})
    ledger_rows.insert(0, "date", target_date.isoformat())
    ledger_rows["logged_at"] = datetime.now().isoformat(timespec="seconds")
    for c in ["closing_over_odds", "closing_under_odds", "actual_so", "pnl", "clv"]:
        ledger_rows[c] = np.nan
    ledger_rows["result"] = pd.array([None] * len(ledger_rows), dtype="object")

    if LEDGER_PATH.exists():
        existing = pd.read_csv(LEDGER_PATH)
        existing["result"] = existing["result"].astype("object")
        existing = existing[existing["date"] != target_date.isoformat()]
        ledger_rows = pd.concat([existing, ledger_rows], ignore_index=True)
    ledger_rows.to_csv(LEDGER_PATH, index=False)
    print(f"ledger updated: {LEDGER_PATH} ({len(ledger_rows)} total rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    args = ap.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    run(target, args.dry_run)
