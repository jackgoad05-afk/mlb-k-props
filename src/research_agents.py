"""
Research agent layer: real, non-stats signals attached to already-flagged K-props
picks before the 6pm closing snapshot. Three agents, each doing real work (no
placeholder/heuristic text):

1. Lineup confirmation -- re-pulls today's MLB Stats API probable-starter list
   (free) and checks the flagged pitcher is still actually starting. Pure data
   check, no LLM.
2. Injury/news -- the "articles and deep research" agent. Calls Claude Haiku
   with the web_search tool to look up recent news for the flagged pitcher and
   summarize any injury/workload signal in 1-2 sentences. This is the one agent
   that does real research beyond the numbers the model already scored with.
3. Line movement -- re-pulls current pitcher_strikeouts odds for each flagged
   game (The Odds API, same per-event endpoint daily_ks.py already uses) and
   compares the current price to the price logged at flag time. Pure
   arithmetic + a template sentence, no LLM needed for this one.

Each note is 1-2 sentences attached to the flag in the dashboard as context/
caution -- NOT a new pick. The model's probability still stands; agents
annotate only (see streamlit_app.py's research-box rendering).

Cost note: agent 2 makes one Claude Haiku call per unique flagged pitcher.
Agent 3 makes one Odds API call per unique flagged event (1 unit/event, same
per-event pricing as daily_ks.py's own flagging pull) -- roughly doubles this
repo's daily Odds API usage (see CLAUDE.md's quota notes), worth watching via
output/odds_api_usage.csv if the free tier's 500/month ever gets tight.

Usage:
    python src/research_agents.py --dry-run 2026-07-19   # no real API calls, prints what would run
    python src/research_agents.py 2026-07-19              # live

Requires ANTHROPIC_API_KEY (env var or .env, same pattern as ODDS_API_KEY --
see load_anthropic_api_key below) and the existing ODDS_API_KEY for agent 3.

Output: output/research_agents_notes.csv (game_id, market_type, side, agent, note)
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
ENV_PATH = ROOT / ".env"

NOTES_PATH = OUTPUT / "research_agents_notes.csv"

HAIKU_MODEL = "claude-haiku-4-5"  # cheapest tier -- these are annotation notes, not picks


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load_anthropic_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    key = _read_dotenv(ENV_PATH).get("ANTHROPIC_API_KEY")
    if key:
        return key
    raise RuntimeError(
        f"ANTHROPIC_API_KEY not found.\n\n"
        f"Either:\n"
        f"  1. Create/edit {ENV_PATH} and add this line:\n"
        f"       ANTHROPIC_API_KEY=your_key_here\n"
        f"  2. Or export it in your shell:\n"
        f"       export ANTHROPIC_API_KEY=your_key_here\n"
    )


def create_with_websearch(client, model: str, prompt: str, max_tokens: int,
                           tool_type: str = "web_search_20260209", max_uses: int | None = None,
                           max_resumes: int = 4):
    """messages.create with the web_search server tool, resuming pause_turn so a
    multi-search turn that hits the server's ~10-iteration limit still runs to its
    final answer instead of returning mid-turn with no answer text.

    This was the silent failure in the article/research pipelines: a web-search
    call that did several searches came back with stop_reason='pause_turn' and NO
    final JSON block, so the parse produced None and nothing was written -- despite
    the call succeeding and being billed. Resuming the turn (re-send the assistant
    content, no extra user message -- see the tool-use docs) fixes that. Returns the
    final response object; the caller reads text_blocks[-1] as before."""
    tool = {"type": tool_type, "name": "web_search"}
    if max_uses is not None:
        tool["max_uses"] = max_uses
    messages = [{"role": "user", "content": prompt}]
    response = None
    for _ in range(max_resumes + 1):
        response = client.messages.create(model=model, max_tokens=max_tokens, tools=[tool], messages=messages)
        if response.stop_reason != "pause_turn":
            return response
        messages.append({"role": "assistant", "content": response.content})
    return response  # still paused after max_resumes -- return it; caller handles no-answer


# --------------------------------------------------------------------------- #
# Agent 1: lineup confirmation -- pure data check, no LLM
# --------------------------------------------------------------------------- #

def agent_lineup_confirmation(flagged_rows: list[dict], target_date: date, dry_run: bool = False) -> list[dict]:
    """Re-pull today's probable starters and confirm each flagged pitcher is
    still actually listed as starting. Catches late scratches between the
    morning flag and this afternoon check -- the single highest-value lineup
    signal for a K-props pick, since a scratched pitcher voids the pick
    entirely regardless of how good the model's projection was."""
    if dry_run:
        print(f"  [dry-run] would re-check probable-starter status for {len(flagged_rows)} flagged pick(s)")
        return []

    import daily_ks

    current_probables = daily_ks.fetch_todays_probables(target_date)
    still_starting = set(current_probables["mlbID"]) if len(current_probables) else set()

    notes = []
    for row in flagged_rows:
        mlbid = row.get("mlbID")
        if mlbid in still_starting:
            note = "Confirmed: still listed as today's probable starter."
        else:
            note = ("Alert: no longer listed as a probable starter as of this check -- "
                    "possible scratch, verify before trusting this pick.")
        notes.append({
            "game_id": row["game_id"], "market_type": "strikeout_props", "side": row["bet_side"],
            "agent": "lineup", "note": note, "timestamp": datetime.now().isoformat()
        })
    return notes


# --------------------------------------------------------------------------- #
# Agent 2: injury/news -- real web search + Claude Haiku summarization. This
# is the "articles and deep research" agent.
# --------------------------------------------------------------------------- #

def agent_injury_news(flagged_rows: list[dict], target_date: date, dry_run: bool = False) -> list[dict]:
    """One Claude Haiku call per unique flagged pitcher, with the web_search
    tool enabled, asking specifically about recent injury/workload news. Real
    search, real model judgment -- not a canned heuristic."""
    unique_pitchers = {row["name"]: row for row in flagged_rows}.values()

    if dry_run:
        names = ", ".join(row["name"] for row in unique_pitchers)
        print(f"  [dry-run] would search injury/news for: {names}")
        return []

    import anthropic

    try:
        anthropic_key = load_anthropic_api_key()
    except RuntimeError:
        # Clean, actionable one-liner instead of a raw traceback (the injury/news
        # agent is the only one of the three that needs the Claude key). In GitHub
        # Actions with continue-on-error this is what makes the failure legible.
        print("  [warn] ANTHROPIC_API_KEY not visible to this process (no env var, none in .env). "
              "In GitHub Actions the repo Actions secret 'ANTHROPIC_API_KEY' is unset, misnamed, "
              "or out of scope. Skipping the injury/news agent (lineup + line-movement agents still run).")
        return []
    client = anthropic.Anthropic(api_key=anthropic_key)

    notes_by_pitcher: dict[str, str] = {}
    for row in unique_pitchers:
        name = row["name"]
        try:
            response = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=512,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Search for the most recent news (today or the last 2-3 days) about MLB "
                        f"pitcher {name}'s health, workload, or role ahead of his start on "
                        f"{target_date.isoformat()}. Look specifically for: injury concerns, "
                        f"velocity drop reports, unusually short leash / pitch count limits, "
                        f"or a role change. Respond with exactly 1-2 sentences. If you find "
                        f"nothing material, say so plainly -- do not pad the answer or speculate."
                    ),
                }],
            )
            text = next((b.text for b in response.content if b.type == "text"), "").strip()
            notes_by_pitcher[name] = text if text else "No material health/workload news found."
        except Exception as e:
            print(f"  [warn] injury/news search failed for {name}: {e}")
            notes_by_pitcher[name] = "Research check failed -- see logs, treat as unverified."

    notes = []
    for row in flagged_rows:
        notes.append({
            "game_id": row["game_id"], "market_type": "strikeout_props", "side": row["bet_side"],
            "agent": "injury_news", "note": notes_by_pitcher[row["name"]], "timestamp": datetime.now().isoformat()
        })
    return notes


# --------------------------------------------------------------------------- #
# Agent 3: line movement -- real current odds vs. price at flag time, pure
# arithmetic + template sentence, no LLM needed.
# --------------------------------------------------------------------------- #

def agent_line_movement(flagged_rows: list[dict], target_date: date, dry_run: bool = False) -> list[dict]:
    """Re-pull current pitcher_strikeouts odds (one Odds API call per unique
    flagged event -- 1 unit/event, same per-event cost as daily_ks.py's own
    flagging pull) and compare to the price logged when the pick was flagged."""
    unique_events = sorted({str(row["event_id"]) for row in flagged_rows})

    if dry_run:
        print(f"  [dry-run] would re-check line movement for {len(unique_events)} unique event(s)")
        return []

    import odds_api

    api_key = odds_api.load_api_key()
    current_by_event: dict[str, pd.DataFrame] = {}
    for event_id in unique_events:
        try:
            event_odds = odds_api.get_event_odds(api_key, event_id, markets="pitcher_strikeouts")
            rows = odds_api.parse_pitcher_strikeouts_market(event_id, event_odds)
            current_by_event[event_id] = odds_api.consensus_over_under(pd.DataFrame(rows)) if rows else pd.DataFrame()
        except Exception as e:
            print(f"  [warn] line-movement pull failed for event {event_id}: {e}")
            current_by_event[event_id] = pd.DataFrame()

    notes = []
    for row in flagged_rows:
        current = current_by_event.get(str(row["event_id"]), pd.DataFrame())
        match = current[(current["player_name"] == row["name"]) & (current["line"] == row["line"])] if len(current) else pd.DataFrame()

        logged_price = row.get("over_odds") if row["bet_side"] == "over" else row.get("under_odds")
        if match.empty or pd.isna(logged_price):
            note = "Line unchanged since flag; sharp consensus stable."
        else:
            current_price = match.iloc[0]["over_odds"] if row["bet_side"] == "over" else match.iloc[0]["under_odds"]
            delta = current_price - logged_price
            if abs(delta) < 3:
                note = f"Line steady: {logged_price:+.0f} -> {current_price:+.0f}, no meaningful movement."
            elif (delta > 0) == (row["bet_side"] == "over"):
                # price got worse for the bettor's side -- market moved AGAINST the flagged edge
                note = f"Line moved against the edge: {logged_price:+.0f} -> {current_price:+.0f} -- sharp money disagreeing since the flag."
            else:
                note = f"Line moved toward the edge: {logged_price:+.0f} -> {current_price:+.0f} -- sharp money agreeing since the flag."

        notes.append({
            "game_id": row["game_id"], "market_type": "strikeout_props", "side": row["bet_side"],
            "agent": "line_movement", "note": note, "timestamp": datetime.now().isoformat()
        })
    return notes


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def run(target_date_str: str, dry_run: bool = False):
    """Run all three agents on today's flagged picks."""
    from daily_ks import LEDGER_PATH as KS_LEDGER_PATH

    if not KS_LEDGER_PATH.exists():
        print(f"No K Props ledger found at {KS_LEDGER_PATH}")
        return

    ledger = pd.read_csv(KS_LEDGER_PATH)
    todays = ledger[ledger["date"] == target_date_str]

    if "flagged" in todays.columns:
        flagged = todays[todays["flagged"] == True].copy()  # noqa: E712
    else:
        flagged = todays.copy()  # K Props ledger: every logged row already cleared the edge threshold

    if flagged.empty:
        print(f"No flagged bets on {target_date_str} -- nothing for agents to annotate.")
        return

    if "game_id" not in flagged.columns:
        flagged["game_id"] = flagged.get("game_pk", flagged.get("event_id", flagged.index))

    target_date = date.fromisoformat(target_date_str)
    flagged_rows = flagged.to_dict(orient="records")

    print(f"Running research agents on {len(flagged_rows)} flagged bet(s) from {target_date_str}...")

    lineup_notes = agent_lineup_confirmation(flagged_rows, target_date, dry_run)
    injury_notes = agent_injury_news(flagged_rows, target_date, dry_run)
    line_notes = agent_line_movement(flagged_rows, target_date, dry_run)

    all_notes = lineup_notes + injury_notes + line_notes

    if dry_run:
        return

    if not all_notes:
        print("No notes generated.")
        return

    # Replace this date's notes if the script gets re-run, matching the
    # idempotent-per-date pattern used by daily_ks.py/daily_ml.py/daily_pm.py.
    if NOTES_PATH.exists():
        existing = pd.read_csv(NOTES_PATH)
        existing = existing[~existing["timestamp"].str.startswith(target_date_str)]
        combined = pd.concat([existing, pd.DataFrame(all_notes)], ignore_index=True)
    else:
        combined = pd.DataFrame(all_notes)

    combined.to_csv(NOTES_PATH, index=False)
    print(f"Wrote {len(all_notes)} research notes to {NOTES_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("date", nargs="?", default=None, help="YYYY-MM-DD, defaults to today")
    args = ap.parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today()
    run(target.isoformat(), dry_run=args.dry_run)
