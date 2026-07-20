"""
WNBA data pull: ESPN's hidden API (site.api.espn.com), free, no key. Mirrors
fetch.py's structure/conventions (_cached_csv pattern, RAW/PROCESSED paths) for
a second sport.

Confirmed live (2026-07-13): a single `/scoreboard?dates=YYYYMMDD-YYYYMMDD`
call covers an ENTIRE season (up to `limit=1000` events) with final scores
already embedded -- no per-game box-score pull needed for the schedule/scores
table itself (unlike MLB, where the schedule and pitcher game logs are
separate pulls). One HTTP call per season.

`season.type` in ESPN's response: 1 = preseason (includes exhibitions against
non-league opponents like "Brazil" or "Toyota Antelopes"), 2 = regular season
(also includes the mid-season Commissioner's Cup games between real teams --
kept -- AND the All-Star Game, which is also tagged type 2 but between
fictitious draft-team names like "Team Stewart"/"Team Clark" -- excluded via
REAL_TEAM_NAMES), 3 = postseason. Only real-team regular-season games are used
for training (mirrors MLB's `gameType: "R"` filter).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"

# The real WNBA franchises -- used to drop All-Star/exhibition entries, which
# ESPN otherwise mixes into the same season.type as real games (see above).
# Golden State Valkyries joined as an expansion team in 2025; Portland Fire and
# Toronto Tempo joined in 2026 (confirmed live via the 2026 schedule pull --
# these two have zero prior-season history, so the rolling-form cold-start
# fallback in rolling_wnba.py's league-average prior is what carries them
# until they build up real in-season form).
REAL_TEAM_NAMES = {
    "Atlanta Dream", "Chicago Sky", "Connecticut Sun", "Dallas Wings",
    "Golden State Valkyries", "Indiana Fever", "Las Vegas Aces",
    "Los Angeles Sparks", "Minnesota Lynx", "New York Liberty",
    "Phoenix Mercury", "Portland Fire", "Seattle Storm", "Toronto Tempo",
    "Washington Mystics",
}

_session = requests.Session()
_retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retry))


def _get(url: str, params: dict | None = None) -> dict:
    r = _session.get(url, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def _cached_csv(path: Path, builder, *, refresh: bool = False) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_csv(path)
    df = builder()
    df.to_csv(path, index=False)
    return df


def _parse_scoreboard_events(events: list[dict], require_real_teams: bool = True) -> list[dict]:
    rows = []
    for e in events:
        if e.get("season", {}).get("type") != 2:
            continue
        comp = e["competitions"][0]
        home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        if require_real_teams and (home["team"]["displayName"] not in REAL_TEAM_NAMES
                                    or away["team"]["displayName"] not in REAL_TEAM_NAMES):
            continue
        completed = comp["status"]["type"]["completed"]
        rows.append({
            "game_id": str(e["id"]), "official_date": e["date"][:10],
            "home_team_id": int(home["team"]["id"]), "home_team_name": home["team"]["displayName"],
            "away_team_id": int(away["team"]["id"]), "away_team_name": away["team"]["displayName"],
            "home_score": int(home["score"]) if completed and home.get("score") is not None else None,
            "away_score": int(away["score"]) if completed and away.get("score") is not None else None,
            "completed": completed,
        })
    return rows


def fetch_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    """One real, completed regular-season game per row. WNBA regular season
    runs roughly May-October; requesting the full May 1 - Oct 31 window in one
    call covers it (confirmed: 288 real regular-season games for 2025, all
    events returned in a single response with `limit=1000`)."""
    path = RAW / f"schedule_wnba_{season}.csv"

    def build():
        d = _get(f"{ESPN_BASE}/scoreboard", params={
            "dates": f"{season}0501-{season}1031", "limit": 1000,
        })
        rows = _parse_scoreboard_events(d.get("events", []))
        rows = [r for r in rows if r["completed"]]
        df = pd.DataFrame(rows)
        df["season"] = season
        return df

    return _cached_csv(path, build, refresh=refresh)


def fetch_games_on_date(target_date_str: str) -> pd.DataFrame:
    """target_date_str: 'YYYY-MM-DD'. Live/today use -- NOT cached, since a
    game's completed/score status changes throughout the day."""
    yyyymmdd = target_date_str.replace("-", "")
    d = _get(f"{ESPN_BASE}/scoreboard", params={"dates": yyyymmdd})
    rows = _parse_scoreboard_events(d.get("events", []))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2021)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    frames = []
    for season in range(args.start_year, args.end_year + 1):
        df = fetch_schedule(season, refresh=args.refresh)
        print(f"{season}: {len(df)} games, {df['home_team_id'].nunique() + 0} home-team-id slots, "
              f"{pd.concat([df['home_team_name'], df['away_team_name']]).nunique()} unique teams")
        frames.append(df)

    all_games = pd.concat(frames, ignore_index=True)
    all_games.to_parquet(PROCESSED / "games_wnba.parquet", index=False)
    print(f"\ntotal: {len(all_games):,} games, seasons {args.start_year}-{args.end_year}")
    print(all_games.groupby("season").size())
