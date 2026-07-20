"""
Session 1: pull and clean 2021-2025 MLB data into data/raw/ (untouched pulls)
and data/processed/ (model-ready tables).

Sources actually used (see CLAUDE.md for why this differs from the spec):
  - MLB Stats API (statsapi.mlb.com)   -> schedule, final scores, team hitting splits
  - Baseball Savant via pybaseball     -> pitcher xwOBA against, barrel%, whiff%, velo
  - Baseball-Reference via pybaseball  -> pitcher IP/GS/BF/SO/BB/HR (FIP + K-BB% computed)
  - Static CSV                         -> park factors (manually maintained, see note below)

FanGraphs' leaderboard page (pybaseball.pitching_stats / team_batting) is blocked by a
Cloudflare interactive challenge as of 2026-07-08 -- confirmed with plain requests, a
spoofed User-Agent, and cloudscraper, all returning 403. xFIP/SIERA/wRC+ are FanGraphs-
proprietary and not available until that access is restored (see CLAUDE.md).

Usage:
    python src/fetch.py --start-year 2021 --end-year 2025
    python src/fetch.py --start-year 2021 --end-year 2025 --refresh   # ignore raw cache
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# pybaseball is a heavy, research-only dependency (Statcast/bref leaderboards) not
# needed by the live daily pipeline (daily_ks.py / reconcile_ks.py only use the
# lightweight MLB Stats API functions below). Imported lazily so those callers don't
# need it installed at all -- see requirements-actions.txt vs. requirements.txt.
_pb = None


def _pybaseball():
    global _pb
    if _pb is None:
        import pybaseball as pb
        pb.cache.enable()
        _pb = pb
    return _pb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"

_session = requests.Session()
_retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retry))


def _get(url: str, params: dict | None = None) -> dict:
    r = _session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _fix_bref_mojibake(name) -> str:
    """pybaseball's bref scraper leaks literal '\\xHH' escapes for accented names
    (e.g. 'L\\xc3\\xb3pez' instead of 'López'). Decode those runs back to UTF-8."""
    if not isinstance(name, str) or "\\x" not in name:
        return name

    def repl(match: re.Match) -> str:
        hex_bytes = re.findall(r"\\x([0-9a-fA-F]{2})", match.group(0))
        raw = bytes(int(h, 16) for h in hex_bytes)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return match.group(0)

    return re.sub(r"(?:\\x[0-9a-fA-F]{2})+", repl, name)


def _cached_csv(path: Path, builder, *, refresh: bool = False) -> pd.DataFrame:
    """Load `path` if it exists (and not --refresh); otherwise call builder() and save."""
    if path.exists() and not refresh:
        return pd.read_csv(path)
    df = builder()
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------- #
# Teams
# --------------------------------------------------------------------------- #

def fetch_teams(season: int, refresh: bool = False) -> pd.DataFrame:
    path = RAW / f"teams_{season}.csv"

    def build():
        d = _get(f"{STATSAPI_BASE}/teams", params={"sportId": 1, "activeStatus": "Y", "season": season})
        rows = []
        for t in d["teams"]:
            rows.append({
                "season": season,
                "team_id": t["id"],
                "abbreviation": t.get("abbreviation"),
                "name": t.get("name"),
                "venue_id": t.get("venue", {}).get("id"),
                "venue_name": t.get("venue", {}).get("name"),
                "league": t.get("league", {}).get("name"),
            })
        return pd.DataFrame(rows)

    return _cached_csv(path, build, refresh=refresh)


# --------------------------------------------------------------------------- #
# Schedule / final scores
# --------------------------------------------------------------------------- #

def fetch_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    path = RAW / f"schedule_{season}.csv"

    def build():
        d = _get(f"{STATSAPI_BASE}/schedule", params={
            "sportId": 1, "startDate": f"{season}-01-01", "endDate": f"{season}-12-31",
            "gameType": "R", "hydrate": "probablePitcher",
        })
        rows = []
        for date_block in d.get("dates", []):
            for g in date_block.get("games", []):
                if g.get("status", {}).get("codedGameState") != "F":
                    continue
                home, away = g["teams"]["home"], g["teams"]["away"]
                rows.append({
                    "game_pk": g["gamePk"],
                    "season": season,
                    "official_date": g["officialDate"],
                    "home_team_id": home["team"]["id"],
                    "home_team_name": home["team"]["name"],
                    "away_team_id": away["team"]["id"],
                    "away_team_name": away["team"]["name"],
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                    "home_probable_pitcher_id": home.get("probablePitcher", {}).get("id"),
                    "away_probable_pitcher_id": away.get("probablePitcher", {}).get("id"),
                    "venue_id": g.get("venue", {}).get("id"),
                    "venue_name": g.get("venue", {}).get("name"),
                    "day_night": g.get("dayNight"),
                    "doubleheader": g.get("doubleHeader"),
                })
        return pd.DataFrame(rows)

    return _cached_csv(path, build, refresh=refresh)


# --------------------------------------------------------------------------- #
# Pitcher handedness (needed to pick the right team-offense split as a feature)
# --------------------------------------------------------------------------- #

def fetch_pitcher_handedness(pitcher_ids: list[int], refresh: bool = False, sleep: float = 0.2) -> pd.DataFrame:
    path = RAW / "pitcher_handedness.csv"
    if path.exists() and not refresh:
        cached = pd.read_csv(path)
        missing = sorted(set(int(i) for i in pitcher_ids if pd.notna(i)) - set(cached["mlbID"]))
        if not missing:
            return cached
        pitcher_ids = missing
    else:
        cached = pd.DataFrame(columns=["mlbID", "throws"])

    ids = sorted(set(int(i) for i in pitcher_ids if pd.notna(i)))
    rows = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        d = _get(f"{STATSAPI_BASE}/people", params={"personIds": ",".join(str(x) for x in chunk)})
        for p in d.get("people", []):
            rows.append({"mlbID": p["id"], "throws": p.get("pitchHand", {}).get("code")})
        time.sleep(sleep)

    df = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(subset="mlbID")
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------- #
# Per-start pitcher game logs (real current-form data, not season aggregates)
# --------------------------------------------------------------------------- #

def fetch_starter_game_logs(season: int, pitcher_ids: list[int], refresh: bool = False, sleep: float = 0.1) -> pd.DataFrame:
    """refresh=True re-fetches ONLY the given pitcher_ids (e.g. to pick up a pitcher's
    most recent start) -- it must NOT discard other pitchers' already-cached rows for
    this season, or a daily_ks.py run for today's ~15 starters would silently wipe out
    every other pitcher's data collected earlier in the season."""
    path = RAW / f"pitcher_gamelogs_{season}.csv"
    pitcher_ids = sorted(set(int(i) for i in pitcher_ids))
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame()

    if refresh:
        to_fetch = pitcher_ids
        if not cached.empty:
            cached = cached[~cached["mlbID"].isin(to_fetch)]
    else:
        have = set(cached["mlbID"]) if not cached.empty else set()
        to_fetch = sorted(set(pitcher_ids) - have)
        if not to_fetch:
            return cached

    rows = []
    for pid in to_fetch:
        try:
            d = _get(f"{STATSAPI_BASE}/people/{pid}/stats", params={"stats": "gameLog", "group": "pitching", "season": season})
            splits = d.get("stats", [{}])[0].get("splits", [])
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            print(f"  [warn] gamelog {season} pitcher={pid}: {e}", file=sys.stderr)
            splits = []
        for s in splits:
            if s.get("gameType") != "R" or not s.get("stat", {}).get("gamesStarted"):
                continue
            st = s["stat"]
            rows.append({
                "mlbID": pid, "season": season, "official_date": s.get("date"),
                "game_pk": s.get("game", {}).get("gamePk"),
                "ip": st.get("inningsPitched"), "bf": st.get("battersFaced"),
                "so": st.get("strikeOuts"), "bb": st.get("baseOnBalls"),
                "hr": st.get("homeRuns"), "er": st.get("earnedRuns"),
                "pitches": st.get("numberOfPitches"),
                "is_home": s.get("isHome"), "opponent_team_id": s.get("opponent", {}).get("id"),
            })
        time.sleep(sleep)

    df = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(subset=["mlbID", "game_pk"])
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------- #
# Home-plate umpire assignment + game strikeout totals (K model umpire feature)
# --------------------------------------------------------------------------- #

def fetch_game_officials_and_k_totals(game_pks: list[int], refresh: bool = False, sleep: float = 0.05) -> pd.DataFrame:
    """One row per game: home-plate umpire id/name + total strikeouts (both teams
    combined) from the game's box score. Uses /game/{pk}/boxscore, not the full
    /feed/live -- same officials data, ~163KB instead of the full play-by-play, and
    already has both teams' batting strikeOuts totals in the same response."""
    path = RAW / "game_umpires.csv"
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame()
    game_pks = sorted(set(int(g) for g in game_pks))

    if refresh:
        to_fetch = game_pks
        if not cached.empty:
            cached = cached[~cached["game_pk"].isin(to_fetch)]
    else:
        have = set(cached["game_pk"]) if not cached.empty else set()
        to_fetch = sorted(set(game_pks) - have)
        if not to_fetch:
            return cached

    rows = []
    CHECKPOINT_EVERY = 250
    for i, pk in enumerate(to_fetch):
        try:
            d = _get(f"{STATSAPI_BASE}/game/{pk}/boxscore")
            officials = d.get("officials", [])
            hp = next((o for o in officials if o.get("officialType") == "Home Plate"), None)
            teams = d.get("teams", {})
            home_k = teams.get("home", {}).get("teamStats", {}).get("batting", {}).get("strikeOuts")
            away_k = teams.get("away", {}).get("teamStats", {}).get("batting", {}).get("strikeOuts")
            rows.append({
                "game_pk": pk,
                "hp_umpire_id": hp["official"]["id"] if hp else None,
                "hp_umpire_name": hp["official"]["fullName"] if hp else None,
                "game_total_k": (home_k or 0) + (away_k or 0) if home_k is not None and away_k is not None else None,
            })
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            print(f"  [warn] boxscore game_pk={pk}: {e}", file=sys.stderr)

        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(to_fetch):
            checkpoint = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(subset=["game_pk"])
            checkpoint.to_csv(path, index=False)
            print(f"  ... {i + 1}/{len(to_fetch)} boxscores fetched, checkpoint saved ({len(checkpoint)} total rows)")
        time.sleep(sleep)

    return pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(subset=["game_pk"])


# --------------------------------------------------------------------------- #
# Team hitting splits (overall, vs RHP, vs LHP)
# --------------------------------------------------------------------------- #

SPLIT_CODES = {"overall": None, "vs_rhp": "vr", "vs_lhp": "vl"}


def fetch_team_hitting(season: int, teams: pd.DataFrame, refresh: bool = False, sleep: float = 0.15) -> pd.DataFrame:
    path = RAW / f"team_hitting_{season}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)

    rows = []
    for _, team in teams.iterrows():
        for split_name, sit_code in SPLIT_CODES.items():
            params = {"stats": "statSplits" if sit_code else "season", "group": "hitting", "season": season}
            if sit_code:
                params["sitCodes"] = sit_code
            try:
                d = _get(f"{STATSAPI_BASE}/teams/{team['team_id']}/stats", params=params)
                splits = d["stats"][0]["splits"]
                if not splits:
                    continue
                stat = splits[0]["stat"]
            except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                print(f"  [warn] team_hitting {season} team={team['abbreviation']} split={split_name}: {e}", file=sys.stderr)
                continue
            stat = dict(stat)
            stat.update({"season": season, "team_id": team["team_id"], "abbreviation": team["abbreviation"], "split": split_name})
            rows.append(stat)
            time.sleep(sleep)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------- #
# Statcast pitcher leaderboards (Baseball Savant, via pybaseball)
# --------------------------------------------------------------------------- #

def fetch_statcast_pitcher_leaderboards(season: int, refresh: bool = False) -> dict[str, pd.DataFrame]:
    pb = _pybaseball()
    out = {}

    def load_or_build(name, builder):
        path = RAW / f"statcast_{name}_{season}.csv"
        return _cached_csv(path, builder, refresh=refresh)

    out["expected_stats"] = load_or_build("expected_stats", lambda: pb.statcast_pitcher_expected_stats(season, minPA=1))
    out["exitvelo_barrels"] = load_or_build("exitvelo_barrels", lambda: pb.statcast_pitcher_exitvelo_barrels(season, minBBE=1))
    out["arsenal_stats"] = load_or_build("arsenal_stats", lambda: pb.statcast_pitcher_arsenal_stats(season, minPA=1))
    out["pitch_velo"] = load_or_build("pitch_velo", lambda: pb.statcast_pitcher_pitch_arsenal(season, minP=1, arsenal_type="avg_speed"))
    return out


# --------------------------------------------------------------------------- #
# Baseball-Reference pitching lines (via pybaseball) -- IP/GS/BF/SO/BB/HR
# --------------------------------------------------------------------------- #

def fetch_pitching_bref(season: int, refresh: bool = False, sleep: float = 2.0) -> pd.DataFrame:
    path = RAW / f"pitching_bref_{season}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    df = _pybaseball().pitching_stats_bref(season)
    time.sleep(sleep)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------- #
# Park factors -- static, manually maintained (spec: "Static table, update yearly")
# --------------------------------------------------------------------------- #

# Approximate 3-yr-blended FanGraphs-style run park factors (100 = neutral).
# SOURCE: publicly reported park factor rankings; not scraped from FanGraphs.
# Refresh manually each offseason from an authoritative park factor table.
# Keys match MLB Stats API team abbreviations exactly (AZ not ARI, CWS not CHW).
_PARK_FACTOR_SEED = {
    "COL": 112, "CIN": 106, "BOS": 105, "TEX": 104, "PHI": 103, "BAL": 103,
    "MIL": 102, "AZ": 102, "TOR": 101, "CHC": 101, "MIN": 100, "LAA": 100,
    "HOU": 100, "STL": 100, "WSH": 99, "ATL": 99, "SD": 99, "CWS": 99,
    "CLE": 98, "KC": 98, "TB": 97, "LAD": 97, "NYY": 97, "SF": 96,
    "DET": 96, "PIT": 96, "NYM": 95, "SEA": 94, "OAK": 94, "MIA": 92,
    # Athletics relocated to Sutter Health Park (Sacramento, minor-league sized) for 2025+;
    # MLB Stats API abbreviation changed OAK -> ATH. No stable multi-year park factor exists
    # yet for this venue -- 108 is a rough placeholder (small dims, reported as hitter-friendly
    # in early-2025 coverage). Replace with a real figure once enough of a track record exists.
    "ATH": 108,
}


def build_park_factors(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    path = RAW / "park_factors.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    rows = [{"abbreviation": abbr, "season": s, "park_factor_runs": v}
            for abbr, v in _PARK_FACTOR_SEED.items() for s in seasons]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------- #
# Processed: games
# --------------------------------------------------------------------------- #

def build_processed_games(seasons: list[int]) -> pd.DataFrame:
    parts = [pd.read_csv(RAW / f"schedule_{s}.csv") for s in seasons]
    games = pd.concat(parts, ignore_index=True)
    games = games.dropna(subset=["home_score", "away_score"])
    games["home_score"] = games["home_score"].astype(int)
    games["away_score"] = games["away_score"].astype(int)
    games["total_runs"] = games["home_score"] + games["away_score"]
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games["official_date"] = pd.to_datetime(games["official_date"])
    games = games.drop_duplicates(subset=["game_pk"])
    out = PROCESSED / "games.parquet"
    games.to_parquet(out, index=False)
    return games


# --------------------------------------------------------------------------- #
# Processed: pitcher_season
# --------------------------------------------------------------------------- #

# FIP constant is fairly stable year to year (~3.10-3.15); using 3.10 flat for v1.
FIP_CONSTANT = 3.10


def build_processed_pitchers(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        bref = pd.read_csv(RAW / f"pitching_bref_{s}.csv")
        bref = bref.dropna(subset=["mlbID"]).copy()
        bref["mlbID"] = bref["mlbID"].astype(int)

        exp = pd.read_csv(RAW / f"statcast_expected_stats_{s}.csv")
        evb = pd.read_csv(RAW / f"statcast_exitvelo_barrels_{s}.csv")
        arsenal = pd.read_csv(RAW / f"statcast_arsenal_stats_{s}.csv")
        velo = pd.read_csv(RAW / f"statcast_pitch_velo_{s}.csv")

        # arsenal_stats is long (one row per pitch type) -> collapse to pitcher level,
        # weighting whiff% and k% by pitch count.
        arsenal = arsenal.dropna(subset=["pitches"]).copy()
        arsenal["_w_whiff"] = arsenal["whiff_percent"] * arsenal["pitches"]
        arsenal["_w_k"] = arsenal["k_percent"] * arsenal["pitches"]
        g = arsenal.groupby("player_id").agg(
            _w_whiff=("_w_whiff", "sum"), _w_k=("_w_k", "sum"), pitches=("pitches", "sum")
        )
        agg = pd.DataFrame({
            "player_id": g.index,
            "whiff_pct": g["_w_whiff"] / g["pitches"],
            "k_pct_arsenal": g["_w_k"] / g["pitches"],
        }).reset_index(drop=True)

        velo = velo.copy()
        velo["fastball_velo"] = velo["ff_avg_speed"].fillna(velo["si_avg_speed"])
        velo = velo.rename(columns={"pitcher": "player_id"})[["player_id", "fastball_velo"]]

        exp = exp.rename(columns={"est_woba": "xwoba_against"})[["player_id", "xwoba_against"]]
        evb = evb.rename(columns={"brl_percent": "barrel_pct_allowed"})[["player_id", "barrel_pct_allowed"]]

        savant = exp.merge(evb, on="player_id", how="outer") \
                     .merge(agg, on="player_id", how="outer") \
                     .merge(velo, on="player_id", how="outer")

        merged = bref.merge(savant, left_on="mlbID", right_on="player_id", how="left")
        merged["season"] = s
        merged["Name"] = merged["Name"].map(_fix_bref_mojibake)

        ip = merged["IP"].replace(0, np.nan)
        merged["fip"] = ((13 * merged["HR"] + 3 * merged["BB"] - 2 * merged["SO"]) / ip) + FIP_CONSTANT
        merged["k_bb_pct"] = (merged["SO"] - merged["BB"]) / merged["BF"].replace(0, np.nan)

        keep = ["season", "mlbID", "Name", "Tm", "GS", "G", "IP", "BF", "SO", "BB", "HR", "ERA",
                "fip", "k_bb_pct", "xwoba_against", "barrel_pct_allowed", "whiff_pct",
                "k_pct_arsenal", "fastball_velo"]
        frames.append(merged[keep])

    pitchers = pd.concat(frames, ignore_index=True)
    out = PROCESSED / "pitcher_season.parquet"
    pitchers.to_parquet(out, index=False)
    return pitchers


# --------------------------------------------------------------------------- #
# Processed: team_season (with park-adjusted offense index, wRC+ proxy)
# --------------------------------------------------------------------------- #

def build_processed_teams(seasons: list[int]) -> pd.DataFrame:
    park = pd.read_csv(RAW / "park_factors.csv")
    frames = []
    for s in seasons:
        th = pd.read_csv(RAW / f"team_hitting_{s}.csv")
        th["k_pct"] = th["strikeOuts"] / th["plateAppearances"]
        th["obp"] = pd.to_numeric(th["obp"], errors="coerce")
        th["slg"] = pd.to_numeric(th["slg"], errors="coerce")
        th["ops"] = th["obp"] + th["slg"]

        league_ops = th.loc[th["split"] == "overall", "ops"].mean()
        th = th.merge(park[park["season"] == s][["abbreviation", "park_factor_runs"]],
                      on="abbreviation", how="left")
        # OPS+-style index: league-relative OPS, park-adjusted (100 = league average).
        th["off_index"] = 100 * (th["ops"] / league_ops) / (th["park_factor_runs"] / 100)
        frames.append(th[["season", "team_id", "abbreviation", "split", "plateAppearances",
                           "obp", "slg", "ops", "k_pct", "park_factor_runs", "off_index"]])

    teams = pd.concat(frames, ignore_index=True)
    out = PROCESSED / "team_season.parquet"
    teams.to_parquet(out, index=False)
    return teams


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run(start_year: int, end_year: int, refresh: bool = False, lag_year: bool = True):
    seasons = list(range(start_year, end_year + 1))
    # Pull one extra season before start_year so games in start_year have a leak-free
    # prior-season pitcher/team baseline to lag from (features.py uses season Y-1 stats
    # to predict season Y games -- never same-season stats, which would leak future
    # in-season performance into the prediction).
    stat_seasons = ([start_year - 1] + seasons) if lag_year else seasons

    for s in stat_seasons:
        print(f"--- {s}{'  (lag-only, no schedule pull)' if s == start_year - 1 and lag_year else ''} ---")
        teams = fetch_teams(s, refresh=refresh)
        print(f"  teams: {len(teams)}")

        if s in seasons:
            sched = fetch_schedule(s, refresh=refresh)
            print(f"  schedule (final games): {len(sched)}")

        th = fetch_team_hitting(s, teams, refresh=refresh)
        print(f"  team_hitting rows: {len(th)}")

        sc = fetch_statcast_pitcher_leaderboards(s, refresh=refresh)
        for k, v in sc.items():
            print(f"  statcast {k}: {len(v)}")

        bref = fetch_pitching_bref(s, refresh=refresh)
        print(f"  pitching_bref: {len(bref)}")

    build_park_factors(stat_seasons, refresh=refresh)

    print("\n=== building processed tables ===")
    games = build_processed_games(seasons)
    pitchers = build_processed_pitchers(stat_seasons)
    teams_tbl = build_processed_teams(stat_seasons)

    pitcher_ids = set(pitchers["mlbID"].dropna().astype(int))
    pitcher_ids |= set(games["home_probable_pitcher_id"].dropna().astype(int))
    pitcher_ids |= set(games["away_probable_pitcher_id"].dropna().astype(int))
    hands = fetch_pitcher_handedness(sorted(pitcher_ids), refresh=refresh)
    print(f"\npitcher_handedness: {len(hands)} pitchers")

    for name, df in [("games", games), ("pitcher_season", pitchers), ("team_season", teams_tbl)]:
        print(f"\n{name}: {len(df):,} rows, {df.shape[1]} cols")
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df.sample(min(5, len(df)), random_state=0).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2021)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--refresh", action="store_true", help="ignore cached raw files and re-pull")
    ap.add_argument("--no-lag-year", action="store_true", help="skip pulling start_year-1 stats (breaks lagged features)")
    args = ap.parse_args()
    run(args.start_year, args.end_year, refresh=args.refresh, lag_year=not args.no_lag_year)
