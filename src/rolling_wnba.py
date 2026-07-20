"""
As-of-date rolling team form for WNBA -- same method as rolling.py's
build_team_rolling_form for MLB (expanding, strictly-before-this-game-date,
within-season, Bayesian-shrunk toward the prior season's rate early on), just
without a starter-pitcher equivalent (basketball has no per-game "starter" the
way MLB has a starting pitcher -- team-level form is the whole feature set).
Also adds rest_days, which rolling.py computes at the pitcher level for MLB but
has no team-level equivalent of yet -- back-to-backs matter a lot in the WNBA's
5-day-a-week schedule.

Nothing here looks at a game's own result to build that game's features --
every "as of" cutoff is the day before the game.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

TEAM_SHRINK_K = 8     # "prior games" worth of weight given to the season-prior rate
                       # (lower than MLB's 12 -- a WNBA season is ~40 games/team vs.
                       # MLB's 162, so a full season carries proportionally less signal
                       # to shrink toward, and the prior itself is noisier)
TRAILING_WINDOW = 10   # pure trailing window, no shrinkage (NaN until enough games)
DEFAULT_REST_DAYS = 2.0  # WNBA teams commonly play every 2-3 days


def _season_prior_rates(games: pd.DataFrame) -> pd.DataFrame:
    """Each team's points-scored/allowed-per-game and win% for a season, to serve
    as the shrinkage prior for the FOLLOWING season."""
    home = games[["season", "home_team_id", "home_score", "away_score", "home_win"]].rename(
        columns={"home_team_id": "team_id", "home_score": "rs", "away_score": "ra", "home_win": "win"})
    away = games[["season", "away_team_id", "home_score", "away_score", "home_win"]].rename(
        columns={"away_team_id": "team_id", "home_score": "ra", "away_score": "rs"})
    away["win"] = 1 - games["home_win"].values
    both = pd.concat([home, away], ignore_index=True)
    agg = both.groupby(["season", "team_id"]).agg(
        rs_avg=("rs", "mean"), ra_avg=("ra", "mean"), win_pct=("win", "mean"),
    ).reset_index()
    agg["season"] = agg["season"] + 1  # shift forward: this becomes next season's prior
    return agg.set_index(["season", "team_id"]).sort_index()


def build_team_rolling_form(games: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (game_id, side) with as-of-date-before rolling team form:
    games_played, rs_form/ra_form (shrunk toward season prior), win_form (shrunk),
    trail_win_pct (pure trailing-10, NaN until 5+ games), rest_days."""
    priors = _season_prior_rates(games)
    league_prior = games.groupby("season").agg(
        rs_avg=("home_score", "mean"), win_pct=("home_win", "mean")
    )  # crude league-average fallback for a team's first tracked season (e.g. expansion)

    g = games.sort_values(["season", "official_date", "game_id"]).reset_index(drop=True)

    state: dict[tuple[int, int], dict] = {}  # (season, team_id) -> running sums + trailing deque
    last_game_date: dict[int, pd.Timestamp] = {}  # team_id -> most recent game date (any season)
    rows = []

    for (season, date), day_idx in g.groupby(["season", "official_date"], sort=True).groups.items():
        day_idx = list(day_idx)
        date_ts = pd.Timestamp(date)
        snapshot = {}
        for idx in day_idx:
            row = g.loc[idx]
            for team_id in (row["home_team_id"], row["away_team_id"]):
                key = (season, team_id)
                if team_id in snapshot:
                    continue
                st = state.get(key)
                if st is None:
                    if key in priors.index:
                        p = priors.loc[key]
                        prior_rs, prior_ra, prior_wp = p["rs_avg"], p["ra_avg"], p["win_pct"]
                    else:
                        prior_rs = league_prior.loc[season, "rs_avg"] if season in league_prior.index else 82.0
                        prior_ra = prior_rs
                        prior_wp = 0.5
                    st = {"n": 0, "rs_sum": 0.0, "ra_sum": 0.0, "wins": 0.0,
                          "prior_rs": prior_rs, "prior_ra": prior_ra, "prior_wp": prior_wp,
                          "trail": deque(maxlen=TRAILING_WINDOW)}
                    state[key] = st

                n, k = st["n"], TEAM_SHRINK_K
                shrunk_rs = (st["rs_sum"] + k * st["prior_rs"]) / (n + k)
                shrunk_ra = (st["ra_sum"] + k * st["prior_ra"]) / (n + k)
                shrunk_wp = (st["wins"] + k * st["prior_wp"]) / (n + k)
                trail = st["trail"]
                trail_wp = float(np.mean(trail)) if len(trail) >= 5 else np.nan

                prev_date = last_game_date.get(team_id)
                rest = (date_ts - prev_date).days if prev_date is not None else np.nan

                snapshot[team_id] = {
                    "games_played": n, "rs_form": shrunk_rs, "ra_form": shrunk_ra,
                    "win_form": shrunk_wp, "trail_win_pct": trail_wp, "trail_n": len(trail),
                    "rest_days": rest,
                }

        for idx in day_idx:
            row = g.loc[idx]
            h, a = row["home_team_id"], row["away_team_id"]
            rows.append({"game_id": row["game_id"], "team_id": h, "side": "home", **snapshot[h]})
            rows.append({"game_id": row["game_id"], "team_id": a, "side": "away", **snapshot[a]})

        for idx in day_idx:
            row = g.loc[idx]
            for team_id, rs, ra, win in [
                (row["home_team_id"], row["home_score"], row["away_score"], row["home_win"]),
                (row["away_team_id"], row["away_score"], row["home_score"], 1 - row["home_win"]),
            ]:
                st = state[(season, team_id)]
                st["n"] += 1
                st["rs_sum"] += rs
                st["ra_sum"] += ra
                st["wins"] += win
                st["trail"].append(win)
                last_game_date[team_id] = date_ts

    return pd.DataFrame(rows)
