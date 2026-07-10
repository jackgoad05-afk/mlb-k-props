"""
As-of-date rolling features -- the fix for Session 2's flat-signal result.

Session 2 used full-prior-season aggregates for every pitcher/team feature to keep
the backtest leak-free (season-Y games used season-(Y-1) stats). That's leak-free but
throws away current form, and the backtest showed it: Brier score tied a naive
baseline exactly. This module computes the same quantities properly -- expanding,
strictly-before-this-game-date, within the current season -- falling back to the
prior season's rate stats early in the year via Bayesian shrinkage so a team or
pitcher's first few starts/games of a season don't swing wildly on a tiny sample.

Nothing here looks at a game's own result to build that game's features -- every
"as of" cutoff is the day before the game (shared across a doubleheader's two games).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

TEAM_SHRINK_K = 12       # "prior games" worth of weight given to the season-prior rate
TRAILING_WINDOW = 10      # pure trailing window, no shrinkage (NaN until enough games)


def _season_prior_rates(games: pd.DataFrame) -> pd.DataFrame:
    """Each team's runs-scored/allowed-per-game and win% for a season, to serve as the
    shrinkage prior for the FOLLOWING season."""
    home = games[["season", "home_team_id", "home_score", "away_score", "home_win"]].rename(
        columns={"home_team_id": "team_id", "home_score": "rs", "away_score": "ra", "home_win": "win"})
    away = games[["season", "away_team_id", "home_score", "away_score", "home_win"]].rename(
        columns={"away_team_id": "team_id", "away_score": "rs", "home_score": "ra"})
    away["win"] = 1 - games["home_win"].values
    both = pd.concat([home, away], ignore_index=True)
    agg = both.groupby(["season", "team_id"]).agg(
        rs_avg=("rs", "mean"), ra_avg=("ra", "mean"), win_pct=("win", "mean"),
    ).reset_index()
    agg["season"] = agg["season"] + 1  # shift forward: this becomes next season's prior
    return agg.set_index(["season", "team_id"]).sort_index()


def build_team_rolling_form(games: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (game_pk, side) with as-of-date-before rolling team form."""
    priors = _season_prior_rates(games)
    league_prior = games.groupby("season").agg(
        rs_avg=("home_score", "mean"), win_pct=("home_win", "mean")
    )  # crude league-average fallback for a team's first tracked season (no prior at all)

    g = games.sort_values(["season", "official_date", "game_pk"]).reset_index(drop=True)

    state: dict[tuple[int, int], dict] = {}  # (season, team_id) -> running sums + trailing deque
    rows = []

    for (season, date), day_idx in g.groupby(["season", "official_date"], sort=True).groups.items():
        day_idx = list(day_idx)
        snapshot = {}
        for idx in day_idx:
            row = g.loc[idx]
            for team_id in (row["home_team_id"], row["away_team_id"]):
                key = (season, team_id)
                if key in snapshot:
                    continue
                st = state.get(key)
                if st is None:
                    if key in priors.index:
                        p = priors.loc[key]
                        prior_rs, prior_ra, prior_wp = p["rs_avg"], p["ra_avg"], p["win_pct"]
                    else:
                        prior_rs = league_prior.loc[season, "rs_avg"] if season in league_prior.index else 4.3
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
                snapshot[team_id] = {
                    "games_played": n, "rs_form": shrunk_rs, "ra_form": shrunk_ra,
                    "win_form": shrunk_wp, "trail_win_pct": trail_wp, "trail_n": len(trail),
                }

        for idx in day_idx:
            row = g.loc[idx]
            h, a = row["home_team_id"], row["away_team_id"]
            rows.append({"game_pk": row["game_pk"], "team_id": h, "side": "home", **snapshot[h]})
            rows.append({"game_pk": row["game_pk"], "team_id": a, "side": "away", **snapshot[a]})

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

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Starter rolling form (built on top of per-start game logs pulled separately)
# --------------------------------------------------------------------------- #

STARTER_TRAIL_STARTS = 3  # trailing-N-starts window for current-form pitcher features


def _parse_ip(ip_str) -> float:
    """MLB's innings-pitched notation uses thirds, not decimals: '5.1' = 5 1/3, '5.2' = 5 2/3."""
    if pd.isna(ip_str):
        return np.nan
    s = str(ip_str)
    whole, _, frac = s.partition(".")
    whole = float(whole) if whole not in ("", "-") else 0.0
    frac_map = {"": 0.0, "0": 0.0, "1": 1 / 3, "2": 2 / 3}
    return whole + frac_map.get(frac, 0.0)


def build_starter_rolling_form(game_logs: pd.DataFrame, pitcher_season_all: pd.DataFrame) -> pd.DataFrame:
    """
    game_logs: one row per pitcher start -- mlbID, season, official_date, game_pk, ip, bf, so, bb, hr, er
    pitcher_season_all: the full (unlagged) pitcher_season table (mlbID, season, fip, k_bb_pct, IP).
      Shifted forward one season internally so a pitcher's ACTUAL prior season becomes the shrinkage
      prior for his first starts of the following season (or his whole line if he has none yet this
      season -- a rookie or an offseason signee). Passing the raw table straight through without this
      shift would use a pitcher's same-season aggregate as his own early-season prior -- the exact
      in-season leakage this module exists to avoid.

    Returns one row per start with as-of-date-BEFORE trailing-3-start form:
      trail_fip, trail_k_bb_pct, starts_this_season_so_far, days_rest
    """
    lagged = pitcher_season_all.drop_duplicates(subset=["season", "mlbID"]).copy()
    lagged["season"] = lagged["season"] + 1
    priors = lagged.set_index(["season", "mlbID"])[["fip", "k_bb_pct", "IP"]].sort_index()
    league_prior = lagged.groupby("season")[["fip", "k_bb_pct"]].mean()  # also prior-season, same reason

    gl = game_logs.sort_values(["mlbID", "season", "official_date"]).reset_index(drop=True)
    gl["ip_dec"] = gl["ip"].map(_parse_ip)

    out_rows = []
    for mlbID, pdf in gl.groupby("mlbID"):
        pdf = pdf.sort_values(["season", "official_date"])
        history = deque(maxlen=STARTER_TRAIL_STARTS)  # each item: dict(so, bb, hr, ip, er)
        last_date = None
        starts_this_season = 0
        current_season = None
        for _, row in pdf.iterrows():
            season = row["season"]
            if season != current_season:
                # new season: trailing window resets to the prior-season/league prior --
                # otherwise a pitcher's first start of April would silently use last
                # September's starts as his "recent form," and starts_this_season_so_far
                # would carry over a stale count instead of starting at 0.
                current_season = season
                history.clear()
                starts_this_season = 0
                last_date = None
            key = (season, mlbID)
            if key in priors.index:
                p = priors.loc[key]
                prior_fip, prior_kbb = p["fip"], p["k_bb_pct"]
            elif season in league_prior.index:
                prior_fip, prior_kbb = league_prior.loc[season, "fip"], league_prior.loc[season, "k_bb_pct"]
            else:
                prior_fip, prior_kbb = 4.30, 0.15

            if len(history) == 0:
                trail_fip, trail_kbb = prior_fip, prior_kbb
            else:
                ip_sum = sum(h["ip"] for h in history)
                so_sum = sum(h["so"] for h in history)
                bb_sum = sum(h["bb"] for h in history)
                hr_sum = sum(h["hr"] for h in history)
                if ip_sum > 0:
                    trail_fip = (13 * hr_sum + 3 * bb_sum - 2 * so_sum) / ip_sum + 3.10
                    bf_est = ip_sum * 4.3  # rough BF/IP if bf history unavailable
                else:
                    trail_fip = prior_fip
                trail_kbb = ((so_sum - bb_sum) / (ip_sum * 4.3)) if ip_sum > 0 else prior_kbb

            days_rest = (pd.Timestamp(row["official_date"]) - last_date).days if last_date is not None else np.nan

            out_rows.append({
                "mlbID": mlbID, "game_pk": row["game_pk"], "season": season,
                "trail_fip": trail_fip, "trail_k_bb_pct": trail_kbb,
                "starts_this_season_so_far": starts_this_season, "days_rest": days_rest,
            })

            if pd.notna(row["ip_dec"]) and row["ip_dec"] > 0:
                history.append({"ip": row["ip_dec"], "so": row["so"], "bb": row["bb"], "hr": row["hr"]})
            starts_this_season += 1
            last_date = pd.Timestamp(row["official_date"])

    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# K-model rolling form (src/model_ks.py) -- separate from the moneyline rolling
# form above so a change here can't regress the already-validated moneyline pipeline.
# --------------------------------------------------------------------------- #

K_TRAIL_STARTS = 3
K_TRAIL_DAYS = 30


def build_starter_k_form(game_logs: pd.DataFrame, pitcher_season_all: pd.DataFrame) -> pd.DataFrame:
    """
    Per-start, as-of-date-BEFORE rolling K features:
      trail_k_per9_3s, trail_bb_per9_3s, trail_ip_per_start, trail_pitch_count_avg
        -- trailing-3-start window
      trail_k_per9_30d -- trailing calendar-30-day window (separate signal: recent
        HOT/COLD stretch regardless of exactly how many starts that spans)
      starts_this_season_so_far, days_rest -- same definitions as build_starter_rolling_form

    Same season-boundary reset and prior-season-shrinkage fallback discipline as
    build_starter_rolling_form (see that docstring) -- both windows are empty at a
    pitcher's first start of a season and fall back to his actual prior season's K/9
    and BB/9 (or the league average if he has none), never his own same-season stats.
    """
    lagged = pitcher_season_all.drop_duplicates(subset=["season", "mlbID"]).copy()
    lagged["season"] = lagged["season"] + 1
    lagged["k_per9"] = 9 * lagged["SO"] / lagged["IP"].replace(0, np.nan)
    lagged["bb_per9"] = 9 * lagged["BB"] / lagged["IP"].replace(0, np.nan)
    priors = lagged.set_index(["season", "mlbID"])[["k_per9", "bb_per9", "IP"]].sort_index()
    league_prior = lagged.groupby("season")[["k_per9", "bb_per9"]].mean()
    LEAGUE_K9_FALLBACK, LEAGUE_BB9_FALLBACK = 8.5, 3.2  # only hit if even the shifted league table is empty

    gl = game_logs.sort_values(["mlbID", "season", "official_date"]).reset_index(drop=True)
    gl["ip_dec"] = gl["ip"].map(_parse_ip)

    out_rows = []
    for mlbID, pdf in gl.groupby("mlbID"):
        pdf = pdf.sort_values(["season", "official_date"])
        hist3: deque = deque(maxlen=K_TRAIL_STARTS)  # dict(date, ip, so, bb, pitches)
        hist30: list = []
        last_date = None
        starts_this_season = 0
        current_season = None

        for _, row in pdf.iterrows():
            season = row["season"]
            date = pd.Timestamp(row["official_date"])
            if season != current_season:
                current_season = season
                hist3.clear()
                hist30 = []
                starts_this_season = 0
                last_date = None

            key = (season, mlbID)
            if key in priors.index:
                p = priors.loc[key]
                prior_k9, prior_bb9 = p["k_per9"], p["bb_per9"]
            elif season in league_prior.index:
                prior_k9 = league_prior.loc[season, "k_per9"]
                prior_bb9 = league_prior.loc[season, "bb_per9"]
            else:
                prior_k9, prior_bb9 = LEAGUE_K9_FALLBACK, LEAGUE_BB9_FALLBACK
            if pd.isna(prior_k9):
                prior_k9 = LEAGUE_K9_FALLBACK
            if pd.isna(prior_bb9):
                prior_bb9 = LEAGUE_BB9_FALLBACK

            # trailing-3-start window
            if len(hist3) == 0:
                trail_k9_3s, trail_bb9_3s = prior_k9, prior_bb9
                trail_ip_per_start = np.nan  # filled from prior-season IP/GS below
            else:
                ip_sum = sum(h["ip"] for h in hist3)
                trail_k9_3s = (9 * sum(h["so"] for h in hist3) / ip_sum) if ip_sum > 0 else prior_k9
                trail_bb9_3s = (9 * sum(h["bb"] for h in hist3) / ip_sum) if ip_sum > 0 else prior_bb9
                trail_ip_per_start = ip_sum / len(hist3)
            if pd.isna(trail_ip_per_start):
                prior_ip = priors.loc[key, "IP"] if key in priors.index else np.nan
                trail_ip_per_start = (prior_ip / 20) if pd.notna(prior_ip) and prior_ip > 0 else 5.0
                trail_ip_per_start = float(np.clip(trail_ip_per_start, 3.0, 7.0))

            pitch_vals = [h["pitches"] for h in hist3 if pd.notna(h.get("pitches"))]
            trail_pitch_count_avg = float(np.mean(pitch_vals)) if pitch_vals else trail_ip_per_start * 16.0

            # trailing-30-calendar-day window (prune anything older than 30 days first)
            hist30 = [h for h in hist30 if (date - h["date"]).days <= K_TRAIL_DAYS]
            ip30 = sum(h["ip"] for h in hist30)
            trail_k9_30d = (9 * sum(h["so"] for h in hist30) / ip30) if ip30 > 0 else trail_k9_3s

            days_rest = (date - last_date).days if last_date is not None else np.nan

            out_rows.append({
                "mlbID": mlbID, "game_pk": row["game_pk"], "season": season,
                "trail_k_per9_3s": trail_k9_3s, "trail_bb_per9_3s": trail_bb9_3s,
                "trail_k_per9_30d": trail_k9_30d, "trail_ip_per_start": trail_ip_per_start,
                "trail_pitch_count_avg": trail_pitch_count_avg,
                "starts_this_season_so_far": starts_this_season, "days_rest": days_rest,
            })

            if pd.notna(row["ip_dec"]) and row["ip_dec"] > 0:
                entry = {"date": date, "ip": row["ip_dec"], "so": row["so"], "bb": row["bb"],
                         "pitches": row.get("pitches")}
                hist3.append(entry)
                hist30.append(entry)
            starts_this_season += 1
            last_date = date

    return pd.DataFrame(out_rows)
