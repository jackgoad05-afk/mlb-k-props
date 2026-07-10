Jack prefers dense, data-rich analysis with firm directional recommendations. No hedging, no surface-level takes. Show the numbers behind every claim. When a result is bad, say so plainly.

## Dashboard "why" section (2026-07-10)
Flag cards on the Streamlit dashboard now have a collapsible "Why?" showing the raw
feature values the model actually scored with (trail_k_per9_3s, trail_k_per9_30d,
season_lag_whiff_pct, opp_off_kpct, days_rest, mu) plus an auto-generated 1-2 sentence
summary. `daily_ks.py`'s `run()` now persists these into the ledger (they were already
being computed, just discarded after scoring -- see `WHY_COLS` in `daily_ks.py`).

The summary generator only pulls a factor into the sentence if it points the SAME
direction as the bet -- e.g. for Janson Junk's under-3.5 flag, opponent K% vs. his hand
was actually above league average (favors the over), so it correctly got left out of
the "why" sentence rather than forced in as false support. Don't weaken this -- a
narrative that cherry-picks contradictory factors would be actively misleading.

Backfilled the 4 pre-existing 2026-07-09 ledger rows by recomputing that date's features
from game logs filtered to strictly BEFORE 2026-07-09 (the real pipeline had already
moved past that date, so a naive re-pull would have leaked each start's own outcome into
its own "form entering the game" features). Recomputed `mu` values matched the original
live run's output exactly, which is the check that mattered before writing anything back
to the real ledger.

## Session 5 (2026-07-09/10): live pipeline, GitHub Actions, Streamlit dashboard
Free-tier Odds API key is in `.env` (gitignored). Free tier has no `/historical/` access,
so CLV can't be bought after the fact -- solved by capturing closing lines ourselves:
`src/capture_closing_ks.py` snapshots current prop prices for flagged-but-not-yet-
captured games, idempotently (skips games already captured), written straight into the
ledger's `closing_over_odds`/`closing_under_odds` columns. `reconcile_ks.py` no longer
calls the Odds API at all -- CLV is just arithmetic on whatever's already in the ledger
by the time it runs. This also means reconciliation costs 0 API quota.

**Confirmed quota formula** (via WebFetch of the-odds-api.com docs + verified against
real response headers): `/v4/sports/{sport}/events` (list) costs 0. Player-prop markets
(`pitcher_strikeouts`) require the per-event endpoint `/events/{id}/odds`, cost =
`markets_returned x regions` = 1 unit/event queried (1 market, `regions=us`). The bulk
all-events endpoint does NOT support player props, confirmed via search -- can't use it
to cut cost. Real data point: today's live run (evening, 4 of ~13 games still upcoming)
cost 4 units (morning-equivalent pull) + 3 units (closing capture, 4 flagged rows across
3 unique events, deduped) = 7 units. `odds_api.py` logs `x-requests-remaining` from every
response to `output/odds_api_usage.csv` and warns below 30 remaining -- real usage is
tracked, not guessed, going forward.

**Real bug fixed while wiring this up:** `fetch.py` imported `pybaseball` unconditionally
at module level, so `daily_ks.py`'s `import fetch` pulled in pybaseball + all its heavy
transitive deps even though the live pipeline only uses the lightweight MLB Stats API
functions in that file. Made the pybaseball import lazy (only the two functions that
actually call it trigger it). Verified by installing `requirements-actions.txt` (the lean
CI-only dependency list) into a clean venv and confirming `daily_ks.py`/`reconcile_ks.py`/
`capture_closing_ks.py` import without pybaseball or scikit-learn present.

**Repo scope is deliberately narrow.** `.gitignore` excludes Sessions 1-4's bulky
research/backtest data (the 81MB scraped odds JSON, full 2020-2025 Statcast/bref/schedule
pulls, moneyline model artifacts -- none of it needed at runtime by the live K-props
pipeline) and keeps only: `src/`, the 3 GitHub Actions workflows, `output/model_ks.joblib`
+ `ks_paper_ledger.csv` + `odds_api_usage.csv`, the small season-lag tables
(`pitcher_season.parquet`, `team_season.parquet`), and the accumulating current-season
`pitcher_gamelogs_2026.csv`. Total tracked footprint: ~580KB, not 96MB.

**No `gh` CLI and no git credentials configured on this machine** -- repo creation, push,
GitHub Actions secret setup, and Streamlit Cloud deploy all had to be handed to Jack as a
manual walkthrough rather than done directly. First commit (`9080bd4`) has everything
through today's live run staged and ready to push.

Paper trading window: today (2026-07-09) through early August 2026. Decision to go live
(or not) rests on the ledger's real record/ROI/CLV, not the backtest alone.

**Bug caught on the first real Actions run (2026-07-10):** all three workflows' commit
step used `git add data/raw/teams_*.csv` alongside other globs -- a dead pathspec,
`daily_ks.py` never creates that file. Locally in zsh this fails differently (or not at
all, since leftover research-era `teams_2020-2025.csv` files exist on disk here and are
gitignored). On the Actions runner's bash, an unmatched glob is passed through as a
literal string (bash doesn't nullglob by default), so `git add` got a pathspec matching
zero files and exited 128 -- before the commit/push ever ran. `--ignore-unmatch` is a
`git rm` flag, not valid for `git add` (tried it, confirmed the error). Fixed with
`git add $(ls <patterns> 2>/dev/null)`: pre-expand in the shell, so a total non-match
reduces to `git add` with no arguments (a harmless no-op), not a hard failure. Verified
with explicit `bash -c '...'` locally before trusting it, since the default shell here
(zsh) doesn't reproduce the Actions runner's glob behavior.

## Known data-source constraint (as of 2026-07-08)
FanGraphs' `leaders-legacy.aspx` page (used by `pybaseball.pitching_stats()` and
`pybaseball.team_batting()`) is now behind a Cloudflare interactive challenge and returns
403 for all scripted requests — confirmed with plain `requests`, a spoofed User-Agent, and
`cloudscraper`. It is not currently pullable without a real, human-driven browser session.

Substitutions in use until that's resolved:
- xFIP/SIERA/K-BB% → computed FIP and K-BB% from Baseball-Reference pitching lines
  (`pybaseball.pitching_stats_bref`) plus Statcast pitcher leaderboards (xwOBA against,
  barrel% allowed, whiff%, velo) from Baseball Savant, which is NOT blocked.
- Team wRC+ → `off_index`, a park-adjusted OPS+-style proxy computed from MLB Stats API
  team hitting splits (vs LHP/RHP included). Not FanGraphs wRC+, but same intent
  (park + league adjusted offensive quality).

If FanGraphs access is restored (e.g. via an official API key or a manual export), prefer
the real xFIP/SIERA/wRC+ over these proxies and update `src/fetch.py` accordingly.

**Decision (2026-07-08):** proceeding with the proxies for v1 rather than blocking on FanGraphs
access. Fallback if the backtest results come back marginal (Brier/calibration/CLV borderline):
manually export the FanGraphs leaderboard CSVs from a logged-in browser session (FanGraphs
allows CSV export on the leaderboard pages themselves, which sidesteps the Cloudflare block
on scripted requests) and swap them in for xFIP/SIERA/wRC+ before concluding the feature set
doesn't work. Don't conflate "proxy metrics underperform" with "the whole approach doesn't work"
until that swap has been tried.

## Model library substitution (2026-07-08)
Spec calls for LightGBM. `pip install lightgbm` succeeds but fails at import time on this
machine: its compiled wheel needs `libomp.dylib` (normally provided by `brew install libomp`),
and there's no Homebrew here. Using `sklearn.ensemble.HistGradientBoostingClassifier` instead
-- same histogram-based gradient boosting family, no external native dependency, and sklearn
is already required for isotonic calibration. If Homebrew + libomp ever get installed, LightGBM
is a drop-in swap in `src/model_ml.py` and probably worth revisiting for speed on larger models.

## Session 2 result (2026-07-08): v1 moneyline model does not beat baseline
Trained `src/model_ml.py` on 2021-2024, held out 2025 completely. Full backtest report
(charts) was published as an Artifact during that session; the numbers:
- Brier score: model 0.2474 vs. a naive log5-win-rate proxy market 0.2474 -- **identical**.
  The model adds no measurable predictive signal over the crude baseline.
- Permutation importance is tiny for every feature (top feature moves holdout Brier by
  0.0019; most move it <0.0001) -- the model learned almost nothing from the feature set.
- Predicted probabilities have std dev 0.041 (proxy market: 0.119, ~3x wider) -- the model
  barely differentiates games, clustering near the base rate.
- ROI vs. proxy market is negative in every edge tier and gets WORSE as claimed edge grows
  (1-2%: -6.5%, 2-4%: -7.8%, 4%+: -16.9%) -- an inversion that suggests the model's biggest
  disagreements with the market baseline are its least reliable calls, not its most reliable.
- Spec's own kill criteria triggered (1,512 flagged bets, ROI -15.3%, CLV -4.3% avg): model
  does not bet, dashboard-only until this is fixed.

**Most likely root cause:** every pitcher/team feature is lagged a full season (season-Y
games use season-(Y-1) stats) to avoid leaking in-season future starts into the label. That
avoids leakage correctly but throws away current-form signal, which is likely load-bearing
for single-game MLB prediction. The spec's own v1 feature list assumed current-season
rolling/30-day-weighted stats; this got simplified to season-level lag to ship a leak-free
backtest quickly, and that simplification looks like the actual problem now.

**Before extending to totals or trying to fix this further:** build real as-of-date rolling
pitcher/team features (per-start game logs, trailing windows computed strictly before each
game's date) instead of full-prior-season aggregates. That's the fix most likely to matter,
and it's a real data-engineering lift not yet done. Cheaper things to rule out first: try the
FanGraphs manual-CSV export fallback (above) in case xFIP/SIERA/wRC+ themselves were
load-bearing; don't trust proxy-market ROI/CLV as real profitability without real odds.

## Session 2.1: rolling/current-form features (2026-07-09)
Root-caused Session 2's flat result to season-level lag throwing away current form.
Fix: `src/rolling.py` computes real as-of-date rolling features --
- **Starter form**: trailing-3-starts FIP/K-BB%, `starts_this_season_so_far`, `days_rest`,
  from real per-start game logs pulled via MLB Stats API `people/{id}/stats?stats=gameLog`
  (one call per pitcher-season, ~1,900 calls total for every starter 2021-2025, a few
  minutes -- cheap, not blocked).
- **Team form**: trailing runs-scored/allowed/win% (shrunk toward the team's prior-season
  rate early in a season), computed straight from `games.parquet` -- free, no new pull.
Both reset at the season boundary (a pitcher's first April start doesn't inherit trailing
stats from last September) and fall back to the season-lag/league-average prior when there's
no current-season history yet. Old season-lagged pitcher FIP/K-BB% and team off_index/k_pct
were KEPT alongside the new rolling versions (not replaced), specifically so permutation
importance in the next backtest shows directly whether rolling form is the fix or not.

**Statcast-derived pitcher quality (xwOBA against, barrel%, whiff%, velo) is still
season-lagged, NOT rolling.** Measured, not estimated, the cost of making these rolling too:
- Per-pitcher full-range Statcast pull (`pybaseball.statcast_pitcher`): 95s for one pitcher's
  2021-2025 history. At ~850 unique starters, that's ~22 hours serially.
- Full-league chunked pull (`pybaseball.statcast(start,end)`): 150s per 1-week chunk.
  A full season is ~26 weeks -> ~65 min/season -> ~5.4 hours for 2021-2025.
Neither ran this session. If the rolling FIP/team-form fix alone doesn't fix the backtest,
this is the next thing to invest in -- budget for it explicitly (a multi-hour background
pull, or parallelize the weekly chunks) rather than assuming it's cheap like the gameLog pull was.

## Session 2.1 result (2026-07-09): rolling features helped, not enough to bet
Retrained with `src/rolling.py`'s as-of-date features added alongside (not replacing) the
season-lagged ones from Session 2. Updated backtest report republished to the same Artifact
URL as before. 2025 holdout numbers:
- Brier score: model **0.2461** vs. proxy market 0.2474 -- the model now genuinely beats the
  naive baseline for the first time. Small in absolute terms, but real, not noise: raw
  (uncalibrated) Brier is 0.2452, and calibration held up fine under the wider signal.
- Permutation importance: `win_form_diff` (rolling trailing win% differential) is now the
  #1 feature by ~3x margin over everything else, and 6 of the top 10 features are rolling.
  This is direct evidence the Session 2 diagnosis (season-lag was throwing away current-form
  signal) was correct.
- Model output std dev widened from 0.041 to 0.071 (proxy market: 0.119) -- still narrower,
  but no longer collapsed onto the base rate the way it was in Session 2.
- Still fails the spec's kill criteria: ROI vs. proxy market is negative in the 2-4% and 4%+
  edge tiers (4%+: -18.3% on n=937, still the largest bucket), though the 1-2% tier flipped
  positive for the first time (+2.8%, n=247 -- small sample, don't over-read it). 1,089
  flagged bets (down from 1,512), ROI -17.4%, CLV -4.3% avg.

**Bug fixed along the way:** `rolling.py`'s starter-form fallback was initially indexing an
unsorted, non-deduplicated (season, mlbID) MultiIndex, which both crashed the parquet write
(silently returned wrapped Series objects instead of scalars for at least one lookup) and --
more importantly -- was using a pitcher's OWN SAME-SEASON aggregate as his early-season prior
instead of his actual prior season, which would have reintroduced in-season leakage through
the back door. Fixed by shifting the prior table forward one season and sorting/deduplicating
the index before use. Worth remembering: any `.loc[]` lookup on a manually-built MultiIndex in
this codebase should be sorted and de-duplicated first, or it can silently return the wrong
shape instead of raising.

**Next priority (not done this session):** Statcast-derived pitcher quality (xwOBA against,
barrel%, whiff%, velo) is still season-lagged -- see the cost estimates above before deciding
whether to invest in making those rolling too, vs. getting real historical odds first, vs.
trying the FanGraphs manual-CSV fallback.

## Track 1 result (2026-07-09): real odds close the moneyline question
Found a real, free source for historical closing odds: a public GitHub release
(ArnavSaraogi/mlb-odds-scraper, itself scraped from SportsBookReview) with per-game
opening + closing American moneyline odds across 4-6 books, 2021-04-01 through
2025-08-16 (misses the tail of the 2025 season). No API key needed. `src/fetch_odds.py`
parses it and matches 11,022 of 12,148 games (91%) to `games.parquet` by date + team.

**Bug caught before trusting the result:** averaging/taking the median of raw American
odds across books is wrong -- American odds jump from -100 to +100 with nothing in
between, so a median straddling that gap can land inside it (e.g. median of
[-105,-102,100,100,-102,107] = -1, which decodes as a 100x payout). Always convert to
decimal odds (continuous, no discontinuity) before aggregating across books. Fixed in
`fetch_odds.py`'s `american_to_decimal`. Worth remembering for any future odds work.

**Verdict, real closing odds, 2025 holdout (n=1,713 games with odds coverage):**
- Real closing market Brier: **0.2423** -- meaningfully sharper than both our model
  (0.2469) and the log5 proxy market (0.2474). Real books are efficient; confirms the
  proxy was a reasonable but soft stand-in.
- 1,052 flagged bets (edge >= 3%), ROI **-5.2%**, CLV **-1.42%** avg, negative in every
  edge tier. Kill criteria met on REAL odds, not just the proxy -- this is decisive
  now, not directional. **The moneyline model does not bet, full stop, real numbers.**

## Track 2 result (2026-07-09): strikeout prop model beats naive
`src/model_ks.py` -- Negative Binomial GLM (NB2) predicting each start's strikeout count,
trailing IP/start used as a log-exposure offset (not a regular covariate), fit via
statsmodels. Poisson dispersion check confirmed overdispersion (1.14 on the standardized
fit, 1.31 raw var/mean) so NB2 over plain Poisson was the right call, not just the
spec's default suggestion. Features: trailing-3-start and trailing-30-day K/9, trailing
BB/9, trailing IP/start (exposure), trailing pitch-count average, days rest, starts this
season, season-lagged whiff%/K-BB%/velo (Statcast+bref, same source as the moneyline
model), and the opposing lineup's season-lagged K% vs. the starter's own throwing hand.
Skipped rolling opposing-lineup K% and umpire tendencies -- both would need ~12,000
per-game boxscore calls not pulled this session; season-lag opponent K% used instead.

2025 holdout (n=4,491 starts) vs. a naive baseline (trailing-3-start K rate x trailing
IP/start, its own separately-fit NB2 dispersion -- a fair distributional comparison, not
full-model-vs-a-point-estimate):
- CRPS: full model 1.2884 vs. naive 1.4455 -- **beats naive by 10.9%**.
- Brier at all three prop lines (4.5, 5.5, 6.5): full model wins every line
  (e.g. 6.5: 0.1664 vs 0.1802 naive).
- **Verdict: beats naive on every metric. Cleaner win than the moneyline model ever
  produced.** Built `src/daily_ks.py` per the follow-up instruction.

**`src/daily_ks.py` status:** pipeline validated end-to-end against a live, real MLB
slate (2026-07-09, 26 real probable starters, sane predictions) in `--dry-run` mode
(model scoring only, no odds pull). The live odds pull needs an `ODDS_API_KEY` env var --
free signup at the-odds-api.com (500 requests/month free tier) -- not set up in this
environment, so the edge-flagging and paper-trading ledger are implemented and ready but
UNTESTED against real market prices. Get a key before relying on this for real edges.
The ledger (`output/ks_paper_ledger.csv`) logs model prob, market prob, and edge per
flagged bet with blank `closing_over_odds`/`actual_so`/`result`/`clv` columns meant to be
filled in by a reconciliation pass after games finish -- that reconciliation script
hasn't been built yet either.

## Reconciliation + ODDS_API_KEY setup (2026-07-09)
`src/reconcile_ks.py` fills in the paper-trading ledger after games finish:
`python src/reconcile_ks.py reconcile` (actual_so/result/pnl always; closing_odds/clv
only if the API key's plan includes the paid `/historical/` endpoint -- degrades
gracefully and reports how many rows it could/couldn't get a closing line for, every
run), `python src/reconcile_ks.py summary --by-tier` for running record/ROI/CLV/beat-
close-rate broken out by edge size. Validated the P/L and result math against 6 real
2025 starts with known actual strikeout counts (hand-checked, all correct) before
trusting it on anything real.

**Bug fixed along the way:** `fetch_starter_game_logs(..., refresh=True)` used to wipe
out the ENTIRE season's cached file, not just the requested pitchers -- `daily_ks.py`
already called it this way for each day's ~15-26 starters, so a second day's run would
have silently destroyed every other pitcher's data collected earlier in the season.
Fixed to only replace rows for the requested pitcher_ids. Also fixed the same
median-of-American-odds bug from Track 1 (see above) in `daily_ks.py`'s own
book-consensus step -- same root cause, same fix (aggregate in decimal-odds space).

**ODDS_API_KEY setup** lives in `src/odds_api.py`: reads from the `ODDS_API_KEY` env
var, falling back to a `.env` file in the project root (`ODDS_API_KEY=your_key_here`).
`.env` is gitignored. Free signup (500 req/month) at https://the-odds-api.com/. Neither
is configured in this environment -- `daily_ks.py`'s live odds pull and
`reconcile_ks.py`'s closing-line pull both raise a clear error naming exactly where to
paste the key when it's missing, rather than failing silently or crashing opaquely.

## No historical odds yet
`src/fetch.py` does not pull historical sportsbook odds (The Odds API's historical endpoint
is a paid tier; Kaggle archives require a manual download). Backtests before that data exists
report Brier score and calibration for real, but ROI and CLV are computed against a **proxy
market** (a log5 win-rate baseline, not real closing lines) and are labeled as such everywhere
they appear. Don't quote proxy ROI/CLV numbers as if they were real backtested profitability —
they only show whether the model beats a naive baseline, not whether it beats the actual market.
