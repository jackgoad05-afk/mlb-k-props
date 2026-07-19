"""
Reconciliation for the prediction-market paper ledger (output/pm_paper_ledger.csv).
Moneyline only -- totals rows are never flagged (see daily_pm.py), so there's
nothing to reconcile for them.

    python src/reconcile_pm.py reconcile           # fill in actual_result/correct/pnl
    python src/reconcile_pm.py summary              # overall record/ROI/hit-rate
    python src/reconcile_pm.py summary --by-tier     # broken out by edge tier
    python src/reconcile_pm.py summary --by-platform # broken out by Polymarket vs. Kalshi
    python src/reconcile_pm.py summary --vs-sportsbook  # split by whether the sportsbook
                                                          # consensus agreed with the model on
                                                          # this game -- the actual research
                                                          # question: is this edge real, or
                                                          # just something the sharp market
                                                          # already priced in that the model
                                                          # happens to also be right about?

Paper stake is a fixed $50 per flagged row (matching the "$50+ realistically available"
depth gate in daily_pm.py). Contracts here pay $1 if correct, cost pm_price if you buy
them -- so buying $50 worth at price p gets you 50/p contracts, worth 50/p if it hits,
0 if it doesn't. pnl = 50*(1/p - 1) on a win, -50 on a loss.
"""
from __future__ import annotations

import argparse

import pandas as pd

import fetch
from daily_pm import LEDGER_PATH

EDGE_TIER_BOUNDS = [(0.03, 0.05), (0.05, 0.10), (0.10, 1.01)]
STAKE_USD = 50.0


def fetch_actual_winners(pending: pd.DataFrame) -> dict[int, str]:
    """One schedule pull per season actually needed (cached after the first)."""
    out: dict[int, str] = {}
    pending = pending.copy()
    pending["season"] = pd.to_datetime(pending["date"]).dt.year
    for season in pending["season"].unique():
        games = fetch.fetch_schedule(int(season), refresh=True)
        game_pks = set(pending.loc[pending["season"] == season, "game_pk"].astype(int))
        for _, g in games[games["game_pk"].isin(game_pks)].iterrows():
            if pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
                continue
            winner = "home" if g["home_score"] > g["away_score"] else "away"
            out[int(g["game_pk"])] = winner
    return out


def reconcile():
    if not LEDGER_PATH.exists():
        print(f"no ledger found at {LEDGER_PATH} -- nothing to reconcile yet.")
        return

    ledger = pd.read_csv(LEDGER_PATH)
    ledger["correct"] = ledger["correct"].astype("object")
    ledger["actual_result"] = ledger["actual_result"].astype("object")
    pending = ledger[ledger["actual_result"].isna() & (ledger["market_type"] == "moneyline")].copy()
    if pending.empty:
        print("nothing pending -- every moneyline ledger row already has a result.")
        return

    today = pd.Timestamp.now().date()
    pending["game_date"] = pd.to_datetime(pending["date"]).dt.date
    playable = pending[pending["game_date"] < today]
    n_too_soon = len(pending) - len(playable)
    if playable.empty:
        print(f"{n_too_soon} pending row(s), none playable yet (game date >= today).")
        return

    winners = fetch_actual_winners(playable)

    n_updated, n_no_result_yet = 0, 0
    for idx, row in playable.iterrows():
        winning_side = winners.get(int(row["game_pk"]))
        if winning_side is None:
            n_no_result_yet += 1
            continue  # postponed/suspended/not final upstream yet -- retry next run

        won = winning_side == row["side"]
        price = row["pm_price"]
        pnl = STAKE_USD * (1.0 / price - 1.0) if won else -STAKE_USD

        ledger.loc[idx, "actual_result"] = winning_side
        ledger.loc[idx, "correct"] = bool(won)
        ledger.loc[idx, "pnl"] = pnl
        n_updated += 1

    ledger.to_csv(LEDGER_PATH, index=False)
    print(f"reconciled {n_updated} row(s)  |  {n_too_soon} not yet playable  |  "
          f"{n_no_result_yet} playable but no result upstream yet (retry later)")


def summarize(by_tier: bool = False, by_platform: bool = False, vs_sportsbook: bool = False):
    if not LEDGER_PATH.exists():
        print(f"no ledger found at {LEDGER_PATH}.")
        return
    ledger = pd.read_csv(LEDGER_PATH)
    ledger = ledger[ledger["market_type"] == "moneyline"]
    done = ledger[ledger["correct"].notna()].copy()
    done["correct"] = done["correct"].astype(bool)
    pending_n = ledger["correct"].isna().sum()

    print("=== prediction-market paper ledger (moneyline) ===")
    print(f"total flagged: {len(ledger)}  |  reconciled: {len(done)}  |  pending: {pending_n}")
    if done.empty:
        print("no reconciled bets yet.")
        return

    def report(df: pd.DataFrame, label: str):
        n = len(df)
        if n == 0:
            print(f"{label:24s} n=   0  (no reconciled rows in this cut)")
            return
        w = df["correct"].sum()
        roi = df["pnl"].sum() / (n * STAKE_USD)
        print(f"{label:24s} n={n:4d}  record={w}-{n - w}  hit_rate={w / n:.1%}  "
              f"ROI={roi:+.1%}  pnl=${df['pnl'].sum():+.2f}")

    report(done, "ALL")

    if by_tier:
        print("\nby edge tier:")
        for lo, hi in EDGE_TIER_BOUNDS:
            tier = done[(done["edge"] >= lo) & (done["edge"] < hi)]
            if tier.empty:
                continue
            label = f"{lo:.0%}-{hi:.0%}" if hi < 1 else f"{lo:.0%}+"
            report(tier, label)

    if by_platform:
        print("\nby platform:")
        for platform in sorted(done["market"].unique()):
            report(done[done["market"] == platform], platform)

    if vs_sportsbook:
        print("\nvs. sportsbook consensus (does the model agree with sharp books on this game?):")
        has_sb = done[done["sportsbook_prob"].notna()].copy()
        if has_sb.empty:
            print("no rows with a matched sportsbook line yet.")
            return
        model_favors_home = has_sb["model_prob"] >= 0.5
        sb_favors_home = has_sb["sportsbook_prob"] >= 0.5
        agrees = model_favors_home == sb_favors_home
        report(has_sb[agrees], "model agrees w/ book")
        report(has_sb[~agrees], "model disagrees w/ book")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("reconcile")
    summary_p = sub.add_parser("summary")
    summary_p.add_argument("--by-tier", action="store_true")
    summary_p.add_argument("--by-platform", action="store_true")
    summary_p.add_argument("--vs-sportsbook", action="store_true")
    args = ap.parse_args()

    if args.command == "reconcile":
        reconcile()
    elif args.command == "summary":
        summarize(by_tier=args.by_tier, by_platform=args.by_platform, vs_sportsbook=args.vs_sportsbook)
