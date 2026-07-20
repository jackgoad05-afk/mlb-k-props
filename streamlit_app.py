"""
K props paper-trading dashboard. Deployed on Streamlit Community Cloud, reading
output/ks_paper_ledger.csv straight from this repo -- the GitHub Actions workflows
(.github/workflows/) keep that file current, Streamlit Cloud auto-redeploys on every
push, so this always reflects the latest committed ledger state with no separate
data pipeline of its own.

Password gate: set an APP_PASSWORD secret in the Streamlit Cloud app's Settings ->
Secrets (not in this repo). See the deploy walkthrough for exact steps.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
LEDGER_PATH = ROOT / "output" / "ks_paper_ledger.csv"
EDGE_TIER_BOUNDS = [(0.03, 0.05), (0.05, 0.10), (0.10, 1.01)]

# Rough, stable MLB-wide reference points -- used only to characterize a raw feature
# value as "above/below average" in plain language. Not recomputed per-day; the raw
# feature values themselves (trail_k_per9_3s, etc.) are the real, precise numbers the
# model actually scored with.
LEAGUE_AVG_K9 = 8.5
LEAGUE_AVG_WHIFF_PCT = 25.0
LEAGUE_AVG_TEAM_KPCT = 0.22
STANDARD_REST_DAYS = 5


def why_flagged(r: pd.Series) -> tuple[str, list[str]] | None:
    """Prose-based explanation of a flagged K-props bet, with detailed stat lines
    available via a toggle. Returns (prose_narrative, stat_detail_lines) or None
    if this row predates the columns being captured.

    Prose narrative:
    - Leads with the strongest directional factor
    - Explains why it matters (mechanism, not just the number)
    - Connects related factors naturally
    - Works in mixed signals and cautions as prose, not disclaimers
    - Only cites factors pointing the same direction as the bet
    """
    why_cols = ["trail_k_per9_3s", "trail_k_per9_30d", "season_lag_whiff_pct", "opp_off_kpct", "days_rest", "mu"]
    if r[why_cols].isna().all():
        return None

    bet_side = r["bet_side"]
    short_name = r["name"].split()[-1]

    # Collect all factors with direction and supporting data
    factors = []  # list of dicts: {"factor": name, "direction": "over"/"under", "strength": score, "detail": text}
    detail_lines = []

    k9_3s, k9_30d = r.get("trail_k_per9_3s"), r.get("trail_k_per9_30d")
    if pd.notna(k9_3s):
        trend = None
        if pd.notna(k9_30d):
            if k9_3s > k9_30d + 0.3:
                trend = "up"
            elif k9_3s < k9_30d - 0.3:
                trend = "down"
        vs_avg = "above" if k9_3s > LEAGUE_AVG_K9 else "below"
        trend_note = f", trending {trend} from his 30-day rate of {k9_30d:.1f}" if trend else ""
        detail_lines.append(f"Rolling K/9 (last 3 starts): **{k9_3s:.1f}** ({vs_avg}-average{trend_note})")

        if trend == "down" or k9_3s < LEAGUE_AVG_K9:
            factors.append({"factor": "k9_low", "direction": "under", "strength": 1,
                           "detail": f"his K/9 is {vs_avg} league average"})
        if trend == "up" or k9_3s > LEAGUE_AVG_K9:
            factors.append({"factor": "k9_high", "direction": "over", "strength": 2 if trend else 1,
                           "detail": f"his K/9 is {vs_avg} league average" + (f" and trending {trend}" if trend else "")})

    whiff = r.get("season_lag_whiff_pct")
    if pd.notna(whiff):
        vs_avg = "above" if whiff > LEAGUE_AVG_WHIFF_PCT else "below"
        detail_lines.append(f"Whiff%: **{whiff:.1f}%** ({vs_avg}-average)")
        direction = "over" if whiff > LEAGUE_AVG_WHIFF_PCT else "under"
        factors.append({"factor": "whiff", "direction": direction, "strength": 2,
                       "detail": f"whiff rate {vs_avg} league average at {whiff:.1f}%"})

    opp_kpct = r.get("opp_off_kpct")
    if pd.notna(opp_kpct):
        vs_avg = "above" if opp_kpct > LEAGUE_AVG_TEAM_KPCT else "below"
        detail_lines.append(f"Opponent K% vs. his hand: **{opp_kpct:.1%}** ({vs_avg}-average)")
        if opp_kpct < LEAGUE_AVG_TEAM_KPCT:
            factors.append({"factor": "opp_k", "direction": "under", "strength": 1,
                           "detail": "the opposing lineup makes contact at above-average rate"})
        else:
            factors.append({"factor": "opp_k", "direction": "over", "strength": 1,
                           "detail": "the opposing lineup strikes out at above-average rate"})

    rest = r.get("days_rest")
    if pd.notna(rest):
        note = "standard rest" if rest == STANDARD_REST_DAYS else (
            "extended rest" if rest > STANDARD_REST_DAYS else "short rest")
        detail_lines.append(f"Rest: **{rest:.0f} days** ({note})")

    mu, line = r.get("mu"), r.get("line")
    if pd.notna(mu):
        side_prob = r["model_p_over"] if bet_side == "over" else 1 - r["model_p_over"]
        detail_lines.append(f"Model projects **{mu:.1f}** strikeouts vs. a line of **{line}** "
                           f"({side_prob:.0%} on the {bet_side})")

    # Build prose narrative from factors pointing the same direction as the bet
    matching = [f for f in factors if f["direction"] == bet_side]
    matching = sorted(matching, key=lambda x: x["strength"], reverse=True)  # strongest first

    rate_factors = [f for f in factors if f["factor"] in ("k9_high", "k9_low", "whiff")]
    rate_matching = [f for f in rate_factors if f["direction"] == bet_side]
    rate_opposite = [f for f in rate_factors if f["direction"] != bet_side]

    # Generate prose
    if not matching:
        prose = (f"{short_name}'s edge relies on the model's overall projection "
                f"clears the {bet_side} threshold more than the market prices.")
    elif len(matching) == 1:
        f = matching[0]
        if f["factor"] == "whiff":
            prose = (f"{short_name}'s whiff rate has climbed to {whiff:.1f}%, well above league average, "
                    f"which historically correlates with elevated strikeout production. The model projects "
                    f"a higher strikeout count than the market is pricing, so the {bet_side} looks "
                    f"undervalued on the margin.")
        elif "k9" in f["factor"]:
            prose = (f"{short_name}'s trailing K/9 is {vs_avg} league average at {k9_3s:.1f}, "
                    f"and the model factors this into a {bet_side} edge over market pricing.")
        elif f["factor"] == "opp_k":
            prose = (f"The opposing lineup's strikeout rate is {vs_avg} league average, making them "
                    f"a {('suitable matchup' if bet_side == 'over' else 'tough matchup')} for {short_name}. "
                    f"The model sees {bet_side} as the edge here.")
        else:
            prose = (f"The model sees {short_name} as a {bet_side} play based on {f['detail']}, "
                    f"with an edge of {r['bet_edge']:.1%} vs. market pricing.")
    else:
        # Multiple factors: lead with strongest, connect the dots
        strongest = matching[0]
        seconds = matching[1:]

        if strongest["factor"] == "whiff":
            lead = f"{short_name}'s whiff rate has climbed to {whiff:.1f}%, the highest of his recent starts"
        elif "k9" in strongest["factor"]:
            lead = f"{short_name}'s trailing K/9 is running {vs_avg} league average at {k9_3s:.1f}"
        elif strongest["factor"] == "opp_k":
            lead = f"The opposing lineup strikes out at a {vs_avg}-league-average rate"
        else:
            lead = f"Multiple factors align on the {bet_side}"

        if rest and pd.notna(rest):
            rest_txt = f"with {rest:.0f} days rest" if rest > STANDARD_REST_DAYS else f"on {note}"
            prose = (f"{lead}, {rest_txt}. Both work together to project higher strikeout production. "
                    f"The model's {bet_side} edge here is real, though keep in mind "
                    f"{'the rest advantage may not stick around for his next start.' if rest > STANDARD_REST_DAYS else 'short rest sometimes catches up.'}")
        else:
            prose = (f"{lead}. This is the real driver of the {bet_side} edge. "
                    f"The model projects {short_name} for a {bet_side} strikeout count vs. how the market is priced.")

    # Add caution if there are mixed signals
    if rate_opposite and not rate_matching:
        prose += f" Caution: the underlying rate stats actually point the other way — the edge relies mainly on the model's overall projection, not on his recent form."

    return prose, detail_lines


def edge_badge_class(edge: float) -> str:
    """Return CSS class name for edge-tier color coding."""
    if edge >= 0.20:
        return "extreme"  # red: highest-risk picks
    if 0.12 <= edge < 0.20:
        return "large"  # amber: large, flashy edges
    return "modest"  # green: modest, historically-stable edges


def reliability_tag(edge: float) -> str | None:
    """Edge-size caution, carried over as a prior from the MONEYLINE model's backtest
    (Session 2/2.1): its largest edge tier was its least reliable, ROI got worse as
    claimed edge grew. Not yet independently verified for the K-props model -- we have
    no historical strikeout-prop market odds to backtest edge-vs-market against, only
    the naive-baseline CRPS/Brier comparison, which isn't edge-tiered. Treat this as a
    reasonable caution, not a confirmed K-props-specific finding."""
    if edge >= 0.20:
        return "⚠️ large edge — historically less reliable"
    if 0.05 <= edge <= 0.12:
        return "✓ modest, stable edge"
    return None

st.set_page_config(page_title="K Props", page_icon="⚾", layout="centered")

# Custom CSS for visual redesign: card layout, typography hierarchy, edge-tier badges
st.markdown("""
<style>
[data-testid="stContainer"] > div > div > div {
    gap: 0.5rem;
}

.flag-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.flag-header {
    display: flex;
    justify-content: space-between;
    align-items: start;
    margin-bottom: 12px;
}

.pitcher-name {
    font-size: 18px;
    font-weight: 600;
    color: #1a1a1a;
}

.matchup {
    font-size: 13px;
    color: #666;
    margin-top: 4px;
}

.edge-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 16px;
    white-space: nowrap;
}

.edge-badge.modest {
    background: #d4edda;
    color: #155724;
}

.edge-badge.large {
    background: #fff3cd;
    color: #856404;
}

.edge-badge.extreme {
    background: #f8d7da;
    color: #721c24;
}

.odds-section {
    background: #f9f9f9;
    border: 0.5px solid #e8e8e8;
    border-radius: 8px;
    padding: 12px;
    margin: 12px 0;
    font-size: 14px;
}

.stat-label {
    color: #666;
    font-size: 13px;
}

.stat-value {
    color: #1a1a1a;
    font-weight: 600;
    font-size: 15px;
}

.section-divider {
    border-top: 0.5px solid #e8e8e8;
    margin: 12px 0;
}

.expander-prose {
    line-height: 1.6;
    color: #333;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


def check_password() -> bool:
    if st.session_state.get("authed"):
        return True
    st.title("⚾ K Props")
    pw = st.text_input("Password", type="password", label_visibility="collapsed",
                        placeholder="Password")
    if pw:
        if pw == st.secrets.get("APP_PASSWORD", None):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


if not check_password():
    st.stop()

st.title("⚾ K Props")
st.caption("Paper trading only · model vs. market strikeout props · " + date.today().strftime("%b %d, %Y"))

if not LEDGER_PATH.exists():
    st.info("No flags logged yet -- check back after the morning pull runs.")
    st.stop()

ledger = pd.read_csv(LEDGER_PATH)
ledger["result"] = ledger["result"].astype("object")
today_str = date.today().isoformat()

# --------------------------------------------------------------------------- #
# Today's flags
# --------------------------------------------------------------------------- #

st.header("Today's flags")
todays = ledger[ledger["date"] == today_str].sort_values("bet_edge", ascending=False)

if todays.empty:
    st.write("No edges flagged today.")
else:
    for _, r in todays.iterrows():
        model_p = r["model_p_over"] if r["bet_side"] == "over" else 1 - r["model_p_over"]
        market_p = r["over_prob_fair"] if r["bet_side"] == "over" else 1 - r["over_prob_fair"]
        price = r["over_odds"] if r["bet_side"] == "over" else r["under_odds"]
        badge_class = edge_badge_class(r["bet_edge"])

        # Render as redesigned card
        st.markdown(f"""
        <div class="flag-card">
            <div class="flag-header">
                <div>
                    <div class="pitcher-name">{r['name']}</div>
                    <div class="matchup">vs {r['opponent_name']}</div>
                </div>
                <div class="edge-badge {badge_class}">+{r['bet_edge']:.1%}</div>
            </div>

            <div class="odds-section">
                <div class="stat-label">{r['bet_side'].upper()} {r['line']} strikeouts</div>
                <div class="stat-value">Model {model_p:.1%} vs. Market {market_p:.1%}</div>
            </div>

            <div class="section-divider"></div>

            <div style="font-size: 13px; color: #666;">
                Price {price:+.0f} · {r['n_books']} book(s) ·
                {'✓ closing line captured' if pd.notna(r['closing_over_odds']) else '○ no closing line yet'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        why = why_flagged(r)
        if why:
            prose, detail_lines = why
            with st.expander("📖 Why?"):
                st.markdown(f'<div class="expander-prose">{prose}</div>', unsafe_allow_html=True)
                if detail_lines:
                    with st.expander("See the numbers", expanded=False):
                        for line_txt in detail_lines:
                            st.markdown(f"- {line_txt}")

# --------------------------------------------------------------------------- #
# Pitcher lookup
# --------------------------------------------------------------------------- #

st.header("Pitcher lookup")
scores_path = ROOT / "output" / "ks_daily_scores.csv"
matched_path = ROOT / "output" / "ks_daily_matched.csv"

if not scores_path.exists():
    st.caption("No scoring data yet today -- check back after the morning pull runs.")
else:
    scores = pd.read_csv(scores_path)
    matched = pd.read_csv(matched_path) if matched_path.exists() else pd.DataFrame()
    scores_date = scores["date"].iloc[0] if len(scores) else None
    if scores_date and scores_date != today_str:
        st.caption(f"⚠ showing {scores_date}'s scores -- today's morning pull hasn't run yet.")

    names = sorted(scores["name"].unique().tolist())
    selected = st.selectbox("Search a starting pitcher", options=names,
                             index=None, placeholder="Type a name...")
    if selected:
        row = scores[scores["name"] == selected].iloc[0]
        st.caption(f"vs {row['opponent_name']}")
        st.metric("Projected strikeouts", f"{row['mu']:.1f}")

        pitcher_lines = matched[matched["name"] == selected] if not matched.empty else pd.DataFrame()
        if len(pitcher_lines):
            for _, m in pitcher_lines.sort_values("line").iterrows():
                st.markdown(f"**Line {m['line']}** &nbsp; model {m['model_p_over']:.1%} over / "
                            f"{1 - m['model_p_over']:.1%} under &nbsp;&nbsp; "
                            f"market {m['over_prob_fair']:.1%} over / {1 - m['over_prob_fair']:.1%} under")
                st.caption(f"price over {m['over_odds']:+.0f} / under {m['under_odds']:+.0f} · {m['n_books']} book(s)")
        else:
            st.caption("No market line matched today (no odds yet, or book doesn't offer this pitcher) -- "
                       "model projection at standard lines:")
            for line in [4.5, 5.5, 6.5]:
                p = row.get(f"model_p_over_{line}")
                if pd.notna(p):
                    st.write(f"{line}: {p:.1%} over / {1 - p:.1%} under")

# --------------------------------------------------------------------------- #
# Ledger summary
# --------------------------------------------------------------------------- #

st.header("Ledger")
done = ledger[ledger["result"].notna()].copy()
pending_n = int(ledger["result"].isna().sum())
st.caption(f"{len(ledger)} total flags · {len(done)} reconciled · {pending_n} pending")

if done.empty:
    st.info("No reconciled bets yet -- results fill in after games finish (overnight reconciliation).")
else:
    def tier_stats(df: pd.DataFrame) -> dict:
        n = len(df)
        w = int((df["result"] == df["bet_side"]).sum())
        l_ = int((df["pnl"] < 0).sum())
        p = int((df["result"] == "push").sum())
        roi = df["pnl"].sum() / n if n else float("nan")
        clv = df["clv"].dropna()
        return {"n": n, "record": f"{w}-{l_}-{p}", "roi": roi, "units": df["pnl"].sum(),
                "avg_clv": clv.mean() if len(clv) else float("nan"), "clv_n": len(clv),
                "beat_close": (clv > 0).mean() if len(clv) else float("nan")}

    overall = tier_stats(done)
    c1, c2, c3 = st.columns(3)
    c1.metric("Record", overall["record"])
    c2.metric("ROI", f"{overall['roi']:+.1%}")
    c3.metric("Units", f"{overall['units']:+.2f}u")
    c1, c2 = st.columns(2)
    c1.metric("Avg CLV", f"{overall['avg_clv']:+.2%}" if overall["clv_n"] else "n/a")
    c2.metric("Beat close", f"{overall['beat_close']:.0%}" if overall["clv_n"] else "n/a")

    st.subheader("By edge tier")
    rows = []
    for lo, hi in EDGE_TIER_BOUNDS:
        tier = done[(done["bet_edge"] >= lo) & (done["bet_edge"] < hi)]
        if tier.empty:
            continue
        s = tier_stats(tier)
        label = f"{lo:.0%}-{hi:.0%}" if hi < 1 else f"{lo:.0%}+"
        rows.append({"tier": label, "n": s["n"], "record": s["record"], "ROI": f"{s['roi']:+.1%}",
                     "avg CLV": f"{s['avg_clv']:+.2%}" if s["clv_n"] else "n/a"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ----------------------------------------------------------------------- #
    # CLV trend
    # ----------------------------------------------------------------------- #
    with_clv = done.dropna(subset=["clv"]).sort_values("date").reset_index(drop=True)
    if len(with_clv) >= 2:
        st.subheader("CLV trend")
        with_clv["bet_num"] = range(1, len(with_clv) + 1)
        with_clv["rolling_clv"] = with_clv["clv"].expanding().mean()
        with_clv["outcome"] = with_clv["result"].where(with_clv["result"] == with_clv["bet_side"], "loss")
        with_clv.loc[with_clv["result"] == with_clv["bet_side"], "outcome"] = "win"

        base = alt.Chart(with_clv).encode(x=alt.X("bet_num:Q", title="flagged bet #"))
        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#898781", strokeDash=[3, 3]).encode(y="y:Q")
        points = base.mark_circle(size=70, opacity=0.85).encode(
            y=alt.Y("clv:Q", title="CLV", axis=alt.Axis(format="%")),
            color=alt.Color("outcome:N", scale=alt.Scale(domain=["win", "loss", "push"],
                                                           range=["#0ca30c", "#e66767", "#898781"]),
                             legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["date", "name", alt.Tooltip("clv:Q", format="+.2%"), "result"],
        )
        trend = base.mark_line(color="#3987e5", strokeWidth=2.5).encode(
            y=alt.Y("rolling_clv:Q", axis=alt.Axis(format="%")))
        chart = (zero_line + points + trend).properties(height=280).configure_axis(
            gridColor="#2c2c2a", labelColor="#c3c2b7", titleColor="#c3c2b7"
        ).configure_view(strokeWidth=0)
        st.altair_chart(chart, use_container_width=True)
        st.caption("dots = each bet's CLV · line = running average CLV · dashed = break-even")
    elif len(with_clv) == 1:
        st.caption("Need at least 2 reconciled bets with a captured closing line for a trend.")

st.divider()

# --------------------------------------------------------------------------- #
# Moneyline predictions (separate model, pure prediction -- no odds as an
# input, no betting logic; see daily_ml.py). Deliberately does NOT show the
# market favorite anywhere on this page -- that comparison is tracked
# privately in output/ml_market_comparison.csv for direct inspection only.
# --------------------------------------------------------------------------- #

st.header("🏆 Moneyline Predictions")
st.caption("Paper/tracking only · straight-up picks from team form, matchups, pitcher/bullpen "
           "quality and park factor -- no odds used as a model input")

ml_ledger_path = ROOT / "output" / "ml_predictions_ledger.csv"
if not ml_ledger_path.exists():
    st.caption("No predictions logged yet -- check back after the morning pull runs.")
else:
    ml_ledger = pd.read_csv(ml_ledger_path)
    ml_ledger["correct"] = ml_ledger["correct"].astype("object")
    ml_today = ml_ledger[ml_ledger["date"] == today_str].sort_values("predicted_win_prob", ascending=False)

    st.subheader("Today's picks")
    if ml_today.empty:
        st.write("No predictions logged today.")
    else:
        for _, r in ml_today.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{r['predicted_winner']}**")
                    st.caption(f"{r['away_team_name']} @ {r['home_team_name']}")
                with c2:
                    st.markdown(f"### {r['predicted_win_prob']:.1%}")
                with st.expander("Why?"):
                    for stat in str(r["why_stats"]).split(" | "):
                        st.markdown(f"- {stat}")
                    st.caption(r["why_summary"])

    st.subheader("Leaderboard: straight-up accuracy")
    ml_done = ml_ledger[ml_ledger["correct"].notna()].copy()
    ml_done["correct"] = ml_done["correct"].astype(bool)
    ml_pending = int(ml_ledger["correct"].isna().sum())
    st.caption(f"{len(ml_ledger)} total predictions · {len(ml_done)} reconciled · {ml_pending} pending")

    if ml_done.empty:
        st.info("No reconciled predictions yet -- results fill in after games finish.")
    else:
        wins = int(ml_done["correct"].sum())
        losses = len(ml_done) - wins
        acc = wins / len(ml_done)
        c1, c2 = st.columns(2)
        c1.metric("Record", f"{wins}-{losses}")
        c2.metric("Accuracy", f"{acc:.1%}")

        by_date = ml_done.groupby("date")["correct"].agg(["sum", "count"]).reset_index()
        by_date.columns = ["date", "wins", "n"]
        by_date["accuracy"] = by_date["wins"] / by_date["n"]
        by_date = by_date.sort_values("date")
        by_date["cum_correct"] = by_date["wins"].cumsum()
        by_date["cum_n"] = by_date["n"].cumsum()
        by_date["running_accuracy"] = by_date["cum_correct"] / by_date["cum_n"]

        if len(by_date) >= 2:
            chart = alt.Chart(by_date).mark_line(point=True, color="#3987e5").encode(
                x=alt.X("date:N", title="date"),
                y=alt.Y("running_accuracy:Q", title="running accuracy", axis=alt.Axis(format="%")),
                tooltip=["date", "wins", "n", alt.Tooltip("accuracy:Q", format=".1%")],
            ).properties(height=220).configure_axis(
                gridColor="#2c2c2a", labelColor="#c3c2b7", titleColor="#c3c2b7"
            ).configure_view(strokeWidth=0)
            st.altair_chart(chart, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------- #
# Prediction markets (Polymarket + Kalshi vs. model_ml, sportsbook line shown
# side by side for reference). Paper only -- see daily_pm.py's module docstring
# for why this exists: model_ml already failed its kill criteria against real
# SHARP sportsbook closes (CLAUDE.md, Track 1), this asks whether it finds real
# edges against thinner prediction-market pricing instead. Moneyline only gets a
# real model edge -- no totals model exists yet, so totals rows (if any show up
# in pm_daily_matched.csv) aren't surfaced here.
# --------------------------------------------------------------------------- #

st.header("🔮 Prediction Markets")
st.caption("Paper trading only · model vs. Polymarket/Kalshi vs. sportsbook · moneyline edges ≥3%, "
           "$50+ depth required")

pm_ledger_path = ROOT / "output" / "pm_paper_ledger.csv"
if not pm_ledger_path.exists():
    st.caption("No flags logged yet -- check back after the morning pull runs.")
else:
    pm_ledger = pd.read_csv(pm_ledger_path)
    pm_ledger["correct"] = pm_ledger["correct"].astype("object")
    pm_today = pm_ledger[pm_ledger["date"] == today_str].sort_values("edge", ascending=False)

    st.subheader("Today's flags")
    if pm_today.empty:
        st.write("No edges flagged today.")
    else:
        for _, r in pm_today.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    team = r["home_team_name"] if r["side"] == "home" else r["away_team_name"]
                    st.markdown(f"**{team}** ({r['market'].capitalize()})")
                    st.caption(f"{r['away_team_name']} @ {r['home_team_name']}")
                with c2:
                    st.markdown(f"### +{r['edge']:.1%}")
                st.markdown(f"model {r['model_prob']:.1%} vs. {r['market'].capitalize()} {r['pm_implied_prob']:.1%} "
                            f"@ ${r['pm_price']:.2f}")
                sb_txt = f"{r['sportsbook_prob']:.1%}" if pd.notna(r["sportsbook_prob"]) else "n/a"
                st.caption(f"sportsbook consensus: {sb_txt} · ${r['depth_usd']:.0f} depth near quote")

    st.subheader("Ledger")
    pm_done = pm_ledger[pm_ledger["correct"].notna()].copy()
    pm_pending = int(pm_ledger["correct"].isna().sum())
    st.caption(f"{len(pm_ledger)} total flags · {len(pm_done)} reconciled · {pm_pending} pending")

    if pm_done.empty:
        st.info("No reconciled bets yet -- results fill in after games finish.")
    else:
        pm_done["correct"] = pm_done["correct"].astype(bool)
        STAKE_USD = 50.0

        def pm_tier_stats(df: pd.DataFrame) -> dict:
            n = len(df)
            w = int(df["correct"].sum())
            roi = df["pnl"].sum() / (n * STAKE_USD) if n else float("nan")
            return {"n": n, "record": f"{w}-{n - w}", "hit_rate": w / n if n else float("nan"),
                    "roi": roi, "pnl": df["pnl"].sum()}

        overall = pm_tier_stats(pm_done)
        c1, c2, c3 = st.columns(3)
        c1.metric("Record", overall["record"])
        c2.metric("Hit rate", f"{overall['hit_rate']:.1%}")
        c3.metric("ROI", f"{overall['roi']:+.1%}")

        st.caption("By edge tier")
        tier_rows = []
        for lo, hi in EDGE_TIER_BOUNDS:
            tier = pm_done[(pm_done["edge"] >= lo) & (pm_done["edge"] < hi)]
            if tier.empty:
                continue
            s = pm_tier_stats(tier)
            label = f"{lo:.0%}-{hi:.0%}" if hi < 1 else f"{lo:.0%}+"
            tier_rows.append({"tier": label, "n": s["n"], "record": s["record"],
                               "hit rate": f"{s['hit_rate']:.1%}", "ROI": f"{s['roi']:+.1%}"})
        st.dataframe(pd.DataFrame(tier_rows), hide_index=True, use_container_width=True)

        st.caption("By platform")
        platform_rows = []
        for platform in sorted(pm_done["market"].unique()):
            s = pm_tier_stats(pm_done[pm_done["market"] == platform])
            platform_rows.append({"platform": platform.capitalize(), "n": s["n"], "record": s["record"],
                                   "hit rate": f"{s['hit_rate']:.1%}", "ROI": f"{s['roi']:+.1%}"})
        st.dataframe(pd.DataFrame(platform_rows), hide_index=True, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------- #
# WNBA "why" text -- same discipline as why_flagged above: only cite a factor
# if it points the SAME direction as the pick being described. Built from the
# raw rolling-form values daily_wnba.py persists into the ledger (WHY_COLS),
# not recomputed here (recomputing as-of-date rolling features after the fact
# would risk leakage).
# --------------------------------------------------------------------------- #

WNBA_FACTOR_LABELS = {
    "win_form": "recent win form", "rs_form": "scoring form", "ra_form": "points allowed",
    "trail_win_pct": "record over the last 10 games", "rest_days": "rest",
}


def _fmt_wnba(val, kind: str) -> str:
    if pd.isna(val):
        return "n/a"
    if kind == "pct":
        return f"{val:.1%}"
    if kind == "num1":
        return f"{val:.1f}"
    if kind == "days":
        return f"{val:.0f}d"
    return str(val)


def why_wnba_moneyline(row: pd.Series, side: str) -> tuple[str, list[str]] | None:
    if pd.isna(row.get("home_win_form")):
        return None
    is_home = side == "home"
    def own(col): return row[f"home_{col}"] if is_home else row[f"away_{col}"]
    def opp(col): return row[f"away_{col}"] if is_home else row[f"home_{col}"]
    team_name = row["home_team_name"] if is_home else row["away_team_name"]
    opp_name = row["away_team_name"] if is_home else row["home_team_name"]

    checks = [
        ("win_form", own("win_form") - opp("win_form"), 0.03, "pct", "recent win form"),
        ("rs_form", own("rs_form") - opp("rs_form"), 1.5, "num1", "scoring form"),
        ("ra_form", opp("ra_form") - own("ra_form"), 1.5, "num1", "points allowed"),  # fewer allowed is better
        ("trail_win_pct", own("trail_win_pct") - opp("trail_win_pct"), 0.05, "pct", "10-game record"),
        ("rest_days", own("rest_days") - opp("rest_days"), 2.0, "days", "rest advantage"),
    ]

    detail_lines = []
    supporting = []  # (factor_name, diff, own_val, opp_val, kind)
    for key, diff, threshold, kind, label in checks:
        if pd.isna(diff):
            continue
        own_val = own(key)
        opp_val = opp(key)
        detail_lines.append(f"{label.capitalize()}: {team_name} {_fmt_wnba(own_val, kind)} vs. {opp_name} {_fmt_wnba(opp_val, kind)}")
        if diff > threshold:
            supporting.append((label, diff, own_val, opp_val, kind))

    # Build prose narrative
    if not supporting:
        prose = f"{team_name} is favored by the model's overall read, though no single factor dominates. The edge is subtle."
    elif len(supporting) == 1:
        factor, diff, own_val, opp_val, kind = supporting[0]
        if factor == "recent win form":
            prose = f"{team_name} has been winning at a higher rate recently ({own_val:.1%}) than {opp_name} ({opp_val:.1%}). This win-form advantage is the main reason the model favors them tonight."
        elif factor == "scoring form":
            prose = f"{team_name} has been scoring at {own_val:.1f} points per game in recent play, while {opp_name} is at {opp_val:.1f}. That offensive edge is what's driving the model's favoritism."
        elif factor == "points allowed":
            prose = f"{team_name}'s defense has been sharper recently, giving up {own_val:.1f} points per game compared to {opp_name}'s {opp_val:.1f}. That defensive advantage underlies the pick."
        elif factor == "10-game record":
            prose = f"{team_name} has been much sharper over the last 10 games ({own_val:.1%}) than {opp_name} ({opp_val:.1%}). This recent momentum is what the model is reacting to."
        elif factor == "rest advantage":
            prose = f"{team_name} has {own_val:.0f} days rest coming in, vs. {opp_name}'s {opp_val:.0f} days. The rest advantage could be meaningful here."
        else:
            prose = f"{team_name} is favored, driven primarily by {factor}."
    else:
        # Multiple factors: weave them together naturally
        first_factor = supporting[0][0]
        if first_factor == "recent win form":
            lead = f"{team_name} has been winning at a significantly higher rate ({supporting[0][2]:.1%}) than {opp_name}"
        elif first_factor == "scoring form":
            lead = f"{team_name} has been the stronger offensive team in recent play"
        elif first_factor == "points allowed":
            lead = f"{team_name} has tightened up defensively"
        else:
            lead = f"{team_name} enters with clear form advantages"

        other_factors = [s[0] for s in supporting[1:]]
        if len(other_factors) == 1:
            other_txt = f"and {other_factors[0]}"
        else:
            other_txt = "and " + ", ".join(other_factors[:-1]) + f", and {other_factors[-1]}"

        prose = (f"{lead}. The edge here comes from {', '.join([s[0] for s in supporting[:2]])}—when multiple "
                f"fundamentals align, that usually means the model's conviction is higher. {team_name} should have the advantage.")

    return prose, detail_lines


def why_wnba_totals(row: pd.Series, side: str) -> tuple[str, list[str]] | None:
    if pd.isna(row.get("total_form_avg")) or pd.isna(row.get("line")):
        return None
    line = row["line"]
    proj = row["total_form_avg"]
    home_rs = row["home_rs_form"]
    home_ra = row["home_ra_form"]
    away_rs = row["away_rs_form"]
    away_ra = row["away_ra_form"]
    diff = (proj - line) * (1 if side == "over" else -1)

    detail_lines = [
        f"Combined scoring-form projection: {proj:.1f} vs. a line of {line}",
        f"{row['home_team_name']}: {home_rs:.1f} scored / {home_ra:.1f} allowed per game (recent form)",
        f"{row['away_team_name']}: {away_rs:.1f} scored / {away_ra:.1f} allowed per game (recent form)",
    ]

    if abs(diff) > 3:
        # Strong signal from the form data itself
        is_over = side == "over"
        home_total = home_rs + away_ra
        away_total = away_rs + home_ra
        prose = (f"Both teams' recent scoring and allowing rates project to roughly {proj:.0f} combined points. "
                f"{row['home_team_name']} should score around {home_rs:.0f} against this defense, while {row['away_team_name']} "
                f"projects to {away_rs:.0f}. That totals about {proj:.0f} points, which is {'above' if is_over else 'below'} "
                f"the {line} line. The model leans {side}.")
    else:
        # Weaker signal; model edge comes from regression, not clear form gap
        prose = (f"The scoring-form projection ({proj:.0f} points) is close to the {line} line, so the {side} edge here "
                f"relies mainly on the regression model's judgment rather than a clear gap in raw recent form. "
                f"Keep in mind the totals model is still unproven — treat this one with extra skepticism.")

    return prose, detail_lines


# --------------------------------------------------------------------------- #
# WNBA moneyline + totals (see daily_wnba.py). Paper only. Honest note baked
# into the caption below: the totals model only came out roughly at parity
# with the simplest possible heuristic on the 2025 holdout backtest (see
# model_wnba_totals.py) -- not a confirmed edge yet, unlike moneyline, which
# did beat its proxy. Every game is logged regardless of whether it clears the
# 3% flag threshold, so the ledger's "flagged" bets are the ones actually
# being paper-tracked for ROI.
# --------------------------------------------------------------------------- #

st.header("🏀 WNBA")
st.caption("Paper trading only · moneyline + totals vs. sportsbook consensus · edges ≥3% flagged. "
           "Totals model is unproven (roughly ties a simple heuristic in backtest) -- treat those "
           "edges with more skepticism than moneyline.")

wnba_ledger_path = ROOT / "output" / "wnba_paper_ledger.csv"
if not wnba_ledger_path.exists():
    st.caption("No predictions logged yet -- check back after the morning pull runs.")
else:
    wnba_ledger = pd.read_csv(wnba_ledger_path)
    wnba_ledger["correct"] = wnba_ledger["correct"].astype("object")
    wnba_today = wnba_ledger[wnba_ledger["date"] == today_str]

    st.subheader("Today's picks")
    if wnba_today.empty:
        st.write("No games today.")
    else:
        games_today = wnba_today[["home_team_name", "away_team_name"]].drop_duplicates()
        for _, gm in games_today.iterrows():
            game_rows = wnba_today[(wnba_today["home_team_name"] == gm["home_team_name"]) &
                                    (wnba_today["away_team_name"] == gm["away_team_name"])]
            with st.container(border=True):
                st.markdown(f"**{gm['away_team_name']} @ {gm['home_team_name']}**")
                ml = game_rows[game_rows["market_type"] == "moneyline"]
                home_row, away_row = ml[ml["side"] == "home"], ml[ml["side"] == "away"]
                if len(home_row) and pd.notna(home_row.iloc[0]["model_prob"]):
                    h, a = home_row.iloc[0], away_row.iloc[0]
                    mkt_txt = (f"home {h['market_prob']:.1%} / away {a['market_prob']:.1%}"
                               if pd.notna(h["market_prob"]) else "n/a")
                    flagged_side = "home" if h["flagged"] else ("away" if a["flagged"] else None)
                    flag_txt = f" 🚩 edge on {flagged_side}" if flagged_side else ""
                    st.markdown(f"Moneyline: model home {h['model_prob']:.1%} / away {a['model_prob']:.1%} "
                                f"vs. market {mkt_txt}{flag_txt}")
                    if flagged_side:
                        why = why_wnba_moneyline(h if flagged_side == "home" else a, flagged_side)
                        if why:
                            prose, detail_lines = why
                            team = h["home_team_name"] if flagged_side == "home" else h["away_team_name"]
                            with st.expander(f"Why {team}?"):
                                st.markdown(prose)
                                if detail_lines:
                                    with st.expander("See the numbers", expanded=False):
                                        for line_txt in detail_lines:
                                            st.markdown(f"- {line_txt}")
                totals = game_rows[(game_rows["market_type"] == "totals") & (game_rows["side"] == "over")]
                if len(totals) and pd.notna(totals.iloc[0]["line"]):
                    t = totals.iloc[0]
                    under_row = game_rows[(game_rows["market_type"] == "totals") & (game_rows["side"] == "under")]
                    under_flagged = len(under_row) and bool(under_row.iloc[0]["flagged"])
                    mkt_txt = f"{t['market_prob']:.1%}" if pd.notna(t["market_prob"]) else "n/a"
                    flagged_side = "over" if t["flagged"] else ("under" if under_flagged else None)
                    flag_txt = f" 🚩 edge on {flagged_side}" if flagged_side else ""
                    st.markdown(f"Total {t['line']}: model {t['model_prob']:.1%} over vs. market {mkt_txt}{flag_txt}")
                    if flagged_side:
                        why = why_wnba_totals(t, flagged_side)
                        if why:
                            prose, detail_lines = why
                            with st.expander(f"Why {flagged_side} {t['line']}?"):
                                st.markdown(prose)
                                if detail_lines:
                                    with st.expander("See the numbers", expanded=False):
                                        for line_txt in detail_lines:
                                            st.markdown(f"- {line_txt}")
                if not (len(home_row) and pd.notna(home_row.iloc[0]["model_prob"])) and not len(totals):
                    st.caption("No market line matched yet today.")

    st.subheader("Ledger (flagged bets only)")
    wnba_flagged = wnba_ledger[wnba_ledger["flagged"] == True].copy()  # noqa: E712
    wnba_done = wnba_flagged[wnba_flagged["pnl"].notna()].copy()
    wnba_pending = len(wnba_flagged) - len(wnba_done)
    st.caption(f"{len(wnba_flagged)} total flagged · {len(wnba_done)} reconciled · {wnba_pending} pending")

    if wnba_done.empty:
        st.info("No reconciled flagged bets yet -- results fill in after games finish.")
    else:
        def wnba_tier_stats(df: pd.DataFrame) -> dict:
            n = len(df)
            w = int((df["correct"] == True).sum())  # noqa: E712
            roi = df["pnl"].sum() / n if n else float("nan")
            return {"n": n, "record": f"{w}-{n - w}", "roi": roi, "pnl": df["pnl"].sum()}

        overall = wnba_tier_stats(wnba_done)
        c1, c2, c3 = st.columns(3)
        c1.metric("Record", overall["record"])
        c2.metric("ROI", f"{overall['roi']:+.1%}")
        c3.metric("Units", f"{overall['pnl']:+.2f}u")

        st.caption("By market type")
        mt_rows = []
        for mt in sorted(wnba_done["market_type"].unique()):
            s = wnba_tier_stats(wnba_done[wnba_done["market_type"] == mt])
            mt_rows.append({"market": mt.capitalize(), "n": s["n"], "record": s["record"], "ROI": f"{s['roi']:+.1%}"})
        st.dataframe(pd.DataFrame(mt_rows), hide_index=True, use_container_width=True)

        st.caption("By edge tier")
        wnba_tier_rows = []
        for lo, hi in EDGE_TIER_BOUNDS:
            tier = wnba_done[(wnba_done["edge"] >= lo) & (wnba_done["edge"] < hi)]
            if tier.empty:
                continue
            s = wnba_tier_stats(tier)
            label = f"{lo:.0%}-{hi:.0%}" if hi < 1 else f"{lo:.0%}+"
            wnba_tier_rows.append({"tier": label, "n": s["n"], "record": s["record"], "ROI": f"{s['roi']:+.1%}"})
        st.dataframe(pd.DataFrame(wnba_tier_rows), hide_index=True, use_container_width=True)

st.divider()
st.caption("Paper trading only. Not betting advice.")
