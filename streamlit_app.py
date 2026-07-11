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


def why_flagged(r: pd.Series) -> tuple[list[str], str, str | None] | None:
    """Plain-language stat lines, a 1-2 sentence summary, and an optional contradiction
    caution for one flagged bet, built straight from the feature values the model
    scored it with. Returns None if this row predates the columns being captured.

    Each factor is tagged with a category: "rate" (the pitcher's own rolling K/9 and
    whiff%) or "opponent" (opposing lineup's K% vs. his hand). The caution fires only
    when NONE of the pitcher's own rate-stat factors support the bet direction while
    at least one points the other way -- a genuine contradiction, not just one of two
    rate stats being lukewarm. days_rest and mu/line aren't directional votes, they're
    context, so they don't participate in this check.
    """
    why_cols = ["trail_k_per9_3s", "trail_k_per9_30d", "season_lag_whiff_pct", "opp_off_kpct", "days_rest", "mu"]
    if r[why_cols].isna().all():
        return None

    bet_side = r["bet_side"]
    short_name = r["name"].split()[-1]
    stats = []
    factors = []  # list of dicts: category ("rate"/"opponent"), direction ("over"/"under"), clause, short

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
        stats.append(f"Rolling K/9 (last 3 starts): **{k9_3s:.1f}** ({vs_avg}-average{trend_note})")
        if trend == "down" or k9_3s < LEAGUE_AVG_K9:
            factors.append({"category": "rate", "direction": "under", "short": "his trailing K rate",
                             "clause": "his trailing K rate is down" if trend == "down" else "his trailing K rate is below average"})
        if trend == "up" or k9_3s > LEAGUE_AVG_K9:
            factors.append({"category": "rate", "direction": "over", "short": "his trailing K rate",
                             "clause": "his trailing K rate is up" if trend == "up" else "his trailing K rate is above average"})

    whiff = r.get("season_lag_whiff_pct")
    if pd.notna(whiff):
        vs_avg = "above" if whiff > LEAGUE_AVG_WHIFF_PCT else "below"
        stats.append(f"Whiff%: **{whiff:.1f}%** ({vs_avg}-average)")
        direction = "over" if whiff > LEAGUE_AVG_WHIFF_PCT else "under"
        factors.append({"category": "rate", "direction": direction, "short": "his whiff rate",
                         "clause": f"his whiff rate is {vs_avg} average"})

    opp_kpct = r.get("opp_off_kpct")
    if pd.notna(opp_kpct):
        vs_avg = "above" if opp_kpct > LEAGUE_AVG_TEAM_KPCT else "below"
        stats.append(f"Opponent K% vs. his hand: **{opp_kpct:.1%}** ({vs_avg}-average)")
        if opp_kpct < LEAGUE_AVG_TEAM_KPCT:
            factors.append({"category": "opponent", "direction": "under", "short": "opponent contact rate",
                             "clause": "the opposing lineup makes contact at an above-average clip"})
        else:
            factors.append({"category": "opponent", "direction": "over", "short": "opponent strikeout rate",
                             "clause": "the opposing lineup strikes out at an above-average clip"})

    rest = r.get("days_rest")
    if pd.notna(rest):
        note = "standard rest" if rest == STANDARD_REST_DAYS else (
            "extended rest" if rest > STANDARD_REST_DAYS else "short rest")
        stats.append(f"Rest: **{rest:.0f} days** ({note})")

    mu, line = r.get("mu"), r.get("line")
    if pd.notna(mu):
        side_prob = r["model_p_over"] if bet_side == "over" else 1 - r["model_p_over"]
        stats.append(f"Model projects **{mu:.1f}** strikeouts vs. a line of **{line}** "
                     f"({side_prob:.0%} on the {bet_side})")

    matching = [f for f in factors if f["direction"] == bet_side]
    if len(matching) >= 2:
        body = f"{matching[0]['clause']} and {matching[1]['clause']}"
    elif len(matching) == 1:
        body = matching[0]["clause"]
    else:
        body = "the model's overall projection still clears the edge threshold"
    summary = f"For {short_name}, {body}, so the model sees the {bet_side} as more likely than the market prices."

    rate_matching = [f for f in factors if f["category"] == "rate" and f["direction"] == bet_side]
    rate_opposite = [f for f in factors if f["category"] == "rate" and f["direction"] != bet_side]
    caution = None
    if not rate_matching and rate_opposite:
        supporting = [f for f in factors if f["category"] != "rate" and f["direction"] == bet_side]
        support_label = supporting[0]["short"] if supporting else "the model's projected workload/exposure"
        caution = f"⚠️ mixed signals: rate stats point the other way, edge relies mainly on {support_label}."

    return stats, summary, caution


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
        rel_tag = reliability_tag(r["bet_edge"])
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{r['name']}**")
                st.caption(f"vs {r['opponent_name']}")
            with c2:
                st.markdown(f"### +{r['bet_edge']:.1%}")
            st.markdown(f"**{r['bet_side'].upper()} {r['line']}** strikeouts &nbsp;&nbsp; "
                        f"model {model_p:.1%} vs. market {market_p:.1%}")
            st.caption(f"price {price:+.0f} · {r['n_books']} book(s) · "
                       f"{'captured' if pd.notna(r['closing_over_odds']) else 'no closing line yet'}"
                       + (f" · {rel_tag}" if rel_tag else ""))

            why = why_flagged(r)
            if why:
                stats, summary, caution = why
                if caution:
                    st.markdown(f"<span style='color:#fab219; font-size:0.9em;'>{caution}</span>",
                                unsafe_allow_html=True)
                with st.expander("Why?"):
                    for line_txt in stats:
                        st.markdown(f"- {line_txt}")
                    st.caption(summary)

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
st.caption("Paper trading only. Not betting advice.")
