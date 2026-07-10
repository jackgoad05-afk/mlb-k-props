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


def why_flagged(r: pd.Series) -> tuple[list[str], str] | None:
    """Plain-language stat lines + a 1-2 sentence summary for one flagged bet, built
    straight from the feature values the model scored it with. Returns None if this
    row predates the columns being captured (nothing to show)."""
    why_cols = ["trail_k_per9_3s", "trail_k_per9_30d", "season_lag_whiff_pct", "opp_off_kpct", "days_rest", "mu"]
    if r[why_cols].isna().all():
        return None

    bet_side = r["bet_side"]
    short_name = r["name"].split()[-1]
    stats = []
    under_factors, over_factors = [], []

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
            under_factors.append("trailing K rate is down" if trend == "down" else "trailing K rate is below average")
        if trend == "up" or k9_3s > LEAGUE_AVG_K9:
            over_factors.append("trailing K rate is up" if trend == "up" else "trailing K rate is above average")

    whiff = r.get("season_lag_whiff_pct")
    if pd.notna(whiff):
        vs_avg = "above" if whiff > LEAGUE_AVG_WHIFF_PCT else "below"
        stats.append(f"Whiff%: **{whiff:.1f}%** ({vs_avg}-average)")
        (over_factors if whiff > LEAGUE_AVG_WHIFF_PCT else under_factors).append(f"whiff rate is {vs_avg} average")

    opp_kpct = r.get("opp_off_kpct")
    if pd.notna(opp_kpct):
        vs_avg = "above" if opp_kpct > LEAGUE_AVG_TEAM_KPCT else "below"
        stats.append(f"Opponent K% vs. his hand: **{opp_kpct:.1%}** ({vs_avg}-average)")
        if opp_kpct < LEAGUE_AVG_TEAM_KPCT:
            under_factors.append("the opposing lineup makes contact at an above-average clip")
        else:
            over_factors.append("the opposing lineup strikes out at an above-average clip")

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

    factors = under_factors if bet_side == "under" else over_factors
    if len(factors) >= 2:
        body = f"{factors[0]} and {factors[1]}"
    elif len(factors) == 1:
        body = factors[0]
    else:
        body = "the model's overall projection still clears the edge threshold"
    summary = f"{short_name}'s {body}, so the model sees the {bet_side} as more likely than the market prices."

    return stats, summary

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
                       f"{'captured' if pd.notna(r['closing_over_odds']) else 'no closing line yet'}")

            why = why_flagged(r)
            if why:
                stats, summary = why
                with st.expander("Why?"):
                    for line_txt in stats:
                        st.markdown(f"- {line_txt}")
                    st.caption(summary)

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
