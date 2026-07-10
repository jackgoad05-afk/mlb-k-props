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
