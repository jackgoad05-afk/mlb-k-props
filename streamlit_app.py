"""
Sports model dashboard. Deployed on Streamlit Community Cloud, reading straight from
this repo's output/*.csv ledgers -- the GitHub Actions workflows (.github/workflows/)
keep those files current, Streamlit Cloud auto-redeploys on every push, so this always
reflects the latest committed ledger state with no separate data pipeline of its own.

Password gate: set an APP_PASSWORD secret in the Streamlit Cloud app's Settings ->
Secrets (not in this repo). See the deploy walkthrough for exact steps.

Layout: one Overview tab (at-a-glance status across all four systems) plus one tab per
system (K Props, Moneyline, Prediction Markets, WNBA) -- replaces the old single long
scrolling page. Theme is dark-native (see .streamlit/config.toml, base="dark"): card
colors below are chosen to sit correctly on that background, not light-mode colors
dropped onto a dark page.
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


def _html(s: str) -> str:
    """Strip leading whitespace from every line of a multi-line HTML string before
    it reaches st.markdown(..., unsafe_allow_html=True). Markdown treats 4+ leading
    spaces on a line as an indented code block -- harmless when the HTML f-string
    sits at column 0, but this codebase builds cards inside several levels of
    nested Python indentation (for loops inside if/else inside with-tab blocks),
    so the raw f-string's lines inherit 8-30+ spaces of leading whitespace from
    the surrounding code and get rendered as literal text instead of parsed HTML.
    A plain textwrap.dedent() only removes the SHARED prefix across lines, which
    isn't enough here since nested <div> tags are indented relative to each other
    within the same string -- stripping every line individually is what's needed."""
    return "\n".join(line.lstrip() for line in s.strip("\n").split("\n"))

# Rough, stable MLB-wide reference points -- used only to characterize a raw feature
# value as "above/below average" in plain language. Not recomputed per-day; the raw
# feature values themselves (trail_k_per9_3s, etc.) are the real, precise numbers the
# model actually scored with.
LEAGUE_AVG_K9 = 8.5
LEAGUE_AVG_WHIFF_PCT = 25.0
LEAGUE_AVG_TEAM_KPCT = 0.22
LEAGUE_AVG_IP_PER_START = 5.1
STANDARD_REST_DAYS = 5
RESEARCH_NOTES_PATH = ROOT / "output" / "research_agents_notes.csv"


def load_research_notes() -> dict:
    """Load research agent notes by (game_id, market_type, side).
    Returns {(game_id, market_type, side): [(agent, note), ...]}.
    """
    if not RESEARCH_NOTES_PATH.exists():
        return {}
    notes_df = pd.read_csv(RESEARCH_NOTES_PATH)
    grouped = {}
    for _, row in notes_df.iterrows():
        key = (str(row["game_id"]), row["market_type"], row["side"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append((row["agent"], row["note"]))
    return grouped


def research_notes_html(notes: list[tuple[str, str]]) -> str:
    """Render research-agent notes as a distinct informational box -- deliberately
    styled apart from the pick card itself, since these are context/caution the
    agents attach to an already-made model call, never a new pick (see
    research_agents.py's module docstring)."""
    agent_labels = {"lineup": ("👥", "Lineup"), "injury_news": ("🏥", "Injury/news"),
                    "line_movement": ("📊", "Line movement")}
    rows = []
    for agent, note in notes:
        emoji, label = agent_labels.get(agent, ("ℹ️", agent.replace("_", " ").title()))
        rows.append(f'<div class="research-row"><span class="research-tag">{emoji} {label}</span>{note}</div>')
    return _html(f"""
    <div class="research-box">
        <div class="research-header">Research check &middot; pre-closing context, not a new pick</div>
        {''.join(rows)}
    </div>
    """)


def article_alignment_html(alignment: str, article_side: str, stats_side, consensus: str) -> str:
    """Render the article-vs-stats-model alignment block for an article pick card:
    a tag (aligned/contrarian), a one-line articles-say / model-says / verdict, and
    the article-consensus narrative. `alignment` is the explicit "aligned"/"contrarian"/""
    string from the ledger (see daily_article_picks.py -- stored as a string, not a
    bool, to survive CSV round-trips cleanly). An empty alignment means the stats model
    had no pick for this game (e.g. daily_ml.py didn't run), so no tag/verdict is shown."""
    parts = []
    stats_txt = stats_side if (stats_side is not None and str(stats_side) not in ("", "nan")) else "n/a"

    if alignment == "aligned":
        parts.append('<div class="align-tag aligned">✓ Articles + Model aligned</div>')
        verdict = "agree"
    elif alignment == "contrarian":
        parts.append('<div class="align-tag contrarian">⚠ Contrarian — articles vs. model</div>')
        verdict = "disagree"
    else:
        verdict = None

    if verdict is not None:
        parts.append(f'<div class="align-line">Articles say <b>{article_side}</b> · '
                     f'Stats model says <b>{stats_txt}</b> · they <b>{verdict}</b></div>')
    else:
        parts.append(f'<div class="align-line">Articles say <b>{article_side}</b> · '
                     f'Stats model: <b>{stats_txt}</b> (no comparison available)</div>')

    if consensus and str(consensus) not in ("", "nan"):
        parts.append(f'<div class="align-consensus">📰 {consensus}</div>')

    return "".join(parts)


def why_flagged(r: pd.Series) -> tuple[str, list[str]] | None:
    """Prose-based explanation of a flagged K-props bet, with detailed stat lines
    available via a toggle. Returns (prose_narrative, stat_detail_lines) or None
    if this row predates the columns being captured.

    Discipline (do not weaken -- see CLAUDE.md): a factor is only ever cited as
    SUPPORTING evidence for the bet's direction if it genuinely points that way.
    Whiff% specifically is treated as an OVER-ONLY signal -- a below-average whiff
    rate is real information (shown in "see the numbers") but is never cited as
    evidence for an under, since "misses fewer bats than average" is a much
    weaker basis for "will strike out fewer batters" than the pitcher's own
    trailing K/9 or his projected workload (trail_ip_per_start) are. No factor
    ever gets trend language ("climbed", "trending") unless a real trend was
    actually computed -- only K/9 has real 3-start-vs-30-day trend data; whiff,
    opponent K%, and exposure are snapshot comparisons to league average only.

    If the pitcher's own rate stats (K/9, whiff) point the OPPOSITE way from the
    bet while something else (exposure, opponent) is doing the real supporting
    work, the prose says so honestly ("the model likes the X mainly on Y, even
    though Z points the other way") instead of citing the contradictory stat as
    if it agreed, or silently omitting the contradiction.
    """
    why_cols = ["trail_k_per9_3s", "trail_k_per9_30d", "season_lag_whiff_pct", "opp_off_kpct",
                "trail_ip_per_start", "days_rest", "mu"]
    if r.reindex(why_cols).isna().all():
        return None

    bet_side = r["bet_side"]
    short_name = r["name"].split()[-1]

    # factors: {"factor", "category" ("rate"/"exposure"/"opponent"), "direction", "strength"}
    factors = []
    detail_lines = []

    # --- K/9 (pitcher's own rate, has real 3-start-vs-30-day trend data) ---
    k9_3s, k9_30d = r.get("trail_k_per9_3s"), r.get("trail_k_per9_30d")
    k9_vs_avg, k9_trend = None, None
    if pd.notna(k9_3s):
        if pd.notna(k9_30d):
            if k9_3s > k9_30d + 0.3:
                k9_trend = "up"
            elif k9_3s < k9_30d - 0.3:
                k9_trend = "down"
        k9_vs_avg = "above" if k9_3s > LEAGUE_AVG_K9 else "below"
        trend_note = f", trending {k9_trend} from his 30-day rate of {k9_30d:.1f}" if k9_trend else ""
        detail_lines.append(f"Rolling K/9 (last 3 starts): **{k9_3s:.1f}** ({k9_vs_avg}-average{trend_note})")
        if k9_trend == "down" or k9_3s < LEAGUE_AVG_K9:
            factors.append({"factor": "k9_low", "category": "rate", "direction": "under",
                           "strength": 2 if k9_trend == "down" else 1})
        if k9_trend == "up" or k9_3s > LEAGUE_AVG_K9:
            factors.append({"factor": "k9_high", "category": "rate", "direction": "over",
                           "strength": 2 if k9_trend == "up" else 1})

    # --- Whiff% -- over-only signal, no trend data, never claim "climbed" ---
    whiff = r.get("season_lag_whiff_pct")
    if pd.notna(whiff):
        whiff_vs_avg = "above" if whiff > LEAGUE_AVG_WHIFF_PCT else "below"
        detail_lines.append(f"Whiff%: **{whiff:.1f}%** ({whiff_vs_avg}-average)")
        if whiff > LEAGUE_AVG_WHIFF_PCT:
            factors.append({"factor": "whiff", "category": "rate", "direction": "over", "strength": 2})
        # below-average whiff is shown in the numbers but never cited as under-support

    # --- Opponent K% vs. his hand (genuinely bidirectional) ---
    opp_kpct = r.get("opp_off_kpct")
    opp_vs_avg = None
    if pd.notna(opp_kpct):
        opp_vs_avg = "above" if opp_kpct > LEAGUE_AVG_TEAM_KPCT else "below"
        detail_lines.append(f"Opponent K% vs. his hand: **{opp_kpct:.1%}** ({opp_vs_avg}-average)")
        direction = "under" if opp_kpct < LEAGUE_AVG_TEAM_KPCT else "over"
        factors.append({"factor": "opp_k", "category": "opponent", "direction": direction, "strength": 1})

    # --- Projected exposure (trail_ip_per_start) -- the real short/long-outing signal ---
    ip_per_start = r.get("trail_ip_per_start")
    ip_vs_avg = None
    if pd.notna(ip_per_start):
        ip_vs_avg = "above" if ip_per_start > LEAGUE_AVG_IP_PER_START else "below"
        detail_lines.append(f"Trailing IP/start: **{ip_per_start:.1f}** ({ip_vs_avg}-average)")
        direction = "under" if ip_per_start < LEAGUE_AVG_IP_PER_START else "over"
        factors.append({"factor": "exposure", "category": "exposure", "direction": direction, "strength": 2})

    rest = r.get("days_rest")
    if pd.notna(rest):
        rest_note = "standard rest" if rest == STANDARD_REST_DAYS else (
            "extended rest" if rest > STANDARD_REST_DAYS else "short rest")
        detail_lines.append(f"Rest: **{rest:.0f} days** ({rest_note})")

    mu, line = r.get("mu"), r.get("line")
    if pd.notna(mu):
        side_prob = r["model_p_over"] if bet_side == "over" else 1 - r["model_p_over"]
        detail_lines.append(f"Model projects **{mu:.1f}** strikeouts vs. a line of **{line}** "
                           f"({side_prob:.0%} on the {bet_side})")

    def lead_for(f: dict) -> str:
        if f["factor"] in ("k9_high", "k9_low"):
            trend_txt = f", trending {k9_trend}" if k9_trend else ""
            return f"{short_name}'s trailing K/9 is running {k9_vs_avg} league average at {k9_3s:.1f}{trend_txt}"
        if f["factor"] == "whiff":
            return f"{short_name}'s whiff rate is above league average at {whiff:.1f}%"
        if f["factor"] == "opp_k":
            return f"the opposing lineup's strikeout rate vs. his hand is {opp_vs_avg} league average"
        if f["factor"] == "exposure":
            return f"{short_name} has been going {ip_vs_avg} league average in innings per start recently ({ip_per_start:.1f})"
        return f"multiple factors align on the {bet_side}"

    label_for = {"k9_high": "his trailing K/9", "k9_low": "his trailing K/9", "whiff": "his whiff rate",
                 "opp_k": "the opponent's strikeout rate", "exposure": "his projected innings"}

    matching = sorted([f for f in factors if f["direction"] == bet_side], key=lambda x: x["strength"], reverse=True)
    rate_matching = [f for f in matching if f["category"] == "rate"]
    rate_opposing = [f for f in factors if f["category"] == "rate" and f["direction"] != bet_side]

    if not matching:
        # Nothing genuinely supports this direction -- be honest about it, don't fabricate.
        prose = (f"{short_name}'s edge relies on the model's overall projection "
                f"clearing the {bet_side} threshold more than the market prices.")
        if rate_opposing:
            prose += (" Caution: the underlying rate stats actually point the other way — "
                     "this edge relies mainly on the model's overall projection, not his recent form.")
    elif not rate_matching and rate_opposing:
        # The pitcher's own rate stats disagree with the pick, and nothing rate-based
        # supports it either -- say plainly what IS driving it and name the contradiction.
        support_labels = list(dict.fromkeys(label_for[f["factor"]] for f in matching))
        support_desc = support_labels[0] if len(support_labels) == 1 else " and ".join(support_labels)
        contra = rate_opposing[0]
        if contra["factor"] in ("k9_high", "k9_low"):
            contra_txt = f"{short_name}'s trailing K/9 ({k9_3s:.1f}) is actually running {k9_vs_avg} average"
        else:
            contra_txt = f"{short_name}'s whiff rate ({whiff:.1f}%) actually points toward more strikeouts, not fewer"
        prose = f"The model likes the {bet_side} mainly on {support_desc}, even though {contra_txt}."
    elif len(matching) == 1:
        prose = f"{lead_for(matching[0])[0].upper()}{lead_for(matching[0])[1:]}. The model factors this into the {bet_side} edge over market pricing."
    else:
        strongest, second = matching[0], matching[1]
        second_txt = lead_for(second)
        prose = (f"{lead_for(strongest)[0].upper()}{lead_for(strongest)[1:]}, and {second_txt}. "
                f"Together these support the {bet_side}.")

    # Honest mixed-signal note: fires whenever a rate stat genuinely disagrees with the
    # pick, even if another rate stat (or something else) is doing real supporting work.
    # The "not rate_matching and rate_opposing" case above already names the contradiction
    # inline as part of a full-replacement sentence, so skip it here to avoid repeating.
    if rate_matching and rate_opposing:
        contra = rate_opposing[0]
        if contra["factor"] in ("k9_high", "k9_low"):
            contra_txt = f"his trailing K/9 ({k9_3s:.1f}) is actually running {k9_vs_avg} average"
        else:
            contra_txt = f"his whiff rate ({whiff:.1f}%) actually points the other way"
        prose += f" One caution: {contra_txt}, so this isn't a clean sweep across every stat."

    return prose, detail_lines


def edge_badge_class(edge: float) -> str:
    """Return CSS class name for edge-tier color coding."""
    if edge >= 0.20:
        return "extreme"  # red: highest-risk picks
    if 0.12 <= edge < 0.20:
        return "large"  # amber: large, flashy edges
    return "modest"  # green: modest, historically-stable edges


def confidence_badge_class(prob: float) -> str:
    """Moneyline has no betting edge (deliberately excludes market comparison from
    the dashboard -- see the Moneyline tab's caption), so its badge is colored by
    raw model confidence instead of edge size."""
    if prob >= 0.60:
        return "modest"
    if prob >= 0.53:
        return "large"
    return "extreme"


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


st.set_page_config(page_title="Sports Model", page_icon="📊", layout="wide")

# Dark-native design system -- matches .streamlit/config.toml's base="dark" theme
# (backgroundColor #0d0d0d, secondaryBackgroundColor #1a1a19) rather than dropping
# light-mode card colors onto a dark page.
st.markdown("""
<style>
.flag-card {
    background: #171716;
    border: 1px solid #2c2c2a;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 14px;
}

.flag-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 12px;
}

.pitcher-name {
    font-size: 18px;
    font-weight: 600;
    color: #f2f2f0;
}

.matchup {
    font-size: 13px;
    color: #9b9a92;
    margin-top: 2px;
}

.edge-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 15px;
    white-space: nowrap;
}

.edge-badge.modest { background: #12271a; color: #4ade80; }
.edge-badge.large  { background: #2e2308; color: #fbbf24; }
.edge-badge.extreme { background: #301616; color: #f87171; }

.odds-section {
    background: #1e1e1c;
    border: 1px solid #2c2c2a;
    border-radius: 8px;
    padding: 10px 12px;
    margin: 10px 0;
    font-size: 14px;
}

.stat-label {
    color: #9b9a92;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.stat-value {
    color: #f2f2f0;
    font-weight: 600;
    font-size: 15px;
    margin-top: 2px;
}

.section-divider {
    border-top: 1px solid #2c2c2a;
    margin: 10px 0;
}

.meta-line {
    font-size: 13px;
    color: #9b9a92;
}

.expander-prose {
    line-height: 1.6;
    color: #d6d5cd;
    font-size: 14px;
}

.research-box {
    background: #10202e;
    border: 1px solid #1e3a52;
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 10px;
    font-size: 13px;
}

.research-header {
    color: #7dd3fc;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 6px;
}

.research-row {
    color: #c7d6de;
    padding: 3px 0;
    line-height: 1.5;
}

.research-tag {
    color: #7dd3fc;
    font-weight: 600;
    margin-right: 6px;
}

/* Overview tab: system-status cards */
.sys-card {
    background: #171716;
    border: 1px solid #2c2c2a;
    border-radius: 12px;
    padding: 16px 18px;
}

.sys-card-title {
    font-size: 15px;
    font-weight: 600;
    color: #f2f2f0;
    margin-bottom: 10px;
}

.sys-stat-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    font-size: 14px;
    border-top: 1px solid #232322;
}

.sys-stat-row:first-of-type { border-top: none; }

.sys-stat-label { color: #9b9a92; }
.sys-stat-value { color: #f2f2f0; font-weight: 600; }
.sys-stat-value.positive { color: #4ade80; }
.sys-stat-value.negative { color: #f87171; }

/* Article-vs-model alignment tag */
.align-tag { display: inline-block; font-size: 12px; font-weight: 600; padding: 2px 8px;
             border-radius: 6px; margin-bottom: 4px; }
.align-tag.aligned { background: #12271a; color: #4ade80; }
.align-tag.contrarian { background: #2e2308; color: #fbbf24; }
.align-line { font-size: 13px; color: #9b9a92; margin: 2px 0 6px; }
.align-consensus { font-size: 13px; color: #c7d6de; font-style: italic; margin: 2px 0 8px; }
</style>
""", unsafe_allow_html=True)


def check_password() -> bool:
    if st.session_state.get("authed"):
        return True
    st.title("📊 Sports Model")
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

today_str = date.today().isoformat()

st.title("📊 Sports Model")
st.caption("Paper trading only · not betting advice · " + date.today().strftime("%b %d, %Y"))

tab_overview, tab_ks, tab_ml, tab_pm, tab_wnba = st.tabs(
    ["📊 Overview", "⚾ K Props", "🏆 Moneyline", "🔮 Prediction Markets", "🏀 WNBA"]
)

# =============================================================================== #
# Overview -- at-a-glance status across all four systems. Each system's numbers are
# read independently here (small, cheap CSV reads) rather than threaded through from
# the tabs below, so this tab stays decoupled and easy to reason about on its own.
# =============================================================================== #

with tab_overview:
    st.subheader("System status")

    def sys_card(col, icon: str, name: str, rows: list[tuple[str, str, str]]):
        """rows: list of (label, value, css_class_suffix or '')."""
        row_html = "".join(
            f'<div class="sys-stat-row"><span class="sys-stat-label">{label}</span>'
            f'<span class="sys-stat-value {cls}">{value}</span></div>'
            for label, value, cls in rows
        )
        col.markdown(_html(f"""
        <div class="sys-card">
            <div class="sys-card-title">{icon} {name}</div>
            {row_html}
        </div>
        """), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    # K Props
    if LEDGER_PATH.exists():
        ks_ledger = pd.read_csv(LEDGER_PATH)
        ks_today_n = int((ks_ledger["date"] == today_str).sum())
        ks_done = ks_ledger[ks_ledger["result"].notna()]
        if len(ks_done):
            roi = ks_done["pnl"].sum() / len(ks_done)
            sys_card(c1, "⚾", "K Props", [
                ("Today", str(ks_today_n), ""),
                ("Reconciled", str(len(ks_done)), ""),
                ("ROI", f"{roi:+.1%}", "positive" if roi >= 0 else "negative"),
            ])
        else:
            sys_card(c1, "⚾", "K Props", [("Today", str(ks_today_n), ""), ("Reconciled", "0", "")])
    else:
        sys_card(c1, "⚾", "K Props", [("Status", "not started", "")])

    # Moneyline
    ml_path = ROOT / "output" / "ml_predictions_ledger.csv"
    if ml_path.exists():
        ml_ledger = pd.read_csv(ml_path)
        ml_today_n = int((ml_ledger["date"] == today_str).sum())
        ml_done = ml_ledger[ml_ledger["correct"].notna()]
        if len(ml_done):
            acc = ml_done["correct"].astype(bool).mean()
            sys_card(c2, "🏆", "Moneyline", [
                ("Today", str(ml_today_n), ""),
                ("Reconciled", str(len(ml_done)), ""),
                ("Accuracy", f"{acc:.1%}", "positive" if acc >= 0.5 else "negative"),
            ])
        else:
            sys_card(c2, "🏆", "Moneyline", [("Today", str(ml_today_n), ""), ("Reconciled", "0", "")])
    else:
        sys_card(c2, "🏆", "Moneyline", [("Status", "not started", "")])

    # Prediction Markets
    pm_path = ROOT / "output" / "pm_paper_ledger.csv"
    if pm_path.exists():
        pm_ledger = pd.read_csv(pm_path)
        pm_today_n = int((pm_ledger["date"] == today_str).sum())
        pm_done = pm_ledger[pm_ledger["correct"].notna()]
        if len(pm_done):
            roi = pm_done["pnl"].sum() / (len(pm_done) * 50.0)
            sys_card(c3, "🔮", "Prediction Mkts", [
                ("Today", str(pm_today_n), ""),
                ("Reconciled", str(len(pm_done)), ""),
                ("ROI", f"{roi:+.1%}", "positive" if roi >= 0 else "negative"),
            ])
        else:
            sys_card(c3, "🔮", "Prediction Mkts", [("Today", str(pm_today_n), ""), ("Reconciled", "0", "")])
    else:
        sys_card(c3, "🔮", "Prediction Mkts", [("Status", "not started", "")])

    # WNBA
    wnba_path = ROOT / "output" / "wnba_paper_ledger.csv"
    if wnba_path.exists():
        wnba_ledger = pd.read_csv(wnba_path)
        wnba_today_n = int((wnba_ledger["date"] == today_str).sum())
        wnba_flagged = wnba_ledger[wnba_ledger["flagged"] == True]  # noqa: E712
        wnba_done = wnba_flagged[wnba_flagged["pnl"].notna()]
        if len(wnba_done):
            roi = wnba_done["pnl"].sum() / len(wnba_done)
            sys_card(c4, "🏀", "WNBA", [
                ("Today", str(wnba_today_n), ""),
                ("Reconciled", str(len(wnba_done)), ""),
                ("ROI", f"{roi:+.1%}", "positive" if roi >= 0 else "negative"),
            ])
        else:
            sys_card(c4, "🏀", "WNBA", [("Today", str(wnba_today_n), ""), ("Reconciled", "0", "")])
    else:
        sys_card(c4, "🏀", "WNBA", [("Status", "not started", "")])

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
    st.caption("K Props and Prediction Markets track real paper ROI (1-unit / $50 stakes). "
               "Moneyline tracks straight-up accuracy only, no betting logic. WNBA tracks "
               "paper ROI on flagged moneyline + totals edges.")

# =============================================================================== #
# K Props
# =============================================================================== #

with tab_ks:
    st.caption("Model vs. market strikeout props · edges ≥3% flagged")

    if not LEDGER_PATH.exists():
        st.info("No flags logged yet -- check back after the morning pull runs.")
    else:
        ledger = pd.read_csv(LEDGER_PATH)
        ledger["result"] = ledger["result"].astype("object")

        st.subheader("Today's flags")
        todays = ledger[ledger["date"] == today_str].sort_values("bet_edge", ascending=False)

        if todays.empty:
            st.write("No edges flagged today.")
        else:
            research_notes = load_research_notes()
            for _, r in todays.iterrows():
                model_p = r["model_p_over"] if r["bet_side"] == "over" else 1 - r["model_p_over"]
                market_p = r["over_prob_fair"] if r["bet_side"] == "over" else 1 - r["over_prob_fair"]
                price = r["over_odds"] if r["bet_side"] == "over" else r["under_odds"]
                badge_class = edge_badge_class(r["bet_edge"])

                st.markdown(_html(f"""
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

                    <div class="meta-line">
                        Price {price:+.0f} · {r['n_books']} book(s) ·
                        {'✓ closing line captured' if pd.notna(r['closing_over_odds']) else '○ no closing line yet'}
                    </div>
                </div>
                """), unsafe_allow_html=True)

                why = why_flagged(r)
                if why:
                    prose, detail_lines = why
                    with st.expander("📖 Why?"):
                        st.markdown(f'<div class="expander-prose">{prose}</div>', unsafe_allow_html=True)
                        if detail_lines:
                            with st.expander("See the numbers", expanded=False):
                                for line_txt in detail_lines:
                                    st.markdown(f"- {line_txt}")

                notes_key = (str(r.get("game_id", "")), "strikeout_props", r["bet_side"])
                if notes_key in research_notes:
                    st.markdown(research_notes_html(research_notes[notes_key]), unsafe_allow_html=True)

        st.divider()

        # ----------------------------------------------------------------------- #
        # Pitcher lookup
        # ----------------------------------------------------------------------- #

        st.subheader("Pitcher lookup")
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

        st.divider()

        # ----------------------------------------------------------------------- #
        # Ledger summary
        # ----------------------------------------------------------------------- #

        st.subheader("Ledger")
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

            st.markdown("**By edge tier**")
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

            # ------------------------------------------------------------------- #
            # CLV trend
            # ------------------------------------------------------------------- #
            with_clv = done.dropna(subset=["clv"]).sort_values("date").reset_index(drop=True)
            if len(with_clv) >= 2:
                st.markdown("**CLV trend**")
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

        # --------------------------------------------------------------------- #
        # Research model -- separate pipeline (daily_research_ks.py), same edge
        # threshold and reconciliation math, paper-tracked in its own ledger for
        # a real head-to-head comparison against the stats-only model above.
        # --------------------------------------------------------------------- #
        st.divider()
        st.subheader("🔬 Research Model")
        st.caption("Same edge threshold and reconciliation math as the stats model above, but each pick "
                   "also factors in a live web-searched review of pitcher/opponent news (see "
                   "daily_research_ks.py). Paper/tracking only -- a real head-to-head test, not a "
                   "replacement for the stats-only model.")

        research_daily_path = ROOT / "output" / "research_ks_daily.csv"
        research_ledger_path = ROOT / "output" / "research_ks_ledger.csv"

        if not research_daily_path.exists():
            st.caption("No research-model predictions yet -- check back after the pipeline runs.")
        else:
            research_daily = pd.read_csv(research_daily_path)
            research_today = research_daily[research_daily["date"] == today_str].sort_values(
                "bet_edge", ascending=False)

            if research_today.empty:
                st.write("No research picks today.")
            else:
                for _, r in research_today.iterrows():
                    flag_txt = " 🚩" if r["bet_edge"] >= 0.03 else ""
                    st.markdown(f"**{r['name']}** vs {r['opponent_name']}{flag_txt} — "
                                f"projects **{r['projected_strikeouts']:.1f}** Ks vs. line {r['line']}, "
                                f"leans **{r['bet_side']}** (edge {r['bet_edge']:+.1%})")
                    with st.expander("📖 Research reasoning"):
                        st.markdown(f'<div class="expander-prose">{r["reasoning"]}</div>', unsafe_allow_html=True)

        # --------------------------------------------------------------------- #
        # Article-based picks -- third pipeline (daily_article_picks.py). Pure
        # article research, no stats-model numbers shown to Claude at all
        # (contrast with the Research Model above, which explicitly blends the
        # two). Narrowly scoped to the top TOP_N_GAMES games/day by K-props edge
        # size, kept on the SAME specific (pitcher, line) the stats model
        # flagged so this is a genuine same-bet comparison, not just a
        # different pitcher entirely.
        # --------------------------------------------------------------------- #
        st.divider()
        st.subheader("📰 Article-Based Picks")
        st.caption("Pure web-searched article research on the top few games only (see "
                   "daily_article_picks.py) -- Claude never sees the stats model's own numbers here, "
                   "just news/previews. Same specific pitcher+line the stats model flagged, for a "
                   "clean same-bet comparison. Paper/tracking only.")

        article_ks_ledger_path = ROOT / "output" / "article_picks_ks_ledger.csv"
        if not article_ks_ledger_path.exists():
            st.caption("No article-based picks yet -- check back after the pipeline runs.")
        else:
            article_ks_ledger = pd.read_csv(article_ks_ledger_path)
            article_ks_today = article_ks_ledger[article_ks_ledger["date"] == today_str]

            if article_ks_today.empty:
                st.write("No article-based picks today.")
            else:
                for _, r in article_ks_today.iterrows():
                    st.markdown(f"**{r['name']}** vs {r['opponent_name']} — leans **{r['bet_side']} {r['line']}** "
                                f"strikeouts, confidence **{r['confidence']}**")
                    st.markdown(article_alignment_html(str(r.get("alignment", "")), r["bet_side"],
                                                       r.get("stats_model_side"), r.get("article_consensus", "")),
                                unsafe_allow_html=True)
                    with st.expander("📖 Article reasoning"):
                        st.markdown(f'<div class="expander-prose">{r["reasoning"]}</div>', unsafe_allow_html=True)

        st.markdown("**Stats-only vs. stats+research vs. pure article**")
        if not LEDGER_PATH.exists() or not research_ledger_path.exists() or not article_ks_ledger_path.exists():
            st.caption("Need all three ledgers with reconciled results for a side-by-side comparison.")
        else:
            stats_ledger_cmp = pd.read_csv(LEDGER_PATH)
            stats_ledger_cmp["result"] = stats_ledger_cmp["result"].astype("object")
            stats_done_cmp = stats_ledger_cmp[stats_ledger_cmp["result"].notna()]

            research_ledger_cmp = pd.read_csv(research_ledger_path)
            research_ledger_cmp["result"] = research_ledger_cmp["result"].astype("object")
            research_done_cmp = research_ledger_cmp[research_ledger_cmp["result"].notna()]

            article_ledger_cmp = pd.read_csv(article_ks_ledger_path)
            article_ledger_cmp["result"] = article_ledger_cmp["result"].astype("object")
            article_done_cmp = article_ledger_cmp[article_ledger_cmp["result"].notna()]

            def cmp_stats(df: pd.DataFrame) -> dict:
                if df.empty:
                    return {"n": 0, "record": "0-0-0", "roi": float("nan"), "avg_clv": float("nan"), "clv_n": 0}
                n = len(df)
                w = int((df["result"] == df["bet_side"]).sum())
                l_ = int((df["pnl"] < 0).sum())
                p = int((df["result"] == "push").sum())
                clv = df["clv"].dropna()
                return {"n": n, "record": f"{w}-{l_}-{p}", "roi": df["pnl"].sum() / n,
                        "avg_clv": clv.mean() if len(clv) else float("nan"), "clv_n": len(clv)}

            s_stats = cmp_stats(stats_done_cmp)
            r_stats = cmp_stats(research_done_cmp)
            a_stats = cmp_stats(article_done_cmp)

            cs, cr, ca = st.columns(3)
            for col, label, stats in [(cs, "Stats-only", s_stats), (cr, "Stats + research", r_stats),
                                       (ca, "Pure article", a_stats)]:
                with col:
                    st.markdown(f"*{label}*")
                    st.metric("Record", stats["record"])
                    st.metric("ROI", f"{stats['roi']:+.1%}" if stats["n"] else "n/a")
                    st.metric("Avg CLV", f"{stats['avg_clv']:+.2%}" if stats["clv_n"] else "n/a")

            if min(s_stats["n"], r_stats["n"], a_stats["n"]) == 0:
                st.caption("At least one side has no reconciled bets yet -- comparison will fill in as "
                           "all three ledgers accumulate results.")

# =============================================================================== #
# Moneyline predictions (separate model, pure prediction -- no odds as an input, no
# betting logic; see daily_ml.py). Deliberately does NOT show the market favorite
# anywhere on this page -- that comparison is tracked privately in
# output/ml_market_comparison.csv for direct inspection only.
# =============================================================================== #

with tab_ml:
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
                badge_class = confidence_badge_class(r["predicted_win_prob"])
                away_team = r['away_team_name']
                home_team = r['home_team_name']

                st.markdown(_html(f"""
                <div class="flag-card">
                    <div class="flag-header">
                        <div>
                            <div class="pitcher-name">{r['predicted_winner']}</div>
                            <div class="matchup">{away_team} @ {home_team}</div>
                        </div>
                        <div class="edge-badge {badge_class}">{r['predicted_win_prob']:.1%}</div>
                    </div>
                    <div class="meta-line">
                        Straight-up pick based on team form, matchups, pitching quality, and ballpark
                    </div>
                </div>
                """), unsafe_allow_html=True)

                why_stats = str(r.get("why_stats", "")).split(" | ") if pd.notna(r.get("why_stats")) else []
                why_summary = str(r.get("why_summary", ""))
                if why_stats or why_summary:
                    with st.expander("📖 Why?"):
                        if why_summary:
                            st.markdown(f'<div class="expander-prose">{why_summary}</div>', unsafe_allow_html=True)
                        if why_stats and why_stats[0]:
                            with st.expander("See the numbers", expanded=False):
                                for stat in why_stats:
                                    if stat.strip():
                                        st.markdown(f"- {stat}")

        st.divider()
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

    # ----------------------------------------------------------------------- #
    # Article-based moneyline picks -- daily_article_picks.py's second output.
    # Unlike the stats-only model above, this ledger DOES track real PnL/CLV --
    # these are article-driven judgment calls on the same top few games the
    # K-props article picks cover, meant to be compared head-to-head against
    # the other ledgers, not a market-comparison-free straight-up tracker.
    # ----------------------------------------------------------------------- #
    st.divider()
    st.subheader("📰 Article-Based Moneyline Picks")
    st.caption("Pure web-searched article research on the same top few games as the K-props article "
               "picks (see daily_article_picks.py) -- real PnL/CLV tracked here, unlike the stats-only "
               "model above. Paper/tracking only.")

    article_ml_ledger_path = ROOT / "output" / "article_picks_ml_ledger.csv"
    if not article_ml_ledger_path.exists():
        st.caption("No article-based moneyline picks yet -- check back after the pipeline runs.")
    else:
        article_ml_ledger = pd.read_csv(article_ml_ledger_path)
        article_ml_ledger["result"] = article_ml_ledger["result"].astype("object")
        article_ml_today = article_ml_ledger[article_ml_ledger["date"] == today_str]

        st.markdown("**Today's picks**")
        if article_ml_today.empty:
            st.write("No article-based moneyline picks today.")
        else:
            for _, r in article_ml_today.iterrows():
                pick_team = r["home_team_name"] if r["bet_side"] == "home" else r["away_team_name"]
                st.markdown(f"**{r['away_team_name']} @ {r['home_team_name']}** — picks **{pick_team}** "
                            f"to win, confidence **{r['confidence']}**")
                # Translate the stats model's home/away side to a team name for display;
                # the alignment string itself was already computed home/away in the pipeline.
                stats_side_raw = str(r.get("stats_model_side", ""))
                stats_team = (r["home_team_name"] if stats_side_raw == "home"
                              else r["away_team_name"] if stats_side_raw == "away" else None)
                st.markdown(article_alignment_html(str(r.get("alignment", "")), pick_team,
                                                   stats_team, r.get("article_consensus", "")),
                            unsafe_allow_html=True)
                with st.expander("📖 Article reasoning"):
                    st.markdown(f'<div class="expander-prose">{r["reasoning"]}</div>', unsafe_allow_html=True)

        article_ml_done = article_ml_ledger[article_ml_ledger["result"].notna()].copy()
        article_ml_pending = int(article_ml_ledger["result"].isna().sum())
        st.caption(f"{len(article_ml_ledger)} total picks · {len(article_ml_done)} reconciled · "
                   f"{article_ml_pending} pending")

        if article_ml_done.empty:
            st.info("No reconciled picks yet -- results fill in after games finish.")
        else:
            n = len(article_ml_done)
            w = int((article_ml_done["result"] == article_ml_done["bet_side"]).sum())
            roi = article_ml_done["pnl"].sum() / n
            clv = article_ml_done["clv"].dropna()
            c1, c2, c3 = st.columns(3)
            c1.metric("Record", f"{w}-{n - w}")
            c2.metric("ROI", f"{roi:+.1%}")
            c3.metric("Avg CLV", f"{clv.mean():+.2%}" if len(clv) else "n/a")

# =============================================================================== #
# Prediction markets (Polymarket + Kalshi vs. model_ml, sportsbook line shown side by
# side for reference). Paper only -- see daily_pm.py's module docstring for why this
# exists: model_ml already failed its kill criteria against real SHARP sportsbook
# closes (CLAUDE.md, Track 1), this asks whether it finds real edges against thinner
# prediction-market pricing instead.
# =============================================================================== #

with tab_pm:
    st.caption("Model vs. Polymarket/Kalshi vs. sportsbook · moneyline edges ≥3%, $50+ depth required")

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
                badge_class = edge_badge_class(r['edge'])
                team = r["home_team_name"] if r["side"] == "home" else r["away_team_name"]
                market_name = r['market'].capitalize()

                st.markdown(_html(f"""
                <div class="flag-card">
                    <div class="flag-header">
                        <div>
                            <div class="pitcher-name">{team} ({market_name})</div>
                            <div class="matchup">{r['away_team_name']} @ {r['home_team_name']}</div>
                        </div>
                        <div class="edge-badge {badge_class}">+{r['edge']:.1%}</div>
                    </div>

                    <div class="odds-section">
                        <div class="stat-label">{market_name} prediction market</div>
                        <div class="stat-value">Model {r['model_prob']:.1%} vs. {market_name} {r['pm_implied_prob']:.1%} @ ${r['pm_price']:.2f}</div>
                    </div>

                    <div class="section-divider"></div>

                    <div class="meta-line">
                        Sportsbook consensus: {"n/a" if pd.isna(r["sportsbook_prob"]) else f'{r["sportsbook_prob"]:.1%}'} · ${r['depth_usd']:.0f} depth near quote
                    </div>
                </div>
                """), unsafe_allow_html=True)

        st.divider()
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

            st.markdown("**By edge tier**")
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

            st.markdown("**By platform**")
            platform_rows = []
            for platform in sorted(pm_done["market"].unique()):
                s = pm_tier_stats(pm_done[pm_done["market"] == platform])
                platform_rows.append({"platform": platform.capitalize(), "n": s["n"], "record": s["record"],
                                       "hit rate": f"{s['hit_rate']:.1%}", "ROI": f"{s['roi']:+.1%}"})
            st.dataframe(pd.DataFrame(platform_rows), hide_index=True, use_container_width=True)

# =============================================================================== #
# WNBA "why" text -- same discipline as why_flagged above: only cite a factor if it
# points the SAME direction as the pick being described. Built from the raw
# rolling-form values daily_wnba.py persists into the ledger (WHY_COLS), not
# recomputed here (recomputing as-of-date rolling features after the fact would risk
# leakage).
# =============================================================================== #

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
        is_over = side == "over"
        prose = (f"Both teams' recent scoring and allowing rates project to roughly {proj:.0f} combined points. "
                f"{row['home_team_name']} should score around {home_rs:.0f} against this defense, while {row['away_team_name']} "
                f"projects to {away_rs:.0f}. That totals about {proj:.0f} points, which is {'above' if is_over else 'below'} "
                f"the {line} line. The model leans {side}.")
    else:
        prose = (f"The scoring-form projection ({proj:.0f} points) is close to the {line} line, so the {side} edge here "
                f"relies mainly on the regression model's judgment rather than a clear gap in raw recent form. "
                f"Keep in mind the totals model is still unproven — treat this one with extra skepticism.")

    return prose, detail_lines


# =============================================================================== #
# WNBA moneyline + totals (see daily_wnba.py). Paper only. Honest note baked into the
# caption below: the totals model only came out roughly at parity with the simplest
# possible heuristic on the 2025 holdout backtest (see model_wnba_totals.py) -- not a
# confirmed edge yet, unlike moneyline, which did beat its proxy. Every game is logged
# regardless of whether it clears the 3% flag threshold, so the ledger's "flagged"
# bets are the ones actually being paper-tracked for ROI.
# =============================================================================== #

with tab_wnba:
    st.caption("Moneyline + totals vs. sportsbook consensus · edges ≥3% flagged. "
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

                st.markdown(_html(f"""
                <div class="flag-card">
                    <div class="flag-header">
                        <div>
                            <div class="pitcher-name">{gm['away_team_name']}</div>
                            <div class="matchup">@ {gm['home_team_name']}</div>
                        </div>
                    </div>
                </div>
                """), unsafe_allow_html=True)

                ml = game_rows[game_rows["market_type"] == "moneyline"]
                home_row, away_row = ml[ml["side"] == "home"], ml[ml["side"] == "away"]
                if len(home_row) and pd.notna(home_row.iloc[0]["model_prob"]):
                    h, a = home_row.iloc[0], away_row.iloc[0]
                    mkt_txt = (f"home {h['market_prob']:.1%} / away {a['market_prob']:.1%}"
                               if pd.notna(h["market_prob"]) else "n/a")
                    flagged_side = "home" if h["flagged"] else ("away" if a["flagged"] else None)
                    flag_emoji = " 🚩" if flagged_side else ""
                    st.markdown(f"**Moneyline{flag_emoji}** — model home {h['model_prob']:.1%} / away {a['model_prob']:.1%} vs. market {mkt_txt}")
                    if flagged_side:
                        why = why_wnba_moneyline(h if flagged_side == "home" else a, flagged_side)
                        if why:
                            prose, detail_lines = why
                            team = h["home_team_name"] if flagged_side == "home" else h["away_team_name"]
                            with st.expander(f"📖 Why {team}?"):
                                st.markdown(f'<div class="expander-prose">{prose}</div>', unsafe_allow_html=True)
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
                    flag_emoji = " 🚩" if flagged_side else ""
                    st.markdown(f"**Total {t['line']}{flag_emoji}** — model {t['model_prob']:.1%} over vs. market {mkt_txt}")
                    if flagged_side:
                        why = why_wnba_totals(t, flagged_side)
                        if why:
                            prose, detail_lines = why
                            with st.expander(f"📖 Why {flagged_side} {t['line']}?"):
                                st.markdown(f'<div class="expander-prose">{prose}</div>', unsafe_allow_html=True)
                                if detail_lines:
                                    with st.expander("See the numbers", expanded=False):
                                        for line_txt in detail_lines:
                                            st.markdown(f"- {line_txt}")

                if not (len(home_row) and pd.notna(home_row.iloc[0]["model_prob"])) and not len(totals):
                    st.caption("No market line matched yet today.")

                st.divider()

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

            st.markdown("**By market type**")
            mt_rows = []
            for mt in sorted(wnba_done["market_type"].unique()):
                s = wnba_tier_stats(wnba_done[wnba_done["market_type"] == mt])
                mt_rows.append({"market": mt.capitalize(), "n": s["n"], "record": s["record"], "ROI": f"{s['roi']:+.1%}"})
            st.dataframe(pd.DataFrame(mt_rows), hide_index=True, use_container_width=True)

            st.markdown("**By edge tier**")
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
