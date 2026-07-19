"""
Read-only clients for Polymarket (Gamma + CLOB) and Kalshi -- both free, no API key,
no auth needed for any of the calls below. Used by daily_pm.py to compare model_ml's
moneyline probabilities against real prediction-market prices (see CLAUDE.md: the
model already failed its kill criteria against real SHARP sportsbook closes -- this
asks whether it finds real edges against these thinner, less efficient markets instead).

Polymarket: Gamma API (gamma-api.polymarket.com) is the metadata/discovery layer --
events, markets, outcomes, clobTokenIds, and (conveniently) the current touch
bid/ask/spread already embedded on each market object, no extra call needed for that.
The CLOB API (clob.polymarket.com) is the order-book/pricing layer, keyed by
token_id -- used here only for full depth beyond the touch price.

Kalshi: external-api.kalshi.com/trade-api/v2. Series KXMLBGAME (moneyline) and
KXMLBTOTAL (totals) are the two of interest. /markets already returns yes/no
bid+ask (dollars) and top-of-book size directly; /markets/{ticker}/orderbook gives
full depth (bids only, both sides -- sufficient, since a market's YES ask is always
1 - NO bid and vice versa in this binary-contract structure).

Confirmed live 2026-07-13 (during the All-Star break -- see daily_pm.py's module
docstring for why real per-game markets are thin/absent right now).
"""
from __future__ import annotations

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"

MLB_SERIES_ID = "3"  # Polymarket Gamma series id for MLB (gamma-api.polymarket.com/series?slug=mlb)
KALSHI_MONEYLINE_SERIES = "KXMLBGAME"
KALSHI_TOTALS_SERIES = "KXMLBTOTAL"


def _get(url: str, params: dict | None = None) -> dict | list:
    r = requests.get(url, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------

def get_polymarket_mlb_events(active: bool = True, closed: bool = False, limit: int = 200) -> list[dict]:
    """One page of MLB events (each with a nested `markets` array covering
    moneyline/totals/spreads/nrfi for that game). 200 is comfortably above a
    full day's MLB slate (max ~15 games x ~1 event each)."""
    return _get(f"{GAMMA_BASE}/events", {
        "series_id": MLB_SERIES_ID, "active": str(active).lower(),
        "closed": str(closed).lower(), "limit": limit,
    })


def _outcomes_and_prices(market: dict) -> list[tuple[str, float]]:
    """Gamma stores `outcomes` and `outcomePrices` as JSON-encoded string lists,
    e.g. '["New York Mets","Philadelphia Phillies"]' / '["0.5","0.5"]'."""
    import json
    outcomes = json.loads(market["outcomes"])
    prices = json.loads(market["outcomePrices"])
    return list(zip(outcomes, [float(p) for p in prices]))


def parse_polymarket_moneyline(event: dict) -> dict | None:
    """Returns {event_slug, teams: {team_name: {token_id, best_bid, best_ask}}}
    for the one moneyline-tagged market in this event, or None if there isn't one."""
    for m in event.get("markets", []):
        if m.get("sportsMarketType") != "moneyline":
            continue
        token_ids = __import__("json").loads(m.get("clobTokenIds", "[]"))
        outcomes = __import__("json").loads(m["outcomes"])
        if len(token_ids) != len(outcomes):
            continue
        teams = {}
        for name, token_id in zip(outcomes, token_ids):
            teams[name] = {
                "token_id": token_id,
                "best_bid": m.get("bestBid"),
                "best_ask": m.get("bestAsk"),
            }
        return {"event_slug": event.get("slug"), "market_id": m.get("id"),
                "game_start_time": m.get("gameStartTime"), "teams": teams}
    return None


def parse_polymarket_totals(event: dict) -> list[dict]:
    """One dict per offered O/U line: {event_slug, line, teams: {outcome: {...}}}.
    Polymarket structures totals as its own moneyline-shaped market per line (e.g.
    'O/U 8.5' with outcomes ["Over","Under"]), not a single market with many lines."""
    import json
    out = []
    for m in event.get("markets", []):
        if m.get("sportsMarketType") != "totals":
            continue
        question = m.get("question", "")
        line = None
        if "O/U" in question:
            try:
                line = float(question.rsplit("O/U", 1)[1].strip())
            except ValueError:
                line = None
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        outcomes = json.loads(m["outcomes"])
        if len(token_ids) != len(outcomes):
            continue
        sides = {name: {"token_id": tid, "best_bid": m.get("bestBid"), "best_ask": m.get("bestAsk")}
                 for name, tid in zip(outcomes, token_ids)}
        out.append({"event_slug": event.get("slug"), "market_id": m.get("id"), "line": line, "teams": sides})
    return out


def get_polymarket_orderbook(token_id: str) -> dict:
    """{bids:[{price,size}...], asks:[{price,size}...]} -- size in shares, $1 notional each."""
    return _get(f"{CLOB_BASE}/book", {"token_id": token_id})


# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------

def get_kalshi_markets(series_ticker: str, status: str = "open", limit: int = 200) -> list[dict]:
    d = _get(f"{KALSHI_BASE}/markets", {"series_ticker": series_ticker, "status": status, "limit": limit})
    return d.get("markets", [])


def get_kalshi_mlb_moneyline_markets() -> list[dict]:
    return get_kalshi_markets(KALSHI_MONEYLINE_SERIES)


def get_kalshi_mlb_totals_markets() -> list[dict]:
    return get_kalshi_markets(KALSHI_TOTALS_SERIES)


def get_kalshi_orderbook(ticker: str) -> dict:
    """Confirmed live schema (2026-07-13, differs from the docs summary): top-level
    key is `orderbook_fp`, with `yes_dollars`/`no_dollars`, each a list of
    [price_dollars_str, size_str] pairs, price already in dollars (not cents) and
    size in contracts ($1 notional each). Bids only, both sides -- a market's YES
    ask at price p is definitionally a NO bid at (1 - p), so this fully describes
    the book without a separate ask array."""
    return _get(f"{KALSHI_BASE}/markets/{ticker}/orderbook")


# ---------------------------------------------------------------------------
# Shared: order-book depth near the touch price, in real dollars
# ---------------------------------------------------------------------------

def usd_depth_polymarket(book: dict, side: str, touch_price: float, slippage: float = 0.01) -> float:
    """side='asks' to check buying depth. Sums price*size (shares, $1 notional
    each) for levels within `slippage` of the touch price."""
    levels = book.get(side, [])
    total = 0.0
    for lvl in levels:
        price, size = float(lvl["price"]), float(lvl["size"])
        if abs(price - touch_price) <= slippage:
            total += price * size
    return total


def usd_depth_kalshi_buy(opposite_side_levels: list[list[str]], buy_touch_price: float, slippage: float = 0.01) -> float:
    """Depth available to BUY one side of a Kalshi market. Kalshi's orderbook only
    carries bids (see get_kalshi_orderbook) -- a resting NO bid at price q is the
    same thing as a YES ask at (1 - q) for the same contract size, and vice versa.
    So to price buying YES, pass the `no_dollars` levels (and buy_touch_price =
    yes_ask_dollars); to price buying NO, pass `yes_dollars` (and buy_touch_price =
    no_ask_dollars). Dollar cost per level is size * (1 - price), NOT size * price
    -- that (1 - price) IS the buy-side price once you take the complement."""
    total = 0.0
    for price, size in opposite_side_levels:
        implied_ask = 1.0 - float(price)
        if abs(implied_ask - buy_touch_price) <= slippage:
            total += implied_ask * float(size)
    return total
