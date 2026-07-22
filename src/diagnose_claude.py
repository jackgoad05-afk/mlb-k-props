"""
One-shot diagnostic for the Claude-powered pipelines (daily_research_ks.py,
daily_article_picks.py). When those produce no output but ANTHROPIC_API_KEY is
present, this pinpoints WHICH layer fails by isolating each in turn and printing
the exact exception -- instead of guessing from the outside.

Makes at most two tiny Claude calls (a few tokens each, cost is negligible):
  1. A plain claude-sonnet-5 call, no tools -> tests key validity + model access.
  2. The same model WITH the web_search tool the pipelines use -> tests whether
     web search is enabled/available for this account.

Run locally (needs ANTHROPIC_API_KEY) or via the morning-pull workflow's
manual-dispatch diagnostic step.
"""
from __future__ import annotations

import sys

from daily_article_picks import RESEARCH_MODEL, WEB_SEARCH_TOOL_TYPE
from research_agents import load_anthropic_api_key


def main() -> int:
    try:
        key = load_anthropic_api_key()
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 1
    print(f"key loaded, length {len(key)}")

    import anthropic
    print(f"anthropic SDK version: {anthropic.__version__}")
    client = anthropic.Anthropic(api_key=key)

    # --- Layer 1: plain model call, no tools ---
    print(f"\n[1/2] plain {RESEARCH_MODEL} call (no tools)...")
    try:
        r = client.messages.create(
            model=RESEARCH_MODEL, max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        )
        txt = next((b.text for b in r.content if b.type == "text"), "")
        print(f"  PASS -- model responded: {txt!r}")
    except Exception as e:
        print(f"  FAIL -- {type(e).__name__}: {e}")
        print("  => the model call itself is broken (bad key, no access to "
              f"{RESEARCH_MODEL}, or SDK issue). Web-search test skipped.")
        return 1

    # --- Layer 2: same model WITH the web_search tool the pipelines use ---
    print(f"\n[2/2] {RESEARCH_MODEL} call WITH the {WEB_SEARCH_TOOL_TYPE} tool...")
    try:
        r = client.messages.create(
            model=RESEARCH_MODEL, max_tokens=256,
            tools=[{"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": 1}],
            messages=[{"role": "user", "content": "Search the web for today's date and reply with it."}],
        )
        used_search = any(getattr(b, "type", "") in ("server_tool_use", "web_search_tool_result") for b in r.content)
        txt = next((b.text for b in r.content if b.type == "text"), "")
        print(f"  PASS -- call succeeded (stop_reason={r.stop_reason}, web_search_invoked={used_search})")
        print(f"  model text: {txt[:120]!r}")
        print("\nALL PASS: key, model, and web search all work. If the pipelines still "
              "produce no output, the issue is in response parsing or game selection, not the API.")
        return 0
    except Exception as e:
        print(f"  FAIL -- {type(e).__name__}: {e}")
        print(f"  => the model works but the {WEB_SEARCH_TOOL_TYPE} web-search tool does not. "
              "Most likely web search isn't enabled for this Anthropic account/org, or the tool "
              "type isn't available on this plan. That's why both Claude pipelines produce nothing "
              "-- every call raises here and is caught per-game.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
