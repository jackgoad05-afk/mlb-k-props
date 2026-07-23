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
    # The key check: web search errors do NOT raise -- the call returns 200 with an
    # error nested in a web_search_tool_result block (error_code: too_many_requests /
    # unavailable / max_uses_exceeded / ...). So we must inspect the result blocks, not
    # just whether the call raised. This is exactly the "search unavailable" the pipelines
    # hit: the call succeeds, Claude falls back, but the search itself did nothing.
    print(f"\n[2/2] {RESEARCH_MODEL} call WITH the {WEB_SEARCH_TOOL_TYPE} tool...")
    try:
        r = client.messages.create(
            model=RESEARCH_MODEL, max_tokens=256,
            tools=[{"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": 1}],
            messages=[{"role": "user", "content": "Search the web for today's date and reply with it."}],
        )
    except Exception as e:
        print(f"  FAIL (call raised) -- {type(e).__name__}: {e}")
        print("  => if this says web search is not enabled, an admin disabled it at "
              "console.anthropic.com/settings/privacy. Re-enable it there.")
        return 1

    # Look for the search error code inside the result blocks.
    search_error = None
    search_ok = False
    n_results = 0
    for b in r.content:
        if getattr(b, "type", "") == "web_search_tool_result":
            content = b.content
            if isinstance(content, list):
                search_ok = True
                n_results = len(content)
            else:  # single error object
                search_error = getattr(content, "error_code", None) or getattr(content, "type", "error")

    if search_error:
        print(f"  SEARCH ERROR -- the call succeeded but web search returned error_code={search_error!r}")
        if search_error == "too_many_requests":
            print("  => your org's WEB SEARCH RATE LIMIT is being hit (often ~0 on a new/low-tier org). "
                  "Check console.anthropic.com/settings/limits -- if the web search limit is very low, "
                  "request an increase (contact sales from that page). THIS is why the pipelines get "
                  "'search unavailable' and produce no real article picks.")
        elif search_error == "unavailable":
            print("  => transient internal web-search error; retry. If it persists every run, contact support.")
        else:
            print(f"  => web search returned {search_error}; see the web-search error-code docs.")
        return 1

    if search_ok:
        print(f"  PASS -- web search executed and returned {n_results} result(s) (stop_reason={r.stop_reason}).")
        print("\nALL PASS: key, model, and web search all work. If the pipelines still produce no "
              "output, the issue is downstream (parsing/selection), not web search.")
        return 0

    print(f"  INCONCLUSIVE -- call succeeded (stop_reason={r.stop_reason}) but no web_search_tool_result "
          "block was present; the model may have answered without searching. Text: "
          f"{next((b.text for b in r.content if b.type == 'text'), '')[:160]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
