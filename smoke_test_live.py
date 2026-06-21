"""
Live pre-flight smoke test — run this against PRODUCTION before a demo.

It drives a real care-home enquiry through the deployed /api/query and checks the
exact failure modes Nathan hit:
  * the "trouble reaching live results" fallback (a downstream error)
  * Fliss re-asking for the location after it was given (context loss)
  * never reaching any results

Usage (point BASE_URL at the live app — the Railway URL or whatever serves
/api/query):

    BASE_URL=https://web-production-b9f4b.up.railway.app python smoke_test_live.py

This makes REAL model calls (small cost) and is non-deterministic, so treat a
FAIL as "investigate", not "definitely broken" — but a clean PASS on the real
URL is the strongest pre-demo signal we have.
"""
from __future__ import annotations

import os
import sys
import uuid

import httpx

BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
TIMEOUT = 90
SESSION = str(uuid.uuid4())

# A realistic demo path: give location + who + condition up front, then answer
# the follow-ups (condition info, funding, wellbeing) and ask for options.
SCRIPT = [
    "I'm looking for a care home in Brighton for my mum who has dementia",
    "no",                    # "more info about dementia?" -> no
    "no",                    # funding offer -> no
    "I'm doing ok thanks",   # wellbeing check-in response
    "yes please show me the options",
    "yes",                   # in case results are deferred one more turn
]

FALLBACK_MARKERS = ("trouble reaching live results",)
REASK_LOCATION_MARKERS = (
    "whereabouts are you looking", "where are you looking",
    "which town or postcode", "what's the location", "where would you like",
)


def _post(message: str) -> dict:
    r = httpx.post(
        f"{BASE_URL}/api/query",
        json={"query": message, "mode": "text",
              "context": {"session_id": SESSION}, "type": "CAREHOME"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    if not BASE_URL:
        print("ERROR: set BASE_URL to the live app, e.g.\n"
              "  BASE_URL=https://web-production-b9f4b.up.railway.app python smoke_test_live.py")
        return 2

    print(f"Target: {BASE_URL}\nSession: {SESSION}\n")

    # 1. Health
    try:
        h = httpx.get(f"{BASE_URL}/health", timeout=20).json()
        print(f"/health -> {h}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: /health unreachable: {e}")
        return 1

    saw_fallback = False
    saw_reask = False
    saw_results = False

    for turn, msg in enumerate(SCRIPT, 1):
        try:
            data = _post(msg)
        except Exception as e:  # noqa: BLE001
            print(f"\nFAIL: turn {turn} request error: {e}")
            return 1
        answer = (data.get("answer") or "")
        low = answer.lower()
        n = len(data.get("results") or [])
        if n:
            saw_results = True
        if any(m in low for m in FALLBACK_MARKERS):
            saw_fallback = True
        # Location was given in turn 1, so any later re-ask is the context-loss bug.
        if turn > 1 and any(m in low for m in REASK_LOCATION_MARKERS):
            saw_reask = True
        print(f"\n[turn {turn}] you: {msg}")
        print(f"          fliss: {answer[:160]}")
        print(f"          results: {n}")

    print("\n" + "=" * 60)
    problems = []
    if saw_fallback:
        problems.append("saw the 'trouble reaching live results' fallback (downstream error)")
    if saw_reask:
        problems.append("Fliss re-asked for the location after it was given (context loss)")
    if not saw_results:
        problems.append("never returned any results across the whole flow")

    if problems:
        print("FAIL — issues to investigate before the demo:")
        for p in problems:
            print(f"  - {p}")
        print("\nGrab the Railway Deploy Logs (search 'FLISS-ERROR') for the cause.")
        return 1

    print("PASS — full enquiry completed live: results returned, no fallback, "
          "no re-asking. This is the real pre-demo green light.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
