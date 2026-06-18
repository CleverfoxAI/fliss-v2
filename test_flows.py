"""
Conversation-flow tests (offline; mocks the Anthropic client + search backend).

Locks in page-type-correct behaviour:
  * JOBS is a job-seeker flow — it must NOT trigger the care-seeker wellbeing
    check-in, and must return job results directly.
  * CAREHOME still defers the first results behind the wellbeing check-in.

Run: python test_flows.py   (or: pytest)
"""
from __future__ import annotations

import asyncio
import json
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")


class _Block:
    def __init__(self, type, text=None, name=None, input=None, id=None):
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.id = id


class _Resp:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    async def create(self, **_kw):
        r = self._responses[self._i]
        self._i += 1
        return r


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_jobs_flow_skips_wellbeing_and_returns_results():
    from chat.engine import ConversationEngine, WELLBEING_CHECKIN_QUESTION

    eng = ConversationEngine("JOBS")
    eng.client = _FakeClient([
        _Resp([_Block("tool_use", name="search_jobs",
                      input={"location": "Hertfordshire"}, id="t1")], "tool_use"),
        _Resp([_Block("text", text="Here are some care roles near Hertfordshire. "
                                   "Take a look at the options!")], "end_turn"),
    ])

    async def fake_job_search(_inp):
        jobs = [{"title": "Care Assistant", "jobLocation": "Hertfordshire"}]
        return json.dumps({"results": jobs}), jobs, {"latitude": 51.8, "longitude": -0.2}

    eng._handle_job_search = fake_job_search

    res = asyncio.run(eng.chat("care home jobs in Hertfordshire", []))
    assert res["answer"] != WELLBEING_CHECKIN_QUESTION, "JOBS must not get the wellbeing check-in"
    assert res["results"] == [{"title": "Care Assistant", "jobLocation": "Hertfordshire"}]
    assert res["intent"] == "listings"


def test_carehome_flow_still_defers_to_wellbeing_checkin():
    from chat.engine import ConversationEngine, WELLBEING_CHECKIN_QUESTION

    eng = ConversationEngine("CAREHOME")
    eng.client = _FakeClient([
        _Resp([_Block("tool_use", name="search_listings",
                      input={"location": "Brighton"}, id="t1")], "tool_use"),
        _Resp([_Block("text", text="Here are some homes. Take a look at the options!")], "end_turn"),
    ])

    async def fake_search(_inp):
        homes = [{"organisationName": "Sunrise Manor"}]
        return json.dumps({"results": homes}), homes, {"latitude": 50.8, "longitude": -0.1}

    eng._handle_search = fake_search

    # "my mum" satisfies the who-guard so the search proceeds.
    res = asyncio.run(eng.chat("care home for my mum in Brighton", []))
    assert res["answer"] == WELLBEING_CHECKIN_QUESTION, "CAREHOME should defer to the wellbeing check-in"


def test_radius_expands_only_when_no_results():
    """Empty nearby search widens the radius; a search with hits is untouched."""
    import chat.engine as engine

    eng = engine.ConversationEngine("JOBS")

    # search_jobs returns [] until the radius crosses 25 (simulating a county
    # whose roles sit outside the default 25 km).
    calls = []

    async def fake_search_jobs(*, latitude, longitude, radius_km, keywords, job_type, limit):
        calls.append(radius_km)
        return [{"title": "Care Assistant"}] if radius_km > 25 else []

    async def fake_geocode(_loc):
        return {"latitude": 51.8, "longitude": -0.2, "formatted_address": "Hertfordshire, UK"}

    orig_search, orig_geo = engine.search_jobs, engine.geocode_location
    engine.search_jobs = fake_search_jobs
    engine.geocode_location = fake_geocode
    try:
        _j, rows, _g = asyncio.run(eng._handle_job_search({"location": "Hertfordshire"}))
    finally:
        engine.search_jobs, engine.geocode_location = orig_search, orig_geo

    assert rows == [{"title": "Care Assistant"}], "expansion should surface the county's roles"
    assert max(calls) > 25, f"radius should have widened beyond 25; tried {calls}"

    # Control: when the first (25 km) search already returns a hit, no widening.
    calls.clear()

    async def fake_hit(*, latitude, longitude, radius_km, keywords, job_type, limit):
        calls.append(radius_km)
        return [{"title": "Nurse"}]

    engine.search_jobs = fake_hit
    engine.geocode_location = fake_geocode
    try:
        _j, rows, _g = asyncio.run(eng._handle_job_search({"location": "Watford"}))
    finally:
        engine.search_jobs, engine.geocode_location = orig_search, orig_geo

    assert rows == [{"title": "Nurse"}]
    assert calls == [25], f"no widening when results exist; tried {calls}"


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failures = 0
    for t in _TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    raise SystemExit(1 if failures else 0)
