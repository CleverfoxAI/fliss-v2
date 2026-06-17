"""
Model-resolution regression tests (offline, no server, no network).

These lock in the fix for the production incident where a retired Anthropic
model snapshot (claude-sonnet-4-20250514) took Fliss down with 404s. Run in CI
before deploy so a reintroduced/retired model fails the build, not production:

    python test_model_resolution.py        # or: pytest test_model_resolution.py
"""
from __future__ import annotations

from config import RETIRED_MODELS, resolve_model, Settings


def test_retired_sonnet4_maps_to_active_sonnet45():
    # Sonnet 4 -> Sonnet 4.5 (preserves tool-calling; 4.6 under-triggers tools).
    assert resolve_model("claude-sonnet-4-20250514") == "claude-sonnet-4-5"


def test_retired_opus4_maps_to_active_opus48():
    assert resolve_model("claude-opus-4-20250514") == "claude-opus-4-8"


def test_active_model_passes_through_unchanged():
    for active in ("claude-sonnet-4-5", "claude-opus-4-8", "claude-haiku-4-5"):
        assert resolve_model(active) == active


def test_resolution_is_idempotent():
    for raw in ("claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-sonnet-4-5"):
        once = resolve_model(raw)
        assert resolve_model(once) == once, f"{raw!r} -> {once!r} not stable"


def test_production_remap_targets_are_not_retired():
    # The models we actively migrate the production snapshots to must be live.
    for snapshot in ("claude-sonnet-4-20250514", "claude-opus-4-20250514"):
        assert resolve_model(snapshot) not in RETIRED_MODELS


def test_config_default_model_is_not_retired():
    # If FLISS_MODEL is ever unset, the in-code default must still be a live model.
    default = Settings.model_fields["fliss_model"].default
    assert default not in RETIRED_MODELS, f"default model {default!r} is retired"
    assert resolve_model(default) not in RETIRED_MODELS


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failures = 0
    for t in _TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    raise SystemExit(1 if failures else 0)
