from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Postgres
    database_url: str = ""

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""
    pinecone_environment: str = ""
    pinecone_top_k: int = 5

    # Anthropic
    anthropic_api_key: str
    # Default to an active model ALIAS (not a dated snapshot). Aliases roll
    # forward and are not retired on the per-snapshot schedule, so an unset/
    # mistyped FLISS_MODEL can't silently fall back to a retired model.
    fliss_model: str = "claude-sonnet-4-5"

    # Google Maps
    google_maps_api_key: str = ""

    @property
    def use_live_db(self) -> bool:
        return bool(self.database_url and "localhost" not in self.database_url)

    @property
    def use_live_pinecone(self) -> bool:
        return bool(self.pinecone_api_key and not self.pinecone_api_key.startswith("your-"))

    @property
    def use_live_geocoding(self) -> bool:
        return bool(self.google_maps_api_key and not self.google_maps_api_key.startswith("your-"))

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ── Model resolution ─────────────────────────────────────────────────────────
# Models Anthropic has retired. Requesting any of these returns a 404 and takes
# Fliss down. Keep this in sync with Anthropic's deprecation notices — it is the
# single source of truth for the startup validity check (see main.py) and the
# regression test (test_model_resolution.py).
RETIRED_MODELS = frozenset({
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
})

# Remap stale/retired model strings to an active equivalent so a stale
# FLISS_MODEL value (e.g. left on Railway) can't take the service down. This is
# a defensive fallback; the env var should still be set to an active model.
#
# Sonnet 4 -> Sonnet 4.5 (NOT 4.6): 4.5 is the closest active model to the
# retired Sonnet 4 and preserves its tool-calling behaviour. The 4.6 family
# under-triggers tools, which made Fliss narrate "here are the listings"
# without calling search_listings, so no result cards rendered.
_MODEL_MIGRATIONS = {
    "claude-sonnet-4-20250514": "claude-sonnet-4-5",
    "claude-opus-4-20250514": "claude-opus-4-8",
    # Legacy Haiku remaps (pre-existing behaviour, kept as-is).
    "claude-3-haiku-20240307": "claude-3-5-haiku-20241022",
    "claude-haiku-4-5-20251001": "claude-3-5-haiku-20241022",
}


def resolve_model(raw: str) -> str:
    """Map a configured model string to the model actually used at runtime.

    Centralised so the engine and the startup validity check always agree on
    the effective model. Unknown/active models pass through unchanged.
    """
    return _MODEL_MIGRATIONS.get(raw, raw)
