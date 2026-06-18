from __future__ import annotations
import asyncio
import logging
import asyncpg
from config import get_settings


_pool: asyncpg.Pool | None = None

# Transient DB errors worth retrying (stale pooled connection, server blip,
# network reset, timeout). Built defensively so a missing asyncpg attribute
# can't crash import.
_DB_RETRY_ERRORS = tuple(
    e for e in (
        getattr(asyncpg, "PostgresConnectionError", None),
        getattr(asyncpg, "InterfaceError", None),
    ) if e is not None
) + (ConnectionError, OSError, asyncio.TimeoutError)

RATING_TEXT_TO_NUMBER = {
    "outstanding": 5,
    "good": 4,
    "requires improvement": 2,
    "inadequate": 1,
}


def _rating_to_number(value) -> float:
    """Convert an overallRating to a numeric totalRate."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return float(RATING_TEXT_TO_NUMBER.get(str(value).strip().lower(), 0))

# Map from our internal types to the DB enum values
TYPE_MAP = {
    "CAREHOME": "CAREHOME",
    "NURSERY": "NURSERY",
    "HOMECARE": "HOMECARE",
}


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(
            settings.database_url,
            # Recycle idle connections so the pool never hands out a stale one
            # the DB has already closed — a common cause of intermittent
            # "trouble reaching live results".
            max_inactive_connection_lifetime=120,
            command_timeout=30,
        )
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def _run_query(query: str, params: list, attempts: int = 3) -> list:
    """Run a fetch with retry on transient connection errors.

    asyncpg discards a broken connection when it errors, so simply retrying the
    acquire+fetch lets the pool hand us a healthy one — without tearing down the
    shared pool (which would disrupt concurrent requests).
    """
    last_exc = None
    for i in range(attempts):
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                return await conn.fetch(query, *params)
        except _DB_RETRY_ERRORS as exc:
            last_exc = exc
            logging.warning(
                "FLISS-ERROR step=db_query attempt=%d/%d err=%s: %s",
                i + 1, attempts, type(exc).__name__, exc,
            )
            await asyncio.sleep(0.3 * (2 ** i))
    raise last_exc


async def search_listings(
    page_type: str,
    latitude: float | None = None,
    longitude: float | None = None,
    radius_km: float = 15.0,
    keywords: list[str] | None = None,
    limit: int = 8,
) -> list[dict]:
    """Search organisations in the real Caretopia Postgres database.

    Args:
        page_type: 'CAREHOME', 'NURSERY', or 'HOMECARE'.
        latitude: Latitude for geo search (from geocoding).
        longitude: Longitude for geo search (from geocoding).
        radius_km: Search radius in kilometres.
        keywords: Conditions/specialisms to filter by.
        limit: Max results (default 8).
    """
    settings = get_settings()
    if not settings.use_live_db:
        return _filter_test_data(page_type, keywords, limit)

    db_type = TYPE_MAP.get(page_type, page_type)
    kw_list = keywords or []

    conditions = [
        "type = $1",
        '"isDeleted" = false',
        "status = 'ACTIVE'",
        "latitude IS NOT NULL",
        "longitude IS NOT NULL",
    ]
    params: list = [db_type]
    param_idx = 2

    # Haversine geo filter (earth radius = 6371 km)
    if latitude is not None and longitude is not None:
        conditions.append(f"""
            (6371 * acos(
                LEAST(GREATEST(
                    cos(radians(${param_idx})) * cos(radians(latitude))
                    * cos(radians(longitude) - radians(${param_idx + 1}))
                    + sin(radians(${param_idx})) * sin(radians(latitude))
                , -1), 1)
            )) <= ${param_idx + 2}
        """)
        params.extend([latitude, longitude, radius_km])
        param_idx += 3

    # Keyword filter — match against keywords JSON or description
    if kw_list:
        kw_conditions = []
        for kw in kw_list:
            kw_conditions.append(
                f"(keywords::text ILIKE ${param_idx} OR description ILIKE ${param_idx})"
            )
            params.append(f"%{kw}%")
            param_idx += 1
        conditions.append(f"({' OR '.join(kw_conditions)})")

    where_clause = " AND ".join(conditions)

    # Build distance calculation
    if latitude is not None and longitude is not None:
        distance_expr = f""",
            round((6371 * acos(
                LEAST(GREATEST(
                    cos(radians(${2})) * cos(radians(latitude))
                    * cos(radians(longitude) - radians(${3}))
                    + sin(radians(${2})) * sin(radians(latitude))
                , -1), 1)
            ))::numeric, 2) AS distance_km
        """
        order_clause = "ORDER BY distance_km ASC"
    else:
        distance_expr = ""
        order_clause = 'ORDER BY "createdAt" DESC'

    query = f"""
        SELECT
            o.id,
            o."organisationName",
            o."addressLine1",
            o."townCity",
            o.postcode,
            o.latitude,
            o.longitude,
            o.type,
            o.description,
            o.keywords,
            o."cqcGrade",
            o."ofstedGrade",
            o."overallRating",
            o."contactPhone",
            o."contactEmail",
            o.website,
            o."weeklyFeesGuide",
            o."facilitiesAndServices",
            o."registeredPlaces",
            o."fullAddress",
            o."managerName",
            o.slug,
            o."logoUrl",
            o."bannerUrl",
            o."minAge",
            o."maxAge",
            o.parking,
            o."companyStatus",
            o."inclusiveNursery",
            o."createdAt",
            o."socialMedia"
            {distance_expr}
        FROM organisations o
        WHERE {where_clause}
        {order_clause}
        LIMIT ${param_idx}
    """
    params.append(limit)

    rows = await _run_query(query, params)
    results = []
    for row in rows:
        r = dict(row)
        # Convert Decimal latitude/longitude to float for JSON serialization
        if r.get("latitude") is not None:
            r["latitude"] = float(r["latitude"])
        if r.get("longitude") is not None:
            r["longitude"] = float(r["longitude"])
        if r.get("distance_km") is not None:
            r["distance_km"] = float(r["distance_km"])
        r.setdefault("slug", None)
        r.setdefault("overallRating", None)
        r["totalRate"] = _rating_to_number(r.get("overallRating"))
        results.append(r)
    return results


# ── Jobs search ──────────────────────────────────────────────────────────────

async def search_jobs(
    latitude: float | None = None,
    longitude: float | None = None,
    radius_km: float = 25.0,
    keywords: list[str] | None = None,
    job_type: str | None = None,
    limit: int = 8,
) -> list[dict]:
    """Search jobs in the Caretopia Postgres database.

    Joins to organisations table for geo coordinates.

    Args:
        latitude: Latitude for geo search (from geocoding).
        longitude: Longitude for geo search (from geocoding).
        radius_km: Search radius in kilometres.
        keywords: Job title, role, or skill keywords to filter by.
        job_type: FULLTIME, PARTTIME, TEMPORARY, CONTRACT, FLEXIBLE, INTERNSHIP.
        limit: Max results (default 8).
    """
    settings = get_settings()
    if not settings.use_live_db:
        return _filter_test_jobs(keywords, limit)

    conditions = [
        "j.status = 'ACTIVE'",
    ]
    params: list = []
    param_idx = 1

    # Haversine geo filter via the joined organisation
    if latitude is not None and longitude is not None:
        conditions.append(f"""
            o.latitude IS NOT NULL AND o.longitude IS NOT NULL AND
            (6371 * acos(
                LEAST(GREATEST(
                    cos(radians(${param_idx})) * cos(radians(o.latitude))
                    * cos(radians(o.longitude) - radians(${param_idx + 1}))
                    + sin(radians(${param_idx})) * sin(radians(o.latitude))
                , -1), 1)
            )) <= ${param_idx + 2}
        """)
        params.extend([latitude, longitude, radius_km])
        param_idx += 3

    # Job type filter
    if job_type:
        conditions.append(f"j.\"jobType\" = ${param_idx}")
        params.append(job_type)
        param_idx += 1

    # Keyword filter — match against title, jobRole, or summary
    kw_list = keywords or []
    if kw_list:
        kw_conditions = []
        for kw in kw_list:
            kw_conditions.append(
                f"(j.title ILIKE ${param_idx} OR j.\"jobRole\" ILIKE ${param_idx} "
                f"OR j.summary ILIKE ${param_idx} OR j.location ILIKE ${param_idx})"
            )
            params.append(f"%{kw}%")
            param_idx += 1
        conditions.append(f"({' OR '.join(kw_conditions)})")

    where_clause = " AND ".join(conditions)

    # Build distance calculation
    if latitude is not None and longitude is not None:
        lat_p = next(i for i, v in enumerate(params, 1) if v == latitude)
        lng_p = lat_p + 1
        distance_expr = f""",
            round((6371 * acos(
                LEAST(GREATEST(
                    cos(radians(${lat_p})) * cos(radians(o.latitude))
                    * cos(radians(o.longitude) - radians(${lng_p}))
                    + sin(radians(${lat_p})) * sin(radians(o.latitude))
                , -1), 1)
            ))::numeric, 2) AS distance_km
        """
        order_clause = "ORDER BY distance_km ASC"
    else:
        distance_expr = ""
        order_clause = 'ORDER BY j."createdAt" DESC'

    query = f"""
        SELECT
            j.id,
            j.title,
            j."jobRole",
            j."jobType",
            j.location AS "jobLocation",
            j.summary,
            j."minSalaryRange",
            j."maxSalaryRange",
            j."minWorkTime",
            j."maxWorkTime",
            j.shifts,
            j."minExperience",
            j."maxExperience",
            j."whyWorkHere",
            j."startTime",
            j."expireTime",
            j.qualifications,
            j."createdAt",
            o.id AS "organisationId",
            o."organisationName",
            o."townCity",
            o.postcode,
            o.latitude,
            o.longitude,
            o."logoUrl",
            o.slug AS "orgSlug"
            {distance_expr}
        FROM jobs j
        LEFT JOIN organisations o ON j."organizationId" = o.id
        WHERE {where_clause}
        {order_clause}
        LIMIT ${param_idx}
    """
    params.append(limit)

    rows = await _run_query(query, params)
    results = []
    for row in rows:
        r = dict(row)
        if r.get("latitude") is not None:
            r["latitude"] = float(r["latitude"])
        if r.get("longitude") is not None:
            r["longitude"] = float(r["longitude"])
        if r.get("distance_km") is not None:
            r["distance_km"] = float(r["distance_km"])
        results.append(r)
    return results


# ── Test data fallback (no DB) ───────────────────────────────────────────────

TEST_CARE_HOMES = [
    {
        "id": 1, "organisationName": "Sunrise Manor Care Home", "type": "CAREHOME",
        "addressLine1": "14 Marine Parade", "townCity": "Brighton", "postcode": "BN2 1TL",
        "latitude": 50.8194, "longitude": -0.1235,
        "description": "A warm, family-run care home specialising in dementia and residential care.",
        "keywords": ["dementia", "residential", "respite"],
        "cqcGrade": "Good", "contactPhone": "01273 555 001",
    },
    {
        "id": 2, "organisationName": "The Willows Nursing Home", "type": "CAREHOME",
        "addressLine1": "8 Preston Road", "townCity": "Brighton", "postcode": "BN1 6AF",
        "latitude": 50.8411, "longitude": -0.1494,
        "description": "CQC Outstanding nursing home with specialist dementia unit.",
        "keywords": ["nursing", "dementia", "physiotherapy"],
        "cqcGrade": "Outstanding", "contactPhone": "01273 555 002",
    },
]

TEST_NURSERIES = [
    {
        "id": 101, "organisationName": "Little Stars Nursery", "type": "NURSERY",
        "addressLine1": "28 Church Road", "townCity": "Hove", "postcode": "BN3 2FN",
        "latitude": 50.8350, "longitude": -0.1720,
        "description": "Ofsted Outstanding nursery for ages 3 months to 5 years.",
        "keywords": ["forest school", "SEN support"],
        "ofstedGrade": "Outstanding", "contactPhone": "01273 555 101",
    },
]

TEST_HOME_CARE = [
    {
        "id": 201, "organisationName": "Compassionate Home Care", "type": "HOMECARE",
        "addressLine1": "Unit 3, Enterprise Point", "townCity": "Brighton", "postcode": "BN1 4GH",
        "latitude": 50.8380, "longitude": -0.1410,
        "description": "CQC Outstanding domiciliary care provider.",
        "keywords": ["personal care", "dementia", "live-in"],
        "cqcGrade": "Outstanding", "contactPhone": "01273 555 201",
    },
]

TEST_JOBS = [
    {
        "id": 301, "title": "Care Assistant", "jobRole": "Care Assistant",
        "jobType": "FULLTIME", "jobLocation": "Brighton", "shifts": "MORNING",
        "summary": "Caring for elderly residents in a supportive environment.",
        "minSalaryRange": "22000", "maxSalaryRange": 26000,
        "minExperience": 0, "maxExperience": 2,
        "organisationName": "Sunrise Manor Care Home", "townCity": "Brighton",
        "postcode": "BN2 1TL", "latitude": 50.8194, "longitude": -0.1235,
    },
    {
        "id": 302, "title": "Senior Nurse", "jobRole": "Nurse",
        "jobType": "FULLTIME", "jobLocation": "Brighton", "shifts": "NIGHT",
        "summary": "Experienced nurse needed for dementia unit.",
        "minSalaryRange": "32000", "maxSalaryRange": 38000,
        "minExperience": 3, "maxExperience": 10,
        "organisationName": "The Willows Nursing Home", "townCity": "Brighton",
        "postcode": "BN1 6AF", "latitude": 50.8411, "longitude": -0.1494,
    },
]

TEST_DATA = {
    "CAREHOME": TEST_CARE_HOMES,
    "NURSERY": TEST_NURSERIES,
    "HOMECARE": TEST_HOME_CARE,
}


def _filter_test_jobs(keywords: list[str] | None, limit: int) -> list[dict]:
    listings = TEST_JOBS[:]
    if keywords:
        filtered = [
            l for l in listings
            if any(
                kw.lower() in (l.get("title", "") + " " + l.get("summary", "") + " " + l.get("jobRole", "")).lower()
                for kw in keywords
            )
        ]
        listings = filtered or listings
    results = []
    for i, listing in enumerate(listings[:limit]):
        results.append({**listing, "distance_km": round(1.2 + i * 1.8, 1)})
    return results


def _filter_test_data(page_type: str, keywords: list[str] | None, limit: int) -> list[dict]:
    listings = TEST_DATA.get(page_type, [])
    if keywords:
        filtered = [
            l for l in listings
            if any(
                kw.lower() in (l.get("description", "") + " ".join(l.get("keywords", []))).lower()
                for kw in keywords
            )
        ]
        listings = filtered or listings
    results = []
    for i, listing in enumerate(listings[:limit]):
        r = {**listing, "distance_km": round(1.2 + i * 1.8, 1)}
        r.setdefault("slug", None)
        r.setdefault("overallRating", None)
        r["totalRate"] = float(r["overallRating"]) if r.get("overallRating") is not None else 0
        results.append(r)
    return results
