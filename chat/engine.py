from __future__ import annotations
import json
import logging
import re
from anthropic import AsyncAnthropic
from config import get_settings, resolve_model
from chat.prompts import get_system_prompt
from tools.search import search_listings, search_jobs
from tools.knowledge import search_knowledge_base
from tools.geocoding import geocode_location


# Map frontend type values to our internal page_type for prompts
FRONTEND_TYPE_TO_PAGE = {
    "CAREHOME": "care_homes",
    "NURSERY": "nurseries",
    "HOMECARE": "home_care",
    "JOBS": "jobs",
}


TOOLS_LISTINGS = [
    {
        "name": "search_listings",
        "description": (
            "Search the Caretopia database for care providers. "
            "IMPORTANT: Do NOT call this tool until you have gathered: (1) a location, "
            "(2) who the care is for, and (3) at least one specific need or preference. "
            "If any of these are missing, ask the user first. Never call on greetings "
            "or vague messages. The location will be geocoded automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location name or postcode to search near.",
                },
                "radius_km": {
                    "type": "number",
                    "description": "Search radius in kilometres. Default 25.",
                    "default": 25,
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CRITICAL: Must include ALL conditions, specialisms, preferences, and requirements mentioned ANYWHERE in the entire conversation history — not just the latest message. Re-read every user message and collect every criterion. Example: if user mentioned 'dementia' earlier and now says 'garden', pass BOTH ['dementia', 'garden']. Never drop previous keywords.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return. Default 8.",
                    "default": 8,
                },
            },
            "required": ["location"],
        },
    },
]

TOOLS_JOBS = [
    {
        "name": "search_jobs",
        "description": (
            "Search the Caretopia jobs database for care sector jobs. "
            "Use this when the user wants to find a job — at minimum a location. "
            "The location will be geocoded automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location name or postcode to search near.",
                },
                "radius_km": {
                    "type": "number",
                    "description": "Search radius in kilometres. Default 25.",
                    "default": 25,
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Job role, title, or skill keywords (e.g. 'nurse', 'care assistant', 'support worker').",
                },
                "job_type": {
                    "type": "string",
                    "enum": ["FULLTIME", "PARTTIME", "TEMPORARY", "CONTRACT", "FLEXIBLE", "INTERNSHIP"],
                    "description": "Filter by job type.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return. Default 8.",
                    "default": 8,
                },
            },
            "required": ["location"],
        },
    },
]

TOOL_KNOWLEDGE = {
    "name": "search_knowledge_base",
    "description": (
        "Search the knowledge base for general care information — funding, "
        "conditions, organisations, charities, processes. Use for informational "
        "queries, NOT for finding specific providers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The information query (e.g. 'how does care home funding work?').",
            },
        },
        "required": ["query"],
    },
}


def get_tools(frontend_type: str) -> list[dict]:
    """Return the appropriate tool set for the page type."""
    if frontend_type == "JOBS":
        return TOOLS_JOBS + [TOOL_KNOWLEDGE]
    return TOOLS_LISTINGS + [TOOL_KNOWLEDGE]


# Negative responses that clearly decline funding info
_NEGATIVE_PATTERNS = re.compile(
    r"^\s*(no|nah|nope|skip|no thanks|no thank you|just show|show me|"
    r"don\'t need|not interested|not right now|maybe later|"
    r"just search|just find|go ahead and search|search for)\b",
    re.IGNORECASE,
)


def _last_assistant_offered_funding(messages: list[dict]) -> bool:
    """Check if the last assistant message ASKED the funding question.

    Only triggers on the offer/question (contains "would you like" AND "funding"),
    NOT on the answer (contains funding info but no "would you like").
    """
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            content = msg.get("content", "")
            # Handle content that is a list of blocks (from API responses)
            if isinstance(content, list):
                text = " ".join(
                    getattr(block, "text", "") if hasattr(block, "text")
                    else block.get("text", "") if isinstance(block, dict)
                    else ""
                    for block in content
                )
            elif isinstance(content, str):
                text = content
            else:
                return False
            text_lower = text.lower()
            has_offer = "would you like" in text_lower and "funding" in text_lower
            # Don't match if this is the funding INFO response (contains
            # actual funding details) — that means funding was already provided
            # and we should NOT block the next turn.
            has_funding_info = any(
                term in text_lower
                for term in [
                    "local authority", "attendance allowance",
                    "continuing healthcare", "self-fund",
                    "deferred payment",
                ]
            )
            return has_offer and not has_funding_info
    return False


def _user_declined_funding(message: str) -> bool:
    """Check if the user explicitly declined funding info.

    Default assumption: if the user didn't clearly say no, treat it as yes.
    This is the inverse approach — block search unless user clearly declines.
    """
    return bool(_NEGATIVE_PATTERNS.search(message.strip()))


# Words that indicate "who" the care is for — relationship terms, self-references
WHO_INDICATORS = [
    # Family relationships
    "mum", "mom", "mother", "dad", "father", "parent", "parents",
    "nan", "nana", "nanny", "grandmother", "grandma", "grandad",
    "grandfather", "grandpa", "gran",
    "son", "daughter", "child", "children", "kid", "kids", "baby",
    "toddler", "boy", "girl",
    "wife", "husband", "partner", "spouse",
    "brother", "sister", "sibling",
    "uncle", "aunt", "auntie",
    "friend", "neighbour", "neighbor",
    # Self-references
    "myself", "i need care", "i'm looking for care for me",
    # Generic person references
    "year old", "years old", "yo ",
    "elderly", "loved one",
    # Possessive patterns that imply who
    "my ",  # "my mum", "my daughter", etc.
    "our ",  # "our mother"
]

NURSERY_AGE_PATTERN = re.compile(
    r"\b(\d{1,2}|one|two|three|four|five)\s*"
    r"(year|years|yr|yrs|yo|month|months|mths)?\s*(old)?\b",
    re.IGNORECASE,
)


def _conversation_mentions_who(messages: list[dict], frontend_type: str | None = None) -> bool:
    """Check if any message in the conversation mentions who the care is for."""
    is_nursery = (frontend_type or "").upper() == "NURSERY"
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        text = content.lower()
        if is_nursery and NURSERY_AGE_PATTERN.search(text):
            return True
        for indicator in WHO_INDICATORS:
            if indicator in text:
                return True
    return False


WELLBEING_CHECKIN_QUESTION = (
    "I've found some lovely options for you — I'll show them to you in just a "
    "moment. But first, how are you doing? Looking for care can be a lot to carry."
)

WELLBEING_ACKNOWLEDGMENT = (
    "Thank you for sharing — that means a lot. Here are the options I found for you:"
)

_NEGATIVE_WELLBEING_WORDS = {
    "struggling", "stressed", "worried", "terrible", "awful", "bad",
    "difficult", "overwhelmed", "anxious", "depressed", "lonely",
    "scared", "unwell", "rubbish", "exhausted", "drained", "shattered",
    "miserable", "hopeless", "helpless", "upset", "crying", "sad",
    "horrible", "rough", "crap", "crappy", "knackered",
}

_POSITIVE_WELLBEING_WORDS = {
    "well", "good", "great", "ok", "okay", "fine", "alright",
    "coping", "happy", "grand", "brilliant",
}

# Matches "not", "n't", "no", common typos ("nto", "nt"), and negated auxiliaries.
_NEGATION_PATTERN = re.compile(
    r"\b(not|no|nto|nt|nope|nah|never|ain'?t|aint|don'?t|dont|"
    r"can'?t|cant|isn'?t|isnt|wasn'?t|wasnt|couldn'?t|couldnt|"
    r"wouldn'?t|wouldnt|hardly|barely)\b"
)
_NEGATIVE_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(_NEGATIVE_WELLBEING_WORDS) + r")\b"
)
_POSITIVE_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(_POSITIVE_WELLBEING_WORDS) + r")\b"
)
_HARD_TIME_PATTERN = re.compile(
    r"\b(hard|tough|rough)\s+(time|day|days|week|month|patch|going|spell)\b"
)

_WELLBEING_SUPPORT_CARER = (
    "I'm really sorry to hear that. Please know you're not alone — "
    "Carers UK (carersuk.org, 0808 808 7777) and Age UK (ageuk.org.uk, "
    "0800 678 1602) both have advisers who can talk things through with "
    "you. Your GP can also help if things feel overwhelming.\n\n"
)

_WELLBEING_SUPPORT_PARENT = (
    "I'm really sorry to hear that. Please know you're not alone — "
    "Family Lives (familylives.org.uk, 0808 800 2222) and Home-Start "
    "(home-start.org.uk, 0116 464 5490) both have people who can talk "
    "things through with you. Your health visitor or GP can also help "
    "if things feel overwhelming.\n\n"
)


def _wellbeing_support_message(frontend_type: str) -> str:
    if (frontend_type or "").upper() == "NURSERY":
        return _WELLBEING_SUPPORT_PARENT
    return _WELLBEING_SUPPORT_CARER


def _wellbeing_response_is_negative(user_message: str) -> bool:
    """Check if the user's response to the wellbeing check-in contains negative sentiment."""
    text = user_message.lower()
    if _NEGATIVE_WORD_PATTERN.search(text):
        return True
    if _HARD_TIME_PATTERN.search(text):
        return True
    # Negation + positive word catches "not well", "nto doing well",
    # "not great", "not coping", "isn't ok", etc. — robust to typos.
    if _NEGATION_PATTERN.search(text) and _POSITIVE_WORD_PATTERN.search(text):
        return True
    return False


def _assistant_text(msg: dict) -> str:
    """Extract plain text from an assistant message's content (str or block list)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            getattr(block, "text", "") if hasattr(block, "text")
            else block.get("text", "") if isinstance(block, dict)
            else ""
            for block in content
        )
    return ""


def _wellbeing_checkin_done(messages: list[dict]) -> bool:
    """True if any assistant message in history contains the exact wellbeing check-in question."""
    target = WELLBEING_CHECKIN_QUESTION.lower()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        text = _assistant_text(msg).lower()
        if target in text:
            return True
    return False


def _wellbeing_checkin_offered(messages: list[dict]) -> bool:
    """True if the exact wellbeing check-in question has been asked at any point."""
    target = WELLBEING_CHECKIN_QUESTION.lower()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        text = _assistant_text(msg).lower()
        if target in text:
            return True
    return False


def _get_pending_results(conversation_history: list[dict]) -> dict | None:
    """Return pending_results stored on the most recent assistant message, if any."""
    for msg in reversed(conversation_history):
        if msg.get("role") == "assistant":
            return msg.get("pending_results")
    return None


_WELLBEING_QUESTION_PATTERN = re.compile(
    r"how\s+(are\s+you|are\s+things|have\s+you\s+been|"
    r"are\s+you\s+(doing|feeling|holding\s+up|coping|getting\s+on))",
    re.IGNORECASE,
)


def _last_assistant_asked_wellbeing(messages: list[dict]) -> bool:
    """True if the most recent assistant message asked a wellbeing-style question.

    Matches the hardcoded WELLBEING_CHECKIN_QUESTION and any LLM paraphrase
    using a 'how are you …' / 'how are things' opener. Matches intent, not
    the exact constant: the constant contains an em-dash that can be
    normalised in transit, and the LLM sometimes asks the question itself
    rather than going through the hardcoded return path.
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return bool(_WELLBEING_QUESTION_PATTERN.search(_assistant_text(msg)))
    return False


# Phrases the model uses when it claims to be showing listings. Result cards
# only render when search_listings actually returns data, so these phrases
# appearing WITHOUT a search are the signature of the "narrated results but no
# cards" failure. Used both to trigger auto-recovery and to alert on it.
_RESULTS_TEXT_PHRASES = (
    "take a look at the options",
    "take a look at the listings",
    "here are some",
    "i've found some",
    "i found some",
    "check out the options",
    "options below",
)


def _looks_like_results_text(answer: str) -> bool:
    low = (answer or "").lower()
    return any(phrase in low for phrase in _RESULTS_TEXT_PHRASES)


class ConversationEngine:
    def __init__(self, frontend_type: str):
        """
        Args:
            frontend_type: 'CAREHOME', 'NURSERY', or 'HOMECARE' (from frontend).
        """
        settings = get_settings()
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        # Resolve the configured model through the central remap (config.resolve_model)
        # so a stale/retired FLISS_MODEL value can't take the service down. Keeping
        # this in one place means the engine and the startup validity check (main.py)
        # always agree on the effective model.
        self.model = resolve_model(settings.fliss_model)
        self.frontend_type = frontend_type
        print(f"[DEBUG] frontend_type={self.frontend_type}")
        page_type = FRONTEND_TYPE_TO_PAGE.get(frontend_type, "care_homes")
        self.page_type = page_type
        self.system_prompt = get_system_prompt(page_type)
        self.tools = get_tools(frontend_type)

    async def chat(
        self, message: str, conversation_history: list[dict]
    ) -> dict:
        """Process a user message and return frontend-compatible response.

        Args:
            message: The user's new message.
            conversation_history: Previous messages in [{role, content}, ...] format.

        Returns:
            {
                "intent": str,
                "confidence": float,
                "answer": str,
                "results": list,
                "title": str,
                "center_lat": float | None,
                "center_lng": float | None,
            }
        """
        # --- Wellbeing support short-circuit (robust) ---
        # Trigger off the visible transcript, not pending_results metadata,
        # which does not survive all session storage paths. If the previous
        # assistant turn asked the wellbeing check-in, the user's reply is
        # answering it — return the hardcoded support message verbatim. The
        # LLM must NEVER compose this reply: it produces markdown link syntax
        # that the frontend truncates and sometimes drops the support orgs.
        # --- DIAGNOSTIC LOGGING (temporary) ---
        _last_asst_msg = None
        for _m in reversed(conversation_history):
            if _m.get("role") == "assistant":
                _last_asst_msg = _m
                break
        _last_asst_raw = _last_asst_msg.get("content") if _last_asst_msg else None
        _last_asst_text = _assistant_text(_last_asst_msg) if _last_asst_msg else ""
        _detected_wellbeing = _last_assistant_asked_wellbeing(conversation_history)
        _detected_negative = _wellbeing_response_is_negative(message)
        print(f"[WB-DIAG] user_message={message!r}", flush=True)
        print(f"[WB-DIAG] last_assistant_content_type={type(_last_asst_raw).__name__}", flush=True)
        print(f"[WB-DIAG] last_assistant_raw={_last_asst_raw!r}", flush=True)
        print(f"[WB-DIAG] last_assistant_extracted_text={_last_asst_text!r}", flush=True)
        print(f"[WB-DIAG] _last_assistant_asked_wellbeing={_detected_wellbeing}", flush=True)
        print(f"[WB-DIAG] _wellbeing_response_is_negative={_detected_negative}", flush=True)
        _checkin_done = _wellbeing_checkin_done(conversation_history)
        print(f"[WB-DIAG] _wellbeing_checkin_done={_checkin_done}", flush=True)
        # --- END DIAGNOSTIC LOGGING ---

        # --- Distress short-circuit ---
        # If the user's current message expresses negative wellbeing AND the
        # verbatim wellbeing check-in has never been delivered, the LLM is
        # about to freelance an empathic reply with markdown links that the
        # frontend truncates. Pre-empt it with the hardcoded support message
        # + acknowledgment. The LLM never sees this turn.
        if _detected_negative and not _checkin_done:
            pending = _get_pending_results(conversation_history) or {}
            answer = _wellbeing_support_message(self.frontend_type) + WELLBEING_ACKNOWLEDGMENT
            print(f"[WB-DIAG] distress_short_circuit=TAKEN answer_first200={answer[:200]!r}", flush=True)
            return {
                "intent": "listings" if pending else "clarify",
                "confidence": 1.0,
                "answer": answer,
                "results": pending.get("results", []),
                "title": pending.get("title", ""),
                "center_lat": pending.get("center_lat"),
                "center_lng": pending.get("center_lng"),
                "filters_used": pending.get("filters_used"),
            }

        if _last_assistant_asked_wellbeing(conversation_history):
            if _wellbeing_response_is_negative(message):
                answer = _wellbeing_support_message(self.frontend_type) + WELLBEING_ACKNOWLEDGMENT
            else:
                answer = WELLBEING_ACKNOWLEDGMENT
            pending = _get_pending_results(conversation_history) or {}
            print(f"[WB-DIAG] short_circuit=TAKEN answer_first200={answer[:200]!r}", flush=True)
            return {
                "intent": "listings",
                "confidence": 1.0,
                "answer": answer,
                "results": pending.get("results", []),
                "title": pending.get("title", ""),
                "center_lat": pending.get("center_lat"),
                "center_lng": pending.get("center_lng"),
                "filters_used": pending.get("filters_used"),
            }
        print(f"[WB-DIAG] short_circuit=NOT_TAKEN — FALLING THROUGH TO LLM", flush=True)

        # Build messages for the API, injecting search context from
        # previous turns into user messages (not assistant messages) so the
        # model has cumulative criteria but never echoes the metadata.
        messages = []
        search_context = None
        for msg in conversation_history:
            content = msg["content"]
            if msg["role"] == "user" and search_context:
                content = (
                    f"[Search context for cumulative criteria — DO NOT repeat "
                    f"this text to the user: {search_context}]\n\n{content}"
                )
                search_context = None
            messages.append({"role": msg["role"], "content": content})
            # Pick up search metadata stored by main.py for the next turn
            if msg.get("filters_used"):
                f = msg["filters_used"]
                loc = f.get("location", "")
                kw = ", ".join(f.get("keywords", []))
                rad = f.get("radius_km", 25)
                search_context = f"location={loc}, keywords={kw}, radius={rad}km"

        # If the last assistant message had filters, attach context to current message
        current_content = message
        if search_context:
            current_content = (
                f"[Search context for cumulative criteria — DO NOT repeat "
                f"this text to the user: {search_context}]\n\n{message}"
            )
        messages.append({"role": "user", "content": current_content})

        # --- Funding state tracker (hard block) ---
        # If the last assistant message mentioned "funding" and the user
        # did NOT clearly decline, remove search tools so the AI physically
        # cannot search. Any response that isn't an explicit "no/skip" is
        # treated as wanting funding info. This is intentionally aggressive
        # because the AI was ignoring prompt-level instructions ~50% of the time.
        awaiting_funding = _last_assistant_offered_funding(messages)
        if awaiting_funding and not _user_declined_funding(message):
            # Strip search tools — only keep knowledge base tool
            turn_tools = [t for t in self.tools if t["name"] == "search_knowledge_base"]
            # Inject nudge via system prompt — do NOT modify messages so the
            # full conversation history (location, who, conditions) is preserved.
            funding_nudge = (
                "\n\n[SYSTEM: The user has asked for funding information. You MUST "
                "provide funding details from the FUNDING INFORMATION section in "
                "your instructions. Do NOT search for listings. Give them the "
                "funding guidance first, then ask if they are ready to see care options.]"
            )
            turn_system = self.system_prompt + funding_nudge
        else:
            turn_tools = self.tools
            turn_system = self.system_prompt

        search_performed = False
        filters_used = None
        listings_results = []
        center_lat = None
        center_lng = None

        # Tool-calling loop
        while True:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=turn_system,
                tools=turn_tools,
                messages=messages,
            )

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        if block.name == "search_listings":
                            # GUARD: Block search if "who" hasn't been mentioned
                            if not _conversation_mentions_who(messages, self.frontend_type):
                                result_json = json.dumps({
                                    "error": "BLOCKED: Cannot search yet. You must first ask the user who the care is for. Ask a clarifying question like 'And who are you looking for care for?' before searching.",
                                    "action": "ask_who",
                                })
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result_json,
                                })
                                continue
                            result_json, raw_results, geo = await self._handle_search(block.input)
                            search_performed = True
                            listings_results = raw_results
                            filters_used = {
                                "location": block.input.get("location"),
                                "keywords": block.input.get("keywords", []),
                                "radius_km": block.input.get("radius_km", 25),
                            }
                            if geo:
                                center_lat = geo["latitude"]
                                center_lng = geo["longitude"]
                        elif block.name == "search_jobs":
                            result_json, raw_results, geo = await self._handle_job_search(block.input)
                            search_performed = True
                            listings_results = raw_results
                            filters_used = {
                                "location": block.input.get("location"),
                                "keywords": block.input.get("keywords", []),
                                "radius_km": block.input.get("radius_km", 25),
                            }
                            if geo:
                                center_lat = geo["latitude"]
                                center_lng = geo["longitude"]
                        elif block.name == "search_knowledge_base":
                            results = await search_knowledge_base(
                                query=block.input["query"],
                                page_type=self.page_type,
                            )
                            result_json = json.dumps(results, default=str)
                        else:
                            result_json = json.dumps({"error": f"Unknown tool: {block.name}"})

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_json,
                        })

                messages.append({"role": "user", "content": tool_results})
                # After processing tool results, restore full tools for
                # any subsequent iterations (e.g. knowledge base follow-up)
                turn_tools = self.tools
            else:
                # Extract final text
                text_parts = [
                    block.text for block in assistant_content if block.type == "text"
                ]
                answer = "\n".join(text_parts)

                # Auto-recovery: if AI wrote results-like text without calling
                # the search tool, extract location and force a search so the
                # frontend gets listing cards to display.
                if not search_performed and _looks_like_results_text(answer):
                    location = self._extract_location_from_history(messages)
                    if location:
                        logging.warning(
                            f"[Fliss] AI wrote results text without calling search tool. "
                            f"Auto-recovering with location={location}, type={self.frontend_type}"
                        )
                        keywords = self._extract_keywords_from_history(messages)
                        tool_input = {"location": location, "keywords": keywords, "limit": 8}
                        result_json, raw_results, geo = await self._handle_search(tool_input)
                        if raw_results:
                            search_performed = True
                            listings_results = raw_results
                            filters_used = {
                                "location": location,
                                "keywords": keywords,
                                "radius_km": 25,
                            }
                            if geo:
                                center_lat = geo["latitude"]
                                center_lng = geo["longitude"]

                # Determine intent (matching existing system's values)
                if search_performed:
                    intent = "listings"
                    confidence = 1.0
                elif any(kw in answer.lower() for kw in [
                    "funding", "organisation", "charity", "cqc", "ofsted",
                    "allowance", "nhs", "council", "salary", "interview",
                ]):
                    intent = "info"
                    confidence = 0.9
                else:
                    intent = "clarify"
                    confidence = 0.8

                # Build title
                if search_performed and filters_used:
                    location = filters_used.get("location", "")
                    if self.frontend_type == "JOBS":
                        title = f"Jobs near {location}" if location else "Job results"
                    else:
                        title = f"Results near {location}" if location else "Search results"
                else:
                    title = ""

                # --- RULE 2: Wellbeing check-in before first results ---
                # If we've just performed a search and the wellbeing check-in
                # has never been offered in this conversation, defer the results
                # by one turn: ask the wellbeing question now, stash the results
                # as pending_results, and let the next turn return them.
                if (
                    search_performed
                    and not _wellbeing_checkin_done(messages[:-1])
                    and not _wellbeing_checkin_offered(messages[:-1])
                ):
                    return {
                        "intent": "clarify",
                        "confidence": 1.0,
                        "answer": WELLBEING_CHECKIN_QUESTION,
                        "results": [],
                        "title": "",
                        "center_lat": None,
                        "center_lng": None,
                        "filters_used": None,
                        "pending_results": {
                            "results": listings_results,
                            "title": title,
                            "center_lat": center_lat,
                            "center_lng": center_lng,
                            "filters_used": filters_used,
                        },
                    }

                # Alert: the model claimed to show listings but no results were
                # produced, so the frontend will render zero cards. This is the
                # signature of a model that stopped calling search_listings (the
                # Sonnet 4 -> 4.6 regression). Log loudly so it's caught fast.
                if not listings_results and _looks_like_results_text(answer):
                    logging.error(
                        "[Fliss] Model narrated results but produced none — no cards "
                        "will render. model=%s frontend_type=%s search_performed=%s "
                        "answer_first120=%r",
                        self.model, self.frontend_type, search_performed, answer[:120],
                    )

                print(f"[WB-DIAG] final_return answer_first200={answer[:200]!r} intent={intent}", flush=True)
                return {
                    "intent": intent,
                    "confidence": confidence,
                    "answer": answer,
                    "results": listings_results,
                    "title": title,
                    "center_lat": center_lat,
                    "center_lng": center_lng,
                    "filters_used": filters_used,
                }

    def _extract_location_from_history(self, messages: list[dict]) -> str | None:
        """Extract the most recent location from conversation history.

        Checks filters_used from previous searches first, then falls back
        to the search context injected into user messages.
        """
        # Check if a previous search had a location in filters_used
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                filters = msg.get("filters_used")
                if filters and filters.get("location"):
                    return filters["location"]
        # Fall back to search context metadata injected into user messages
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            # Look for location in search context
            match = re.search(r'"location":\s*"([^"]+)"', content)
            if match:
                return match.group(1)
        return None

    def _extract_keywords_from_history(self, messages: list[dict]) -> list[str]:
        """Extract keywords from conversation history."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                filters = msg.get("filters_used")
                if filters and filters.get("keywords"):
                    return filters["keywords"]
        return []

    async def _handle_search(self, tool_input: dict) -> tuple:
        """Execute search_listings with location-first fallback.

        Strategy:
        1. Try search with keywords if provided
        2. If keywords return 0 results, retry with location only
        3. Flag whether keyword filtering was applied or fell back

        Returns (json_str, raw_results, geo_dict).
        """
        location = tool_input["location"]
        geo = await geocode_location(location)

        latitude = geo["latitude"] if geo else None
        longitude = geo["longitude"] if geo else None

        # Default 25km radius — if nothing found, prompt user to expand
        radius = tool_input.get("radius_km", 25)
        keywords = tool_input.get("keywords")
        limit = tool_input.get("limit", 8)

        # First try: with keywords (if any)
        results = await search_listings(
            page_type=self.frontend_type,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius,
            keywords=keywords,
            limit=limit,
        )

        keyword_match = True

        # Fallback: if keywords returned nothing, retry location-only
        if not results and keywords:
            keyword_match = False
            results = await search_listings(
                page_type=self.frontend_type,
                latitude=latitude,
                longitude=longitude,
                radius_km=radius,
                keywords=None,
                limit=limit,
            )

        # Build response JSON with metadata about the search
        response_data = {
            "results": results,
            "keyword_match": keyword_match,
            "keywords_requested": keywords or [],
            "location": location,
            "result_count": len(results),
        }

        return json.dumps(response_data, default=str), results, geo

    async def _handle_job_search(self, tool_input: dict) -> tuple:
        """Execute search_jobs with keyword fallback.

        Returns (json_str, raw_results, geo_dict).
        """
        location = tool_input["location"]
        geo = await geocode_location(location)

        latitude = geo["latitude"] if geo else None
        longitude = geo["longitude"] if geo else None

        radius = tool_input.get("radius_km", 25)
        keywords = tool_input.get("keywords")
        job_type = tool_input.get("job_type")
        limit = tool_input.get("limit", 8)

        # First try: with keywords (if any)
        results = await search_jobs(
            latitude=latitude,
            longitude=longitude,
            radius_km=radius,
            keywords=keywords,
            job_type=job_type,
            limit=limit,
        )

        keyword_match = True

        # Fallback: if keywords returned nothing, retry location-only
        if not results and keywords:
            keyword_match = False
            results = await search_jobs(
                latitude=latitude,
                longitude=longitude,
                radius_km=radius,
                keywords=None,
                job_type=job_type,
                limit=limit,
            )

        response_data = {
            "results": results,
            "keyword_match": keyword_match,
            "keywords_requested": keywords or [],
            "location": location,
            "result_count": len(results),
        }

        return json.dumps(response_data, default=str), results, geo
