from contextlib import asynccontextmanager
import logging
import re
from typing import Optional, List, Any
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from chat.engine import ConversationEngine
from chat.persistence import save_conversation
from tools.search import close_pool


def _check_model_config() -> None:
    """Warn loudly at startup if the configured model is retired.

    Offline check against config.RETIRED_MODELS — no network call, so it can
    never itself delay or fail startup. Turns a silent in-production 404 into a
    visible deploy-time log line.
    """
    try:
        from config import get_settings, resolve_model, RETIRED_MODELS

        raw = get_settings().fliss_model
        effective = resolve_model(raw)
        if effective in RETIRED_MODELS:
            logging.critical(
                "FLISS_MODEL=%r resolves to %r, which Anthropic has RETIRED — "
                "requests will 404. Set FLISS_MODEL to an active model alias "
                "(e.g. claude-sonnet-4-5).",
                raw, effective,
            )
        else:
            logging.info("Fliss model config OK: FLISS_MODEL=%r -> %r", raw, effective)
    except Exception:
        # Never let the model check break startup.
        logging.exception("Model config check failed (non-fatal)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_model_config()
    yield
    await close_pool()


app = FastAPI(
    title="Fliss",
    description="AI conversational search engine for Caretopia",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Frontend-compatible API contract ─────────────────────────────────────────

class QueryContext(BaseModel):
    session_id: str = "default"


class QueryRequest(BaseModel):
    query: str
    mode: str = "text"
    context: QueryContext = Field(default_factory=QueryContext)
    type: str = "CAREHOME"  # DEFAULT | CAREHOME | NURSERY | HOMECARE | JOBS


class QueryResponse(BaseModel):
    intent: str
    confidence: float
    answer: str
    results: List[Any] = []
    title: str = ""
    center_lat: Optional[float] = None
    center_lng: Optional[float] = None


# In-memory conversation history per session.
# Bounded so it can't grow without limit and slowly OOM the service: when the
# cap is reached, the oldest-created session is evicted (insertion order).
_sessions: dict = {}
_MAX_SESSIONS = 2000

VALID_PAGE_TYPES = {"CAREHOME", "NURSERY", "HOMECARE", "JOBS"}

CARE_TYPE_LABELS = {
    "CAREHOME": "care homes",
    "HOMECARE": "home care",
    "NURSERY": "day nurseries",
    "JOBS": "care jobs",
}

FIRST_STEP_REPLY = (
    "Of course. Are you looking for a care home, home care, or a day nursery?"
)

BACKEND_FALLBACK_REPLY = (
    "I'm here, but I'm having trouble reaching live results right now. "
    "Are you looking for a care home, home care, or a day nursery?"
)


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _infer_page_type(query: str) -> Optional[str]:
    text = _normalise_text(query)
    if any(term in text for term in ["nursery", "nurseries", "childcare", "child care"]):
        return "NURSERY"
    if any(term in text for term in ["home care", "homecare", "care at home", "domiciliary"]):
        return "HOMECARE"
    if any(term in text for term in ["care home", "carehome", "residential", "nursing home"]):
        return "CAREHOME"
    if any(term in text for term in ["job", "jobs", "career", "vacancy", "work in care"]):
        return "JOBS"
    return None


def _normalise_page_type(raw_type: str, query_text: str) -> str:
    page_type = (raw_type or "").upper()
    if page_type in VALID_PAGE_TYPES:
        return page_type
    return _infer_page_type(query_text) or "CAREHOME"


def _is_greeting_or_first_step(query: str) -> bool:
    text = _normalise_text(query)
    if not text:
        return True
    return text in {
        "hi", "hello", "hey", "hiya", "help", "start", "get started",
        "i need help", "can you help", "can you help me",
    }


def _is_care_type_only(query: str) -> bool:
    text = _normalise_text(query)
    care_type = _infer_page_type(text)
    if not care_type or care_type == "JOBS":
        return False
    filler_removed = re.sub(
        r"\b(i|we|need|want|am|i'm|looking|for|a|an|the|find|please|me|some)\b",
        " ",
        text,
    )
    filler_removed = _normalise_text(filler_removed)
    return filler_removed in {
        "care home", "care homes", "carehome", "home care", "homecare",
        "day nursery", "nursery", "nurseries", "childcare", "child care",
    }


def _first_step_response(query_text: str, page_type: str) -> Optional[QueryResponse]:
    if _is_greeting_or_first_step(query_text):
        return QueryResponse(
            intent="clarify",
            confidence=1.0,
            answer=FIRST_STEP_REPLY,
            results=[],
        )

    if _is_care_type_only(query_text):
        label = CARE_TYPE_LABELS.get(page_type, "care")
        return QueryResponse(
            intent="clarify",
            confidence=1.0,
            answer=f"Great - I can help with {label}. Where should I search?",
            results=[],
        )

    return None


def _fallback_response(page_type: str) -> QueryResponse:
    return QueryResponse(
        intent="clarify",
        confidence=0.0,
        answer=BACKEND_FALLBACK_REPLY,
        results=[],
    )



@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    session_id = req.context.session_id or "default"
    page_type = _normalise_page_type(req.type, req.query)

    try:
        # Get or create conversation history for this session
        session_key = f"{session_id}:{page_type}"
        if session_key not in _sessions:
            if len(_sessions) >= _MAX_SESSIONS:
                # Evict the oldest-created session to bound memory.
                _sessions.pop(next(iter(_sessions)), None)
            _sessions[session_key] = []
        history = _sessions[session_key]

        engine = ConversationEngine(frontend_type=page_type)
        result = await engine.chat(
            message=req.query,
            conversation_history=history,
        )

        # Append to conversation history for next turn
        history.append({"role": "user", "content": req.query})
        # Store the answer as-is — no metadata in the content
        stored_content = result["answer"]
        filters_used = result.get("filters_used")
        assistant_msg = {"role": "assistant", "content": stored_content}
        # Store search metadata separately so engine.py can inject it
        # as context without it leaking into the AI's visible response text
        if filters_used:
            assistant_msg["filters_used"] = filters_used
        if result.get("results"):
            assistant_msg["results"] = result["results"]
            assistant_msg["title"] = result.get("title", "")
            assistant_msg["center_lat"] = result.get("center_lat")
            assistant_msg["center_lng"] = result.get("center_lng")
        # Persist deferred results across the wellbeing check-in turn
        if result.get("pending_results"):
            assistant_msg["pending_results"] = result["pending_results"]
        history.append(assistant_msg)

        # Fire-and-forget persistence of the full conversation for analysis.
        location = None
        for m in reversed(history):
            f = m.get("filters_used") if isinstance(m, dict) else None
            if f and f.get("location"):
                location = f["location"]
                break
        try:
            save_conversation(
                session_id=session_id,
                user_type=page_type,
                location=location,
                messages=history,
            )
        except Exception:
            logging.exception("Failed to save conversation")

        return QueryResponse(**result)
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        logging.error(
            "Failed to process /api/query: %s status=%s",
            type(exc).__name__,
            status_code,
        )
        logging.exception("Failed to process /api/query")
        return _fallback_response(page_type)

# ── History endpoint ─────────────────────────────────────────────────────────

@app.get("/api/history/{session_id}")
async def history(
    session_id: str,
    page_type: str = Query(default="CAREHOME"),
    limit: int = Query(default=20, ge=1, le=100),
):
    session_key = f"{session_id}:{page_type}"
    messages = _sessions.get(session_key, [])
    return {"session_id": session_id, "page_type": page_type, "messages": messages[-limit:]}


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Test chat UI ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def test_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fliss v2 — Test Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; height: 100vh; display: flex; flex-direction: column; }
  .header { background: #2563eb; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 16px; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header select { padding: 4px 8px; border-radius: 4px; border: none; }
  .chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
  .msg { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; white-space: pre-wrap; }
  .msg.user { background: #2563eb; color: white; align-self: flex-end; border-bottom-right-radius: 4px; }
  .msg.assistant { background: white; color: #1f2937; align-self: flex-start; border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .msg.error { background: #fee2e2; color: #991b1b; align-self: center; }
  .results-card { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px; margin-top: 8px; font-size: 13px; }
  .result-item { padding: 8px 0; border-bottom: 1px solid #e5e7eb; }
  .result-item:last-child { border-bottom: none; }
  .result-name { font-weight: 600; color: #1f2937; }
  .result-detail { color: #6b7280; font-size: 12px; }
  .meta { font-size: 11px; color: #6b7280; margin-top: 6px; padding: 4px 0; }
  .input-bar { display: flex; gap: 8px; padding: 16px 24px; background: white; border-top: 1px solid #e5e7eb; }
  .input-bar input { flex: 1; padding: 12px 16px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 15px; outline: none; }
  .input-bar input:focus { border-color: #2563eb; }
  .input-bar button { padding: 12px 24px; background: #2563eb; color: white; border: none; border-radius: 8px; font-size: 15px; cursor: pointer; }
  .input-bar button:disabled { opacity: 0.5; cursor: not-allowed; }
  .typing { color: #6b7280; font-style: italic; align-self: flex-start; padding: 8px 16px; }
</style>
</head>
<body>
<div class="header">
  <h1>Fliss v2</h1>
  <span style="font-weight:300;font-size:14px;">Caretopia AI Search</span>
  <select id="pageType">
    <option value="CAREHOME">Care Homes</option>
    <option value="NURSERY">Nurseries</option>
    <option value="HOMECARE">Home Care</option>
    <option value="JOBS">Jobs</option>
  </select>
</div>
<div class="chat" id="chat"></div>
<div class="input-bar">
  <input type="text" id="input" placeholder="Type your message..." autofocus>
  <button id="send" onclick="sendMessage()">Send</button>
</div>
<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const pageTypeEl = document.getElementById('pageType');
const sessionId = crypto.randomUUID();

inputEl.addEventListener('keydown', e => { if (e.key === 'Enter' && !sendBtn.disabled) sendMessage(); });

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addResults(results, data) {
  if (!results || results.length === 0) return;
  const container = document.createElement('div');
  container.className = 'msg assistant';

  let html = '<div class="results-card">';
  results.forEach(r => {
    const name = r.organisationName || r.name || 'Unknown';
    const loc = [r.townCity, r.postcode].filter(Boolean).join(', ');
    const grade = r.cqcGrade || r.ofstedGrade || '';
    const dist = r.distance_km != null ? r.distance_km + ' km' : '';
    const phone = r.contactPhone || '';
    html += '<div class="result-item">';
    html += '<div class="result-name">' + name + '</div>';
    html += '<div class="result-detail">' + [loc, grade, dist, phone].filter(Boolean).join(' · ') + '</div>';
    if (r.description) html += '<div class="result-detail">' + r.description + '</div>';
    html += '</div>';
  });
  html += '</div>';
  html += '<div class="meta">Intent: ' + data.intent + ' | Title: ' + data.title + '</div>';
  container.innerHTML = html;
  chatEl.appendChild(container);
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function sendMessage() {
  const msg = inputEl.value.trim();
  if (!msg) return;
  inputEl.value = '';
  addMsg('user', msg);
  sendBtn.disabled = true;

  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.textContent = 'Fliss is thinking...';
  chatEl.appendChild(typing);

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        query: msg,
        mode: 'text',
        context: { session_id: sessionId },
        type: pageTypeEl.value,
      }),
    });
    typing.remove();
    const data = await res.json();

    if (!res.ok) {
      addMsg('error', 'Error: ' + (data.detail || res.statusText));
    } else {
      addMsg('assistant', data.answer);
      if (data.results && data.results.length > 0) addResults(data.results, data);
    }
  } catch (err) {
    typing.remove();
    addMsg('error', 'Network error: ' + err.message);
  }
  sendBtn.disabled = false;
  inputEl.focus();
}
</script>
</body>
</html>"""
