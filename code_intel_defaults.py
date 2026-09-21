"""Bounded default guidance for coding turns.

The ``pre_llm_call`` hook invokes this before tool selection. It gives each
session one compact semantic-navigation default, without blocking normal file
or shell tools when semantic tools cannot answer the question.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional

_MAX_SESSIONS = 50
_MAX_CONTEXT_INJECTIONS = 3

# Coding-turn classifier. Deliberately broad but keyword-based: a false
# positive costs one ~350-char line, a false negative costs the whole point of
# the guidance (the agent never learns the semantic tools exist).
_CODING_TERMS = (
    "code", "coding", "bug", "fix", "refactor", "function", "class",
    "method", "test", "typescript", "javascript", "python", "tsx", "api",
    "implement", "compile", "build", "diagnostic", "error", "stack trace",
)

_seen_sessions: "OrderedDict[str, None]" = OrderedDict()
_context_counts: "OrderedDict[str, int]" = OrderedDict()

# pre_llm_call handlers run on the hook worker threads (hermes_cli.plugins_
# dispatch), so the shared LRU state above is touched from more than one thread.
# OrderedDict mutation (move_to_end + popitem) is NOT atomic: unsynchronized
# concurrent access can raise KeyError or corrupt the ordering. One module-level
# lock; the critical sections are dict ops only (no I/O while held).
_state_lock = Lock()


def _lru_touch(
    store: "OrderedDict[str, Any]", key: str, value: Any = None, *, under_lock: bool = False
) -> Any:
    """Get-or-create *key*, LRU-bump it, evict the oldest session past the cap.

    Callers that already hold ``_state_lock`` pass ``under_lock=True`` to avoid
    re-entering it (``Lock`` is not reentrant).
    """
    if under_lock:
        return _lru_touch_locked(store, key, value)
    with _state_lock:
        return _lru_touch_locked(store, key, value)


def _lru_touch_locked(store: "OrderedDict[str, Any]", key: str, value: Any = None) -> Any:
    if key in store:
        store.move_to_end(key)
        return store[key]
    store[key] = value
    if len(store) > _MAX_SESSIONS:
        store.popitem(last=False)
    return store[key]


def _to_message(msg: Any) -> Optional[Dict[str, Any]]:
    if isinstance(msg, dict):
        return msg
    role = getattr(msg, "role", None)
    if role is not None:
        return {"role": role, "content": getattr(msg, "content", "")}
    return None


def normalize_messages(
    *,
    user_message: Any = None,
    messages: Any = None,
    conversation_history: Any = None,
) -> List[Dict[str, Any]]:
    """Normalize the several shapes Hermes passes to ``pre_llm_call``.

    The production call site (``agent/turn_context.py::_collect_pre_llm_call_context``)
    passes ``user_message`` + ``conversation_history`` — never ``messages``.
    Reading only ``messages`` (as this plugin used to) makes the hook
    permanently no-op in real sessions while still passing hand-written tests,
    so every supported shape is folded in here.
    """
    out: List[Dict[str, Any]] = []
    history = conversation_history if isinstance(conversation_history, (list, tuple)) else []
    for msg in history:
        norm = _to_message(msg)
        if norm is not None:
            out.append(norm)
    for msg in messages if isinstance(messages, (list, tuple)) else []:
        norm = _to_message(msg)
        if norm is not None:
            out.append(norm)
    if isinstance(user_message, str) and user_message.strip():
        out.append({"role": "user", "content": user_message})
    return out


def _last_user_text(messages: Iterable[Mapping[str, Any]]) -> str:
    for message in reversed(list(messages)):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str):
            return content
    return ""


def is_coding_turn(messages: Iterable[Mapping[str, Any]]) -> bool:
    """Conservative classifier; no guidance for ordinary conversation."""
    text = _last_user_text(messages).lower()
    if not text:
        return False
    return any(term in text for term in _CODING_TERMS)


def _session_key(session_id: Any, task_id: Any) -> Optional[str]:
    """Only accept non-empty string keys; anything else fails closed."""
    for candidate in (session_id, task_id):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def build_default_guidance(
    messages: Iterable[Mapping[str, Any]],
    session_id: Optional[str],
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Return one compact coding default per bounded session, or ``None``.

    No usable string session key means fail closed: never pool unrelated agents
    into shared guidance state. Fallback tools stay explicitly allowed — this
    steers the default, it never blocks read_file/search_files/terminal.
    """
    session_key = _session_key(session_id, task_id)
    if not session_key or not is_coding_turn(messages):
        return None
    with _state_lock:
        if session_key in _seen_sessions:
            return None
        _lru_touch(_seen_sessions, session_key, None, under_lock=True)
    return (
        "[code-intel default] For code work, start with code_symbols/code_search "
        "or code_definition; use code_references before rename/refactor. Read targeted "
        "ranges or use bounded text/shell search only when semantic output is unavailable "
        "or insufficient. Run code_diagnostics after edits."
    )


def pre_llm_call_context(**kwargs: Any) -> Optional[str]:
    """``pre_llm_call`` handler: one coding default per session (fail closed)."""
    messages = normalize_messages(
        user_message=kwargs.get("user_message"),
        messages=kwargs.get("messages"),
        conversation_history=kwargs.get("conversation_history"),
    )
    return build_default_guidance(messages, kwargs.get("session_id"), kwargs.get("task_id"))


def take_context_slot(session_id: Optional[str], task_id: Optional[str] = None) -> bool:
    """Bound file-context injections per session (cheap rate limit, no I/O)."""
    session_key = _session_key(session_id, task_id)
    if not session_key:
        return False
    with _state_lock:
        count = _lru_touch(_context_counts, session_key, 0, under_lock=True)
        if count >= _MAX_CONTEXT_INJECTIONS:
            return False
        _context_counts[session_key] = count + 1
        return True


def forget_session(session_id: Optional[str], task_id: Optional[str] = None) -> None:
    """Release per-session default state at session end."""
    session_key = _session_key(session_id, task_id)
    if session_key:
        with _state_lock:
            _seen_sessions.pop(session_key, None)
            _context_counts.pop(session_key, None)


def state_sizes() -> Dict[str, int]:
    """Test/diagnostic helper — not on the hot path."""
    with _state_lock:
        return {
            "guidance_sessions": len(_seen_sessions),
            "context_sessions": len(_context_counts),
        }

