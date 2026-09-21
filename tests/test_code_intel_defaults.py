"""Tests for code_intel_defaults.py — the pre_llm_call coding default.

The defect these tests lock down: the hook used to read only
``kwargs["messages"]``, which the production call site
(``agent/turn_context.py::_collect_pre_llm_call_context``) never passes — it
passes ``user_message`` + ``conversation_history``. The hook was therefore a
permanent no-op in real sessions while its own tests stayed green.
"""

from code_intel import code_intel_defaults as d

PROD_KWARGS = {
    "session_id": "sess-1",
    "task_id": "task-1",
    "turn_id": "turn-1",
    "user_message": "fix the bug in the parser",
    "conversation_history": [],
    "is_first_turn": True,
    "model": "m",
    "platform": "cli",
}


def _fresh(session="sess-1"):
    d.forget_session(session)
    return session


# ---------------------------------------------------------------------------
# Production kwargs shape
# ---------------------------------------------------------------------------

def test_production_kwargs_produce_guidance():
    """The exact shape turn_context.py sends must yield guidance."""
    d.forget_session("sess-prod")
    out = d.pre_llm_call_context(**{**PROD_KWARGS, "session_id": "sess-prod"})
    assert out is not None
    assert "code_symbols" in out and "code_references" in out


def test_normalize_messages_covers_all_supported_shapes():
    norm = d.normalize_messages(
        user_message="hello",
        messages=[{"role": "user", "content": "from messages"}],
        conversation_history=[{"role": "user", "content": "from history"}],
    )
    assert [m["content"] for m in norm] == ["from history", "from messages", "hello"]


def test_normalize_messages_accepts_object_messages():
    class Msg:
        role = "user"
        content = "object-shaped"

    norm = d.normalize_messages(conversation_history=[Msg()])
    assert norm == [{"role": "user", "content": "object-shaped"}]


def test_normalize_messages_ignores_junk():
    assert d.normalize_messages(user_message=None, messages="not-a-list",
                                conversation_history=None) == []
    assert d.normalize_messages(user_message="   ") == []


# ---------------------------------------------------------------------------
# One-shot, session-scoped, bounded
# ---------------------------------------------------------------------------

def test_guidance_is_one_shot_per_session():
    s = _fresh("sess-once")
    first = d.pre_llm_call_context(**{**PROD_KWARGS, "session_id": s})
    second = d.pre_llm_call_context(**{**PROD_KWARGS, "session_id": s})
    assert first is not None
    assert second is None


def test_no_session_key_fails_closed():
    d.forget_session("")
    assert d.build_default_guidance([{"role": "user", "content": "fix bug"}], None, None) is None


def test_non_string_session_key_fails_closed():
    """A non-string key must not be coerced into a shared bucket."""
    assert d.build_default_guidance([{"role": "user", "content": "fix bug"}], 12345, None) is None
    assert d.build_default_guidance([{"role": "user", "content": "fix bug"}], "", "  ") is None
    assert d.take_context_slot(12345, ["list"]) is False


def test_non_coding_turn_gets_nothing():
    s = _fresh("sess-noncode")
    out = d.pre_llm_call_context(**{**PROD_KWARGS, "session_id": s,
                                    "user_message": "what is the weather in zurich"})
    assert out is None


def test_task_id_is_used_when_session_id_missing():
    d.forget_session(None, "task-only")
    out = d.pre_llm_call_context(**{**PROD_KWARGS, "session_id": None, "task_id": "task-only"})
    assert out is not None


def test_state_is_lru_bounded():
    for i in range(d._MAX_SESSIONS + 20):
        d.build_default_guidance([{"role": "user", "content": "code"}], "lru-%d" % i)
    assert d.state_sizes()["guidance_sessions"] <= d._MAX_SESSIONS


def test_forget_session_releases_both_stores():
    s = _fresh("sess-forget")
    d.build_default_guidance([{"role": "user", "content": "code"}], s)
    assert d.take_context_slot(s) is True
    before = d.state_sizes()
    d.forget_session(s)
    after = d.state_sizes()
    assert after["guidance_sessions"] == before["guidance_sessions"] - 1
    assert after["context_sessions"] == before["context_sessions"] - 1
    # and the forgotten session really gets guidance again
    assert d.build_default_guidance([{"role": "user", "content": "code"}], s) is not None


# ---------------------------------------------------------------------------
# Context-injection budget
# ---------------------------------------------------------------------------

def test_context_slot_budget_binds_exactly():
    s = _fresh("sess-budget")
    assert [d.take_context_slot(s) for _ in range(d._MAX_CONTEXT_INJECTIONS + 3)] == (
        [True] * d._MAX_CONTEXT_INJECTIONS + [False] * 3
    )


def test_context_slot_without_session_key_is_refused():
    assert d.take_context_slot(None, None) is False


def test_guidance_text_allows_bounded_fallback():
    """Steering must never read as a hard block on read_file/search_files/terminal."""
    text = d.build_default_guidance([{"role": "user", "content": "code"}], "sess-text")
    assert text is not None
    assert "bounded text/shell search" in text
    assert "unavailable" in text and "insufficient" in text
    for forbidden in ("do not use read_file", "never use", "forbidden", "must not"):
        assert forbidden not in text.lower()


# ---------------------------------------------------------------------------
# Concurrency: pre_llm_call runs on hook worker threads
# ---------------------------------------------------------------------------

def test_concurrent_guidance_no_crash_and_no_double_grant():
    """Shared OrderedDict state is touched from hook worker threads.

    Without the module lock, `move_to_end` + `popitem` interleave and raise
    KeyError / corrupt ordering. Assert: (a) no thread errors, (b) exactly one
    worker gets guidance for a session, (c) the LRU cap still holds under load.
    """
    import threading

    errors: list = []
    grants: dict = {}
    start = threading.Barrier(8)

    def worker(i):
        try:
            start.wait(timeout=5)
            for k in range(40):
                sess = "conc-%d-%d" % (i, k % 3)
                out = d.build_default_guidance(
                    [{"role": "user", "content": "refactor this"}], sess
                )
                if out:
                    grants.setdefault(sess, []).append(i)
                d.take_context_slot(sess)
                d.state_sizes()
        except BaseException as exc:  # noqa: BLE001 - the assertion IS the point
            errors.append("%s: %s" % (type(exc).__name__, exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
        assert not t.is_alive()

    assert errors == [], errors
    assert all(len(v) == 1 for v in grants.values()), grants
    assert d.state_sizes()["guidance_sessions"] <= d._MAX_SESSIONS


def test_concurrent_context_budget_never_over_grants():
    """Exactly `_MAX_CONTEXT_INJECTIONS` slots per session under contention."""
    import threading

    d.forget_session("conc-budget")
    granted: list = []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                ok = d.take_context_slot("conc-budget")
            except Exception as exc:  # noqa: BLE001
                granted.append("ERR:%s" % exc)
                return
            if not ok:
                return
            with lock:
                granted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
        assert not t.is_alive()

    assert all(g == 1 for g in granted), granted
    assert len(granted) == d._MAX_CONTEXT_INJECTIONS, len(granted)
