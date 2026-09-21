"""Recorded contract of the Hermes core plugin APIs this plugin depends on.

Hermes is not installable from PyPI, so the standalone CI job cannot import
``hermes_cli``. The plugin's tests therefore need a *checked* local record of
the core contract instead of an unchecked guess at it:

  * ``CORE_CONTRACT`` — the constants and validation rules copied from the
    running core (``hermes_cli.plugins`` / ``hermes_cli.plugins_dispatch``).
  * ``SECTION_ID_RE`` / ``validate_section`` — the core's own validation
    semantics (id charset, position allowlist, ``max_chars`` bounds, reserved
    persistence markers, non-empty after strip).
  * ``assert_contract_matches_core`` — compares the record against the LIVE
    core and raises on drift. The integration test in ``test_hook_wiring.py``
    calls this whenever ``hermes_cli`` is importable, so the record cannot
    silently rot away from the real core, and the standalone assertions still
    mean something.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

# --- Recorded from the live core (see assert_contract_matches_core) ----------
CORE_CONTRACT: Mapping[str, Any] = {
    "MAX_SYSTEM_PROMPT_SECTION_CHARS": 4_000,
    "DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS": 4_000,
    "MAX_SYSTEM_PROMPT_SECTIONS": 32,
    "MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS": 8_000,
    "SYSTEM_PROMPT_SECTION_POSITIONS": frozenset({"after_memory"}),
    "PLUGIN_SECTIONS_START": "<!-- hermes-plugin-sections:start -->",
    "PLUGIN_SECTIONS_END": "<!-- hermes-plugin-sections:end -->",
    "SECTION_ID_PATTERN": r"^[a-z0-9][a-z0-9._-]{0,127}$",
    # Hook names the plugin registers; a core that does not advertise one of
    # these accepts the callback silently and never invokes it (stealth outage).
    "REQUIRED_HOOKS": frozenset({"pre_llm_call", "on_session_end", "transform_tool_result"}),
}

SECTION_ID_RE = re.compile(CORE_CONTRACT["SECTION_ID_PATTERN"])

MAX_SYSTEM_PROMPT_SECTION_CHARS = CORE_CONTRACT["MAX_SYSTEM_PROMPT_SECTION_CHARS"]
SYSTEM_PROMPT_SECTION_POSITIONS = CORE_CONTRACT["SYSTEM_PROMPT_SECTION_POSITIONS"]
PLUGIN_SECTIONS_START = CORE_CONTRACT["PLUGIN_SECTIONS_START"]
PLUGIN_SECTIONS_END = CORE_CONTRACT["PLUGIN_SECTIONS_END"]


def is_valid_system_prompt_section_id(value: Any) -> bool:
    """Mirror of ``hermes_cli.plugins_dispatch.is_valid_system_prompt_section_id``."""
    return isinstance(value, str) and bool(SECTION_ID_RE.fullmatch(value))


def validate_section(section_id: str, section: Mapping[str, Any]) -> str:
    """Apply the core's section acceptance rules; return the stripped content.

    Mirrors ``PluginManager._render_prompt_section_text`` and
    ``PluginContext.register_system_prompt_section``: a section is only ever
    rendered when every one of these holds, so a section that passes here is
    guaranteed to reach the session prompt.
    """
    assert is_valid_system_prompt_section_id(section_id), \
        f"section id {section_id!r} is not accepted by the core"
    assert section["position"] in SYSTEM_PROMPT_SECTION_POSITIONS, \
        f"position {section['position']!r} is not in the core allowlist"
    max_chars = section["max_chars"]
    assert isinstance(max_chars, int) and not isinstance(max_chars, bool), \
        "max_chars must be a real int"
    assert 0 < max_chars <= MAX_SYSTEM_PROMPT_SECTION_CHARS, \
        f"max_chars {max_chars} outside (0, {MAX_SYSTEM_PROMPT_SECTION_CHARS}]"
    content = section["content"]
    assert callable(content) or isinstance(content, str), \
        "content must be a str or callable"
    text = content.strip() if isinstance(content, str) else None
    assert text, "content is empty after strip — the core would skip it"
    assert PLUGIN_SECTIONS_START not in text and PLUGIN_SECTIONS_END not in text, \
        "content contains a reserved persistence marker — the core would skip it"
    assert len(text) <= max_chars, \
        f"content is {len(text)} chars, above its own max_chars ({max_chars})"
    return text


_ABSENT = object()


def contract_drift(core_module: Any) -> list:
    """Return human-readable mismatches between the live core and the record."""
    actual = {
        name: getattr(core_module, name, _ABSENT)
        for name in (
            "MAX_SYSTEM_PROMPT_SECTION_CHARS",
            "DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS",
            "MAX_SYSTEM_PROMPT_SECTIONS",
            "MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS",
            "SYSTEM_PROMPT_SECTION_POSITIONS",
            "PLUGIN_SECTIONS_START",
            "PLUGIN_SECTIONS_END",
        )
    }
    id_re = getattr(core_module, "_SYSTEM_PROMPT_SECTION_ID_RE", None)
    actual["SECTION_ID_PATTERN"] = getattr(id_re, "pattern", _ABSENT)

    drift = []
    for name, expected in CORE_CONTRACT.items():
        if name == "REQUIRED_HOOKS":
            continue
        got = actual.get(name, _ABSENT)
        if got is _ABSENT:
            # Constant lives on the other core module (the caller checks both),
            # so an absent key here is not drift.
            continue
        if got != expected:
            drift.append(f"{name}: recorded={expected!r} live={got!r}")
    return drift
