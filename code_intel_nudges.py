"""Mechanical steering: append short code_intel tips to tool RESULTS.

Why this exists (see VERIFICATION.md): the pre-existing `_CODE_INTEL_STEERING`
text block in `__init__.py` is spliced into `tools.delegate_tool._build_child_system_prompt`,
so it only ever reaches subagents spawned via the in-process `delegate_task`
tool. Paseo-spawned agents (and any directly-run session) never see it —
confirmed by 30 days of session-DB forensics showing 2.5k code_intel calls
against ~104k terminal/read_file/search_files calls.

This module hooks `transform_tool_result` instead. That hook fires from
`model_tools.handle_function_call` for every tool call, regardless of how the
agent process was spawned (Paseo, delegate_task, or a plain CLI session), so
the nudge is mechanical rather than dependent on a system-prompt path.

Design constraints (see AUFTRAG M1/M3):
- one-line hint, appended to the tool result string (never blocks/redirects)
- max _MAX_NUDGE_PER_TOOL nudges PER (session, trigger-type). There are 4
  independent trigger-types (read_file, read_file_repeat, search_files,
  terminal), each with its OWN budget of _MAX_NUDGE_PER_TOOL — so a session
  can see up to 4 * _MAX_NUDGE_PER_TOOL nudges total, not _MAX_NUDGE_PER_TOOL
  overall. This is intentional (each trigger teaches a different tool) but
  easy to misread as a single global cap, so it is spelled out here.
- only fires for real source files (_SOURCE_EXTENSIONS whitelist) — a
  code_symbols call on a .md/.json/.log file throws "Unsupported language"
  and would teach the model the opposite lesson
- code_symbols nudge only on files >= _MIN_LINES_FOR_SYMBOLS_NUDGE lines

State bounding (review fix B1): module-level dicts are per-process and used
to survive across tool calls within a session, so they cannot be function
locals. Long-lived processes (Gateway, Paseo) never restart between
sessions, so an unbounded dict is a real leak: every session_id and every
distinct path ever read stays in memory forever. Fixed with a hard LRU cap
(OrderedDict, evict-oldest) on both the number of tracked sessions and the
number of tracked paths per session — no TTL, because a plain LRU-by-cap
already bounds worst-case memory deterministically (cap * cap entries) and
needs no wall-clock bookkeeping. `forget_session()` additionally lets the
`on_session_end` hook (present in Hermes's VALID_HOOKS, see __init__.py)
proactively drop a session's state the moment it ends, so eviction usually
never even needs to trigger in practice.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

# Real source-code extensions tree-sitter/ast-grep actually support well.
# Deliberately narrow (see M3): never .md/.json/.yaml/.yml/.env/.log/.txt/.sh/.sql,
# never Dockerfile/Makefile, never extension-less files.
_SOURCE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".rs", ".go", ".java",
}

# Max nudges per (session_id, trigger-type). Extends the same budget the
# dead CodeIntelSteering class used (_MAX_NUDGE_PER_TOOL = 3), just wired
# through a hook that actually fires. NOTE: this is PER trigger-type — see
# module docstring, up to 4x this value per session across all triggers.
_MAX_NUDGE_PER_TOOL = 3

# Below this, reading the whole file is cheaper than a code_symbols round-trip.
_MIN_LINES_FOR_SYMBOLS_NUDGE = 200

# Hard caps on in-memory state (review fix B1). Worst case memory is bounded
# by _MAX_SESSIONS * _MAX_PATHS_PER_SESSION entries, independent of uptime.
_MAX_SESSIONS = 50
_MAX_PATHS_PER_SESSION = 200

# Nudge text embeds the triggering path (read_file_repeat case). Long paths
# would make nudge byte-size scale linearly with path length (review fix B5).
_MAX_NUDGE_PATH_CHARS = 60

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{2,}$")
_DECL_RE = re.compile(r"\b(class|function|def|export const)\s+[A-Za-z_]")
_GREP_RE = re.compile(r"\b(grep|rg|find)\b")

# In-memory, per-process. Best-effort session scoping for a soft rate limit —
# resets on process restart, which is acceptable (worst case: a few extra
# nudges after a restart, never a burst within one session). Bounded LRU
# (see module docstring / B1): oldest session/path evicted once over cap.
_nudge_counts: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
_read_counts: "OrderedDict[str, OrderedDict[str, int]]" = OrderedDict()


def _lru_touch_session(
    store: "OrderedDict[str, Any]", session_key: str, factory
) -> Any:
    """Get-or-create *session_key*'s bucket in *store*, LRU-bump it, and evict
    the least-recently-used session if the cap is exceeded."""
    if session_key in store:
        store.move_to_end(session_key)
        return store[session_key]
    bucket = factory()
    store[session_key] = bucket
    if len(store) > _MAX_SESSIONS:
        store.popitem(last=False)
    return bucket


def _lru_bump_path(bucket: "OrderedDict[str, int]", path: str) -> int:
    """Increment *path*'s counter inside a per-session LRU-capped bucket,
    evicting the least-recently-used path if over cap. Returns new count."""
    if path in bucket:
        bucket.move_to_end(path)
        bucket[path] += 1
    else:
        bucket[path] = 1
        if len(bucket) > _MAX_PATHS_PER_SESSION:
            bucket.popitem(last=False)
    return bucket[path]


def forget_session(session_key: Optional[str]) -> None:
    """Drop all nudge state for *session_key*. Wired to `on_session_end` in
    __init__.py so long-lived processes don't rely solely on LRU eviction."""
    if not session_key:
        return
    _nudge_counts.pop(session_key, None)
    _read_counts.pop(session_key, None)


def _state_sizes() -> Dict[str, int]:
    """Introspection helper for tests/diagnostics — not used on the hot path."""
    return {
        "sessions_in_nudge_counts": len(_nudge_counts),
        "sessions_in_read_counts": len(_read_counts),
        "total_paths_tracked": sum(len(v) for v in _read_counts.values()),
    }


def _truncate_path_for_nudge(path: str) -> str:
    """Cap embedded path length so nudge byte-size can't scale with path length."""
    if len(path) <= _MAX_NUDGE_PATH_CHARS:
        return path
    return "..." + path[-_MAX_NUDGE_PATH_CHARS:]


def _is_source_path(path: str) -> bool:
    """True only for a real, existing source file (review fix B2).

    Suffix-only checks previously treated a directory named `folder.ts`, a
    non-existent path, or a broken symlink as a source file. `Path.is_file()`
    follows symlinks, so a symlink pointing at a real file is still allowed;
    a dangling symlink or any OSError (permission, race) fails closed (no
    nudge) rather than risk a wrong-lesson nudge on a bad path.
    """
    if not path:
        return False
    try:
        p = Path(path)
        if p.suffix.lower() not in _SOURCE_EXTENSIONS:
            return False
        return p.is_file()
    except OSError:
        return False


def _is_source_dir(path: str) -> bool:
    # Cap the flat iterdir scan so grep/find on huge non-source dirs
    # (node_modules, dist, .git) can't do thousands of stat() calls per
    # terminal call. 300 entries: a real source dir has at least one
    # source file among its first handful; node_modules-style trees exceed
    # the cap without one and correctly fail closed (no nudge).
    _MAX_SCAN_ENTRIES = 300
    try:
        p = Path(path)
        if not p.is_dir():
            return False
        for i, c in enumerate(p.iterdir()):
            if i >= _MAX_SCAN_ENTRIES:
                return False
            if c.is_file() and c.suffix.lower() in _SOURCE_EXTENSIONS:
                return True
        return False
    except OSError:
        return False


def _iter_candidate_paths(command: str):
    """Yield path-like tokens from a shell command (source dir/file refs).

    Used by the terminal nudge to catch `grep -r X apps/unified-api/app`
    style invocations where the target is a real source path but the
    command string contains no literal source extension (the old suffix
    check missed these — the dominant volume of on-source greps).
    """
    for tok in re.split(r"\s+", command):
        t = tok.strip("'\"").rstrip(";,&|")
        if not t:
            continue
        if t.startswith(("-",)) or t in ("~",):
            continue
        if "/" in t or t in (".", "..") or t.startswith("~"):
            yield t


def _take_nudge_slot(session_key: str, trigger: str) -> bool:
    counts = _lru_touch_session(_nudge_counts, session_key, dict)
    n = counts.get(trigger, 0)
    if n >= _MAX_NUDGE_PER_TOOL:
        return False
    counts[trigger] = n + 1
    return True


def _result_total_lines(result: Any) -> Optional[int]:
    """Best-effort extraction of total_lines from a read_file JSON-ish result."""
    try:
        data = json.loads(result) if isinstance(result, str) else result
        if isinstance(data, dict) and isinstance(data.get("total_lines"), int):
            return data["total_lines"]
    except Exception:
        pass
    return None


def build_nudge(
    tool_name: str,
    args: Dict[str, Any],
    result: Any,
    session_id: Optional[str],
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Return a one-line hint to append to *result*, or None if no trigger fires.

    Rate-limit key (review fix B3): prefer session_id; if the dispatcher
    didn't pass one (e.g. tools/code_execution_tool.py's sandboxed
    handle_function_call only forwards task_id), fall back to task_id. If
    BOTH are empty, fail closed — no nudge — rather than pool unrelated
    agents into a shared `_no_session` budget.
    """
    session_key = session_id or task_id
    if not session_key:
        return None

    if tool_name == "read_file":
        path = args.get("path", "")
        if not _is_source_path(path):
            return None

        # Repeated reads of the SAME file this session (>=3rd call) -> code_capsule,
        # independent of offset/limit or size.
        rc = _lru_touch_session(_read_counts, session_key, OrderedDict)
        count = _lru_bump_path(rc, path)
        if count >= 3:
            if _take_nudge_slot(session_key, "read_file_repeat"):
                short_path = _truncate_path_for_nudge(path)
                return (
                    f"\n\n💡 {short_path} read {count}x this session — "
                    "code_capsule(path, line) gives a one-shot summary instead of re-reading."
                )
            return None

        # Whole-file read (no offset/limit) on a big enough file -> code_symbols.
        if "offset" in args or "limit" in args:
            return None
        total_lines = _result_total_lines(result)
        if total_lines is None or total_lines < _MIN_LINES_FOR_SYMBOLS_NUDGE:
            return None
        if _take_nudge_slot(session_key, "read_file"):
            return (
                f"\n\n💡 {total_lines}-line file read whole — code_symbols(path) lists "
                "functions/classes with line numbers for far fewer tokens."
            )
        return None

    if tool_name == "search_files":
        if args.get("target") not in (None, "content"):
            return None
        pattern = args.get("pattern", "")
        search_path = args.get("path", ".")
        if not pattern:
            return None
        if not (_IDENTIFIER_RE.match(pattern) or _DECL_RE.search(pattern)):
            return None
        if not (_is_source_path(search_path) or _is_source_dir(search_path)):
            return None
        if _take_nudge_slot(session_key, "search_files"):
            return (
                "\n\n💡 Identifier-like pattern — code_workspace_symbols/code_references "
                "finds real symbols, not text matches."
            )
        return None

    if tool_name == "terminal":
        command = args.get("command", "")
        if not command:
            return None
        # Normalize case (review fix B4): `grep Foo X.TS` previously missed
        # the extension match because _SOURCE_EXTENSIONS is lowercase-only.
        command_lower = command.lower()
        if not _GREP_RE.search(command_lower):
            return None
        # `find -name/-iname/-path ...` locates paths by NAME, not a source
        # content scan — never nudge those (would teach the wrong lesson).
        # Any of these flags means a names/location lookup, never a grep-style
        # code search code_intel could have done. (No extension exception:
        # `find -name "*.ts"` is still a name lookup, not a content scan.)
        if re.search(r"\s-(?:name|iname|path)\s", command_lower):
            return None
        # Old check: only a literal source extension in the command string
        # (grep -rn X --include=*.py). Missed the dominant case `grep -rln X
        # apps/unified-api/app` where the target is a source path but no
        # extension literal appears. Accept EITHER a literal extension OR a
        # resolved source path/dir among the command's path-like tokens.
        has_literal_ext = any(ext in command_lower for ext in _SOURCE_EXTENSIONS)
        has_source_path = False
        if not has_literal_ext:
            for cand in _iter_candidate_paths(command):
                if _is_source_path(cand) or _is_source_dir(cand):
                    has_source_path = True
                    break
        if not (has_literal_ext or has_source_path):
            return None
        if _take_nudge_slot(session_key, "terminal"):
            return (
                "\n\n💡 grep/find on source code — code_search/code_workspace_symbols "
                "is AST-aware and won't match comments/strings."
            )
        return None

    return None
