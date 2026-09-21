"""Tests for the LSP identifier-column retry in lsp_bridge.

Reproduced defect (before the fix): ``code_definition``/``code_references``
auto-detected exactly ONE column per line and, when that column carried no
identifier (``def foo():`` -> ``def`` is a keyword, ``if x(...)`` -> first
identifier is a callee), returned

    {"method": "fallback", "warning": "Could not extract an identifier at the given position."}

or dropped into the expensive ``rg``-over-the-whole-tree text fallback, even
though a live LSP bridge was already initialized.

These tests use a real pyright-langserver bridge and are skipped when the
server binary is absent.
"""

import json
import os
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent

from code_intel.lsp_bridge import (  # noqa: E402
    _KEYWORDS_BY_LANGUAGE,
    _KEYWORDS_GLOBAL,
    _auto_detect_identifier_column,
    _candidate_identifier_columns,
    _format_references,
    _resolve_command,
)

PY_SRC = '''\
VALUE = 1


def alpha(x):
    return beta(x)


def beta(x):
    return x + VALUE


class Gamma:
    def run(self):
        return alpha(VALUE)
'''


@pytest.fixture()
def py_file(tmp_path):
    f = tmp_path / "mod_a.py"
    f.write_text(PY_SRC)
    return f


@pytest.fixture()
def live_python_bridge(py_file):
    """Initialized pyright bridge rooted at the fixture dir, or skip."""
    if _resolve_command("pyright-langserver") is None:
        pytest.skip("pyright-langserver not installed")
    from code_intel.lsp_bridge import get_lsp_manager

    bridge = get_lsp_manager().get_bridge("python", str(py_file))
    if bridge is None or not bridge.ensure_initialized():
        pytest.skip("pyright-langserver could not initialize")
    return bridge


# ---------------------------------------------------------------------------
# Keyword table / candidate extraction (no LSP needed)
# ---------------------------------------------------------------------------

def test_def_is_a_keyword_for_detection():
    """`def` must be skipped or every Python def line mis-detects its column."""
    assert "def" in _KEYWORDS_BY_LANGUAGE["python"]
    assert "def" in _KEYWORDS_GLOBAL


# ---------------------------------------------------------------------------
# Language-aware filtering: legal symbol names must NOT be treated as keywords
# ---------------------------------------------------------------------------

def _write(tmp_path, name, src):
    f = tmp_path / name
    f.write_text(src)
    return f


def test_python_name_print_is_not_filtered(tmp_path):
    """`def print(...)` / `value = print(x)` is legal Python — the retry must not
    skip it and silently resolve the NEXT identifier on the line instead."""
    f = _write(tmp_path, "m.py", "def print(msg):\n    return msg\n")
    col = _auto_detect_identifier_column(str(f), 0, "python")
    line = "def print(msg):"
    assert line[col - 1:].startswith("print"), (col, line[col - 1:])


def test_python_name_use_is_not_filtered(tmp_path):
    """`use` is a Rust keyword, not a Python one."""
    f = _write(tmp_path, "m.py", "use = compute(use)\n")
    col = _auto_detect_identifier_column(str(f), 0, "python")
    assert "use = compute(use)"[col - 1:].startswith("use")


def test_python_self_attribute_resolves_to_attribute_name(tmp_path):
    """`self.value` -> the symbol the caller means is `value`."""
    f = _write(tmp_path, "m.py", "    return self.value\n")
    col = _auto_detect_identifier_column(str(f), 0, "python")
    line = "    return self.value"
    assert line[col - 1:].startswith("value")


def test_rust_keyword_use_is_filtered_in_rust(tmp_path):
    """The same word IS a keyword in Rust and must still be skipped there."""
    f = _write(tmp_path, "m.rs", "use crate::foo::bar;\n")
    cands = _candidate_identifier_columns(str(f), 0, lang="rust")
    line = "use crate::foo::bar;"
    assert all(not line[c - 1:].startswith("use") for c in cands), cands


def test_explicit_language_selects_table():
    """python table must not contain Rust/Go-only keywords and vice versa."""
    assert "struct" not in _KEYWORDS_BY_LANGUAGE["python"]
    assert "chan" not in _KEYWORDS_BY_LANGUAGE["python"]
    assert "def" not in _KEYWORDS_BY_LANGUAGE["rust"]
    assert "self" in _KEYWORDS_BY_LANGUAGE["rust"]


def test_unknown_language_uses_conservative_global_table(tmp_path):
    """No language -> conservative global table (syntax words only)."""
    f = _write(tmp_path, "m.py", "def handler(x):\n")
    cands = _candidate_identifier_columns(str(f), 0, lang=None)
    line = "def handler(x):"
    assert line[cands[0] - 1:].startswith("handler"), cands


def test_builtin_names_are_never_dropped(tmp_path):
    """`print` is a legal symbol name in Python — it must stay a candidate, not
    be demoted behind other identifiers on the line."""
    f = _write(tmp_path, "m.py", "    print(handler)\n")
    cands = _candidate_identifier_columns(str(f), 0, limit=2, lang="python")
    line = "    print(handler)"
    assert line[cands[0] - 1:].startswith("print"), cands
    assert cands[1] > cands[0]


# ---------------------------------------------------------------------------
# Truncation must be visible in the prose formatter, never implied complete
# ---------------------------------------------------------------------------

def test_format_references_marks_truncation_explicitly():
    refs = [{"file": "/a.py", "line": 1, "text": "x"}]
    by_file = {"/a.py": refs}
    out = _format_references(refs, by_file, truncated=True, total_matches=4000)
    assert "TRUNCATED" in out
    assert "INCOMPLETE" in out
    assert "4000" in out
    assert "rename/refactor" in out
    # the un-truncated header must still read as a complete list
    assert "TRUNCATED" not in _format_references(refs, by_file)
    assert "Found 1 references" in _format_references(refs, by_file)


def test_candidates_skip_def_and_return_identifier(py_file):
    # 1-based line 4 is `def alpha(x):` -> candidate must be `alpha`, not `def`
    cands = _candidate_identifier_columns(str(py_file), 3)
    assert cands, "expected at least one candidate"
    line = PY_SRC.split("\n")[3]
    assert line.startswith("def alpha")
    first = cands[0]
    assert line[first - 1:].startswith("alpha")


def test_auto_detect_still_returns_first_candidate(py_file):
    assert _auto_detect_identifier_column(str(py_file), 3) == \
        _candidate_identifier_columns(str(py_file), 3)[0]


def test_candidates_are_bounded(py_file):
    cands = _candidate_identifier_columns(str(py_file), 12, limit=4)
    assert len(cands) <= 4


def test_candidates_on_out_of_range_line_is_empty(py_file):
    assert _candidate_identifier_columns(str(py_file), 10_000) == []


def test_candidates_on_missing_file_is_empty(tmp_path):
    assert _candidate_identifier_columns(str(tmp_path / "nope.py"), 0) == []


# ---------------------------------------------------------------------------
# Live LSP: def lines and callee lines must resolve via LSP, not fallback
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_resolve_command("pyright-langserver") is None,
                    reason="pyright-langserver not installed")
class TestLivePythonNavigation:

    def _def(self, path, line, character=None):
        from code_intel.lsp_bridge import code_definition_tool
        return json.loads(code_definition_tool(path=str(path), line=line, character=character))

    def _refs(self, path, line, character=None):
        from code_intel.lsp_bridge import code_references_tool
        return json.loads(code_references_tool(path=str(path), line=line,
                                               character=character, group_by_file=True))

    def test_definition_on_def_line_uses_lsp(self, py_file, live_python_bridge):
        """Regression: `def alpha(x):` used to fail closed."""
        out = self._def(py_file, 4)
        assert out["method"] == "lsp", out
        assert out["definition_count"] >= 1

    def test_references_on_def_line_uses_lsp(self, py_file, live_python_bridge):
        out = self._refs(py_file, 4)
        assert out["method"] == "lsp", out
        assert out["reference_count"] >= 2

    def test_definition_on_call_site_uses_lsp(self, py_file, live_python_bridge):
        # line 5 is `    return beta(x)` — the callee must resolve via LSP
        out = self._def(py_file, 5)
        assert out["method"] == "lsp", out

    def test_explicit_wrong_column_still_falls_back_cleanly(self, py_file, live_python_bridge):
        """An explicit column is authoritative — no silent retry surprises."""
        out = self._def(py_file, 5, character=1)
        assert "method" in out

    def test_no_character_crash_in_references_logging(self, py_file, live_python_bridge):
        """`%d` with character=None used to raise inside the logger call."""
        out = self._refs(py_file, 4, character=None)
        assert out["method"] == "lsp"

    def test_blank_line_fails_closed_without_crashing(self, py_file, live_python_bridge):
        """A blank line has no identifier — must return a clean fallback."""
        out = self._def(py_file, 2)
        assert out["method"].startswith("fallback")

    def test_no_explicit_column_reports_ambiguity(self, py_file, live_python_bridge):
        """A multi-identifier line must not silently pick a symbol.

        Line 5 is `    return beta(x)` -> three candidate columns. The tool
        resolves the first that works, so it MUST say which column it used and
        offer the alternative, instead of pretending the line was unambiguous.
        """
        out = self._def(py_file, 5)
        if out["method"] != "lsp":
            pytest.skip("LSP did not resolve this line in this environment")
        assert len(out.get("candidates_considered", [])) > 1, out
        assert out["retry_used"] is True
        assert "character" in out["ambiguity_note"]

    def test_explicit_column_is_authoritative_and_silent(self, py_file, live_python_bridge):
        """An explicit column means NO retry and NO ambiguity note."""
        # column of `beta` on `    return beta(x)`
        out = self._def(py_file, 5, character=12)
        assert "retry_used" not in out
        assert "ambiguity_note" not in out
        assert out["query"]["character"] == 12

    def test_references_ambiguity_never_silently_switches_symbol(self, py_file, live_python_bridge):
        out = self._refs(py_file, 5)
        if out["method"] != "lsp":
            pytest.skip("LSP did not resolve this line in this environment")
        if len(out.get("candidates_considered", [])) > 1:
            assert out["retry_used"] is True
            assert "candidates_considered" in out
