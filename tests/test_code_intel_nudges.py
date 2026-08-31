"""Tests for code_intel_nudges.py — the mechanical steering hook.

Covers the terminal source-path nudge (added alongside the `_is_source_dir`
scan cap and the `find -name/-iname/-path` guard) so the source-path detection
stays covered against regression.
"""

from code_intel import code_intel_nudges as n


def _build(tool_name, args, session="test-sess", result="out"):
    return n.build_nudge(tool_name, args, result, session, "task-x")


def make_src_dir(tmp_path, nfiles=5):
    d = tmp_path / "srcapp"
    d.mkdir()
    for i in range(nfiles):
        (d / ("mod%02d.ts" % i)).write_text("export const x = %d;\n" % i)
    return d


# ---------------------------------------------------------------------------
# _is_source_dir scan cap
# ---------------------------------------------------------------------------

def test_source_dir_scan_cap_node_modules_like(tmp_path):
    # > 300 non-source entries -> must fail closed (no nudge) so grep/find on
    # huge non-source trees cannot cost thousands of stat() calls per call.
    d = tmp_path / "bigdir"
    d.mkdir()
    for i in range(400):
        (d / ("asset%03d.png" % i)).write_text("")
    assert not n._is_source_dir(str(d))


def test_source_dir_within_cap(tmp_path):
    d = tmp_path / "okdir"
    d.mkdir()
    for i in range(10):
        (d / ("f%02d.json" % i)).write_text("{}")
    (d / "main.py").write_text("x = 1\n")  # source file among the non-source
    assert n._is_source_dir(str(d))


# ---------------------------------------------------------------------------
# _iter_candidate_paths
# ---------------------------------------------------------------------------

def test_iter_candidate_paths_skips_flag_pattern():
    toks = list(n._iter_candidate_paths(
        'grep -rn "x" /src/app/main.ts pnpm-lock.yaml --include=*.py'))
    # -rn filtered, quoted pattern filtered (no '/'), paths kept
    assert "/src/app/main.ts" in toks
    assert "pnpm-lock.yaml" not in toks  # no '/', no '~', not '.'/'..'
    assert "--include=*.py" not in toks


# ---------------------------------------------------------------------------
# build_nudge / terminal
# ---------------------------------------------------------------------------

def test_terminal_grep_src_path_no_ext_literal(tmp_path):
    # The dominant previously-missed case: real source dir but no extension
    # literal anywhere in the command string.
    d = make_src_dir(tmp_path)
    got = _build("terminal", {"command": "grep -rn \"foo\" %s" % d})
    assert got is not None and "code_search" in got


def test_terminal_grep_src_file_no_ext_literal(tmp_path):
    f = make_src_dir(tmp_path) / "mod00.ts"
    got = _build("terminal", {"command": "grep -n \"foo\" %s" % f})
    assert got is not None and "code_search" in got


def test_terminal_literal_ext_still_fires():
    got = _build("terminal", {"command": "grep -rn X --include=*.py /some/dir"})
    assert got is not None


def test_terminal_find_name_lookup_blocked(tmp_path):
    # Locating a dir by -name is a names/location lookup, not a source scan.
    d = make_src_dir(tmp_path)
    assert _build("terminal", {"command": "find %s -maxdepth 1 -name node_modules" % d}) is None
    assert _build("terminal", {"command": "find %s -iname \"*foo*\" -prune -o -print" % d}) is None
    assert _build("terminal", {"command": "find %s -path ./node_modules -prune -o -iname \"*x*\"" % d}) is None


def test_terminal_non_source_target_silent(tmp_path):
    f = tmp_path / "pnpm-lock.yaml"
    f.write_text("lockfileVersion: 9\n")
    assert _build("terminal", {"command": "grep -rn \"x\" %s" % f}) is None


def test_terminal_non_grep_command_silent(tmp_path):
    # awk/sed/cat never trigger the terminal nudge (no code_intel equivalent).
    assert _build("terminal", {"command": "awk '/e/ {print}' /var/log/x.log"}) is None
    assert _build("terminal", {"command": "cat %s" % (make_src_dir(tmp_path) / "mod00.ts")}) is None


def test_terminal_grep_without_grep_word_silent(tmp_path):
    d = make_src_dir(tmp_path)
    assert _build("terminal", {"command": "ls %s" % d}) is None


def test_terminal_nudge_rate_limited(tmp_path):
    d = make_src_dir(tmp_path)
    fired = 0
    for _ in range(5):
        if _build("terminal", {"command": "grep -r foo %s" % d}) is not None:
            fired += 1
    assert fired <= n._MAX_NUDGE_PER_TOOL


# ---------------------------------------------------------------------------
# read_file / search_files regression (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_read_file_whole_ge_min_lines(tmp_path):
    f = tmp_path / "big.ts"
    lines = ["// %d" % i for i in range(n._MIN_LINES_FOR_SYMBOLS_NUDGE + 5)]
    f.write_text("\n".join(lines) + "\n")
    got = _build("read_file", {"path": str(f)}, result={"total_lines": len(lines)})
    assert got is not None and "code_symbols" in got


def test_search_files_identifier_pattern(tmp_path):
    d = make_src_dir(tmp_path)
    got = _build("search_files", {"pattern": "fooBar", "path": str(d), "target": None})
    assert got is not None
