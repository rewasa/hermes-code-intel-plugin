#!/usr/bin/env python3
"""
code_intel health check — tests tools, LSP bridge, and registry.

Produces a concise health report. Silently exits 0 when healthy.
Only outputs to stdout when issues are found.

Exit 0: all healthy
Exit 1: critical failures
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── Config ──────────────────────────────────────────────
# Prefer the PLUGIN's own artifacts (this script ships inside the repo), then
# fall back to the Hermes built-in copies. Paths resolve relative to this
# file so a broken/partial plugin checkout is detected, not masked by a
# healthy built-in.
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
HERMES_AGENT = Path(os.path.expanduser("~/.hermes/hermes-agent"))
_BUILTIN_DIR = HERMES_AGENT / "tools"

def _artifact_path(name: str) -> tuple[Path, str]:
    """Return (path, source) preferring the plugin copy over the built-in."""
    plugin_file = PLUGIN_ROOT / name
    if plugin_file.exists():
        return plugin_file, "plugin"
    return _BUILTIN_DIR / name, "built-in"

CODE_INTEL_PY, _CODE_INTEL_SRC = _artifact_path("code_intel.py")
LSP_BRIDGE_PY, _LSP_SRC = _artifact_path("lsp_bridge.py")
# Path to the plugin's lsp_bridge.py used by the isolated LSP subprocess tests.
BRIDGE_PY_PATH = LSP_BRIDGE_PY if LSP_BRIDGE_PY.exists() else (_BUILTIN_DIR / "lsp_bridge.py")

MONOREPO = Path(os.path.expanduser("~/GIT/AgentSelly/monorepo"))
TS_TARGET = None
if MONOREPO.exists():
    for candidate in sorted(MONOREPO.glob("apps/*/app/**/*.ts"), key=lambda p: p.stat().st_size):
        if candidate.stat().st_size < 500_000 and "node_modules" not in str(candidate) \
           and "test" not in candidate.name.lower() and "spec" not in candidate.name.lower():
            TS_TARGET = candidate
            break
    if not TS_TARGET:
        for candidate in sorted(MONOREPO.glob("packages/*/src/**/*.ts"), key=lambda p: p.stat().st_size):
            if candidate.stat().st_size < 500_000:
                TS_TARGET = candidate
                break

LOG_FILE = Path(os.path.expanduser("~/.hermes/logs/errors.log"))
CUTOFF = datetime.now() - timedelta(hours=6)
LSP_TIMEOUT = 15
VENV_PYTHON = HERMES_AGENT / "venv" / "bin" / "python3"

# ── Results ──────────────────────────────────────────────
_issues = []
_ok = []

def issue(severity: str, component: str, detail: str):
    _issues.append({"severity": severity, "component": component, "detail": detail})

def ok(component: str, detail: str = ""):
    _ok.append({"component": component, "detail": detail})


def timed(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


# ── Checks ──────────────────────────────────────────────

def check_file_integrity():
    for path, label, src in [
        (CODE_INTEL_PY, "code_intel.py", _CODE_INTEL_SRC),
        (LSP_BRIDGE_PY, "lsp_bridge.py", _LSP_SRC),
    ]:
        if not path.exists():
            issue("critical", "files", f"{label} missing at {path}")
        else:
            size = path.stat().st_size
            if size < 1000:
                issue("critical", "files", f"{label} suspiciously small ({size} bytes)")
            else:
                ok("files", f"{label} OK ({size:,} bytes, source={src})")


def check_fast_tools():
    """Run tree-sitter tools (no LSP). Always completes in <1s."""
    os.chdir(str(HERMES_AGENT))
    sys.path.insert(0, str(HERMES_AGENT))

    from tools.code_intel import code_symbols_tool, code_search_tool, code_refactor_tool

    # code_symbols on Python
    r, ms = timed("code_symbols(py)",
        lambda: json.loads(code_symbols_tool(str(CODE_INTEL_PY), kind="function")))
    if isinstance(r, dict) and "error" not in r and r.get("symbol_count", 0) > 0:
        ok("code_symbols", f"{r['symbol_count']} symbols in {ms:.0f}ms (Python)")
    else:
        issue("critical", "code_symbols", f"FAILED: {r.get('error', str(r)[:120])} ({ms:.0f}ms)")

    # code_symbols on TypeScript
    if TS_TARGET and TS_TARGET.exists():
        r, ms = timed("code_symbols(ts)",
            lambda: json.loads(code_symbols_tool(str(TS_TARGET), kind="class")))
        if isinstance(r, dict) and "error" not in r:
            ok("code_symbols", f"{r.get('symbol_count', 0)} classes in {ms:.0f}ms (TypeScript)")
        else:
            issue("warning", "code_symbols", f"TS scan issue: {r.get('error', str(r)[:120])} ({ms:.0f}ms)")

    # code_search
    r, ms = timed("code_search(py)",
        lambda: json.loads(code_search_tool(str(CODE_INTEL_PY), preset="function_calls", pattern="json")))
    if isinstance(r, dict) and "error" not in r:
        ok("code_search", f"AST search OK in {ms:.0f}ms (Python)")
    else:
        issue("critical", "code_search", f"FAILED: {r.get('error', str(r)[:120])} ({ms:.0f}ms)")

    # code_search on TS
    if TS_TARGET and TS_TARGET.exists():
        r, ms = timed("code_search(ts)",
            lambda: json.loads(code_search_tool(str(TS_TARGET), preset="imports")))
        if isinstance(r, dict) and "error" not in r:
            ok("code_search", f"TS import search OK in {ms:.0f}ms")
        else:
            issue("warning", "code_search", f"TS search issue: {r.get('error', str(r)[:120])} ({ms:.0f}ms)")

    # code_refactor dry-run
    r, ms = timed("code_refactor(dry)",
        lambda: json.loads(code_refactor_tool(str(CODE_INTEL_PY), pattern="json.dumps", rewrite="json.dumps")))
    if isinstance(r, dict) and "error" not in r:
        ok("code_refactor", f"dry-run OK in {ms:.0f}ms")
    else:
        issue("critical", "code_refactor", f"FAILED: {r.get('error', str(r)[:120])} ({ms:.0f}ms)")


def _find_def_line(file_path: Path, needle: str) -> Optional[int]:
    """Return the 1-based line number of the first top-level ``needle`` (e.g.
    ``def code_symbols_tool``) in *file_path*, or None if not found."""
    try:
        for i, line in enumerate(file_path.read_text(errors="replace").splitlines(), start=1):
            if line.lstrip().startswith(needle):
                return i
    except OSError:
        pass
    return None


def _lsp_standalone_test(target_file: str, target_line: int) -> dict:
    """Run LSP goto-definition in a clean isolated subprocess.

    Uses venv python with a self-contained script to avoid import
    side effects from this process (stale module cache, open FDs, etc.).
    Loads the PLUGIN's lsp_bridge.py directly by file path so the check
    verifies the shipped artifact, not a same-named Hermes built-in.
    """
    # Pre-kill stale pylsp processes
    subprocess.run(["pkill", "-f", "[p]ylsp"], capture_output=True, timeout=2)

    # Load the plugin's lsp_bridge.py (fallback to the built-in copy if the
    # plugin checkout is broken).
    bridge_path = BRIDGE_PY_PATH

    script = f'''
import sys, os, json, time, importlib.util
HERMES = '{HERMES_AGENT}'
os.chdir(HERMES)
sys.path.insert(0, HERMES)

target = '{target_file}'
line = {target_line}

t0 = time.time()
spec = importlib.util.spec_from_file_location("lsp_bridge_under_test", '{bridge_path}')
mod = importlib.util.module_from_spec(spec)
sys.modules["lsp_bridge_under_test"] = mod  # dataclass needs sys.modules during exec
spec.loader.exec_module(mod)
LSPBridge = mod.LSPBridge
_find_workspace_root = mod._find_workspace_root

root = _find_workspace_root(target)
bridge = LSPBridge(command='pylsp', args=[], root_uri=root, language_id='python')

result = {{}}
if bridge.ensure_initialized():
    locs = bridge.goto_definition(target, line - 1, 5)  # 0-based, col ~5
    elapsed = (time.time() - t0) * 1000
    result = {{
        "ok": True,
        "definition_count": len(locs or []),
        "elapsed_ms": int(elapsed),
    }}
else:
    result = {{"ok": False, "error": "LSP init failed"}}

bridge.shutdown()
print(json.dumps(result))
'''

    try:
        proc = subprocess.run(
            [str(VENV_PYTHON), "-c", script],
            capture_output=True, text=True,
            timeout=LSP_TIMEOUT,
            cwd=str(HERMES_AGENT),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
        return {"ok": False, "error": f"No output (rc={proc.returncode})", "stderr": proc.stderr[:200]}
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "[p]ylsp"], capture_output=True, timeout=2)
        return {"ok": False, "error": "TimeoutExpired", "elapsed_ms": LSP_TIMEOUT * 1000}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def check_lsp():
    """Test LSP bridge via isolated standalone subprocess."""
    # code_definition on a real, stable symbol (code_symbols_tool def).
    # Line numbers in code_intel.py shift as the plugin evolves, so resolve the
    # target by symbol name instead of a hard-coded line.
    _lsp_target_line = _find_def_line(CODE_INTEL_PY, "def code_symbols_tool")
    if _lsp_target_line is None:
        issue("warning", "code_definition", "Cannot locate code_symbols_tool for LSP test")
        return
    r = _lsp_standalone_test(str(CODE_INTEL_PY), _lsp_target_line)
    elapsed = r.get("elapsed_ms", 0)
    if r.get("ok") and r.get("definition_count", 0) > 0:
        ok("code_definition", f"LSP goto-def OK ({r['definition_count']} defs) in {elapsed}ms")
    elif r.get("ok"):
        issue("warning", "code_definition", f"LSP returned 0 definitions ({elapsed}ms)")
    elif "TimeoutExpired" in r.get("error", ""):
        issue("info", "code_definition", f"Timed out after {LSP_TIMEOUT}s")
    else:
        issue("warning", "code_definition", f"LSP failure: {r.get('error', '?')} ({elapsed}ms)")

    # code_references
    refs_script = f'''
import sys, os, json, time, importlib.util
HERMES = '{HERMES_AGENT}'
os.chdir(HERMES)
sys.path.insert(0, HERMES)

target = '{CODE_INTEL_PY}'
line = {_lsp_target_line}

t0 = time.time()
spec = importlib.util.spec_from_file_location("lsp_bridge_under_test_refs", '{BRIDGE_PY_PATH}')
mod = importlib.util.module_from_spec(spec)
sys.modules["lsp_bridge_under_test_refs"] = mod
spec.loader.exec_module(mod)
LSPBridge = mod.LSPBridge
_find_workspace_root = mod._find_workspace_root

root = _find_workspace_root(target)
bridge = LSPBridge(command='pylsp', args=[], root_uri=root, language_id='python')

result = {{}}
if bridge.ensure_initialized():
    locs = bridge.find_references(target, line - 1, 5, True)
    elapsed = (time.time() - t0) * 1000
    result = {{
        "ok": True,
        "reference_count": len(locs or []),
        "elapsed_ms": int(elapsed),
    }}
else:
    result = {{"ok": False, "error": "LSP init failed"}}

bridge.shutdown()
print(json.dumps(result))
'''

    try:
        proc = subprocess.run(
            [str(VENV_PYTHON), "-c", refs_script],
            capture_output=True, text=True,
            timeout=LSP_TIMEOUT + 10,  # references can take longer
            cwd=str(HERMES_AGENT),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            r = json.loads(proc.stdout.strip())
            elapsed = r.get("elapsed_ms", 0)
            ref_count = r.get("reference_count", 0)
            if r.get("ok") and ref_count > 0:
                ok("code_references", f"LSP refs OK ({ref_count} refs) in {elapsed}ms")
            elif r.get("ok"):
                issue("warning", "code_references", f"LSP returned 0 refs ({elapsed}ms)")
            else:
                issue("warning", "code_references", f"LSP refs failure: {r.get('error', '?')}")
        else:
            issue("warning", "code_references", f"Subprocess failed (rc={proc.returncode})")
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "[p]ylsp"], capture_output=True, timeout=2)
        issue("warning", "code_references", f"Timed out after {LSP_TIMEOUT + 10}s")
    except Exception as e:
        issue("warning", "code_references", str(e)[:150])


ALL_TOOLS = {
    # AST (tree-sitter + ast-grep)
    "code_symbols", "code_search", "code_refactor",
    # LSP
    "code_definition", "code_references", "code_diagnostics", "code_hover",
    "code_rename", "code_type_definition", "code_signatures", "code_action",
    "code_workspace_symbols",
    # Composite / convenience
    "code_callers", "code_callees", "code_capsule", "code_impact",
    "code_query", "code_tests_for_symbol", "code_workspace_summary",
}

def check_registry():
    """Verify all code_intel tools are registered AND resolve to the plugin's
    handlers (not a same-named Hermes built-in)."""
    os.chdir(str(HERMES_AGENT))
    sys.path.insert(0, str(HERMES_AGENT))

    try:
        from model_tools import get_tool_definitions
        tools = get_tool_definitions(enabled_toolsets=["code_intel"])
        by_name = {t["function"]["name"]: t for t in tools}
    except Exception as e:
        issue("warning", "registry", f"Cannot query registry: {e}")
        return

    present = ALL_TOOLS & set(by_name)
    missing = ALL_TOOLS - set(by_name)

    # Provenance: ensure the registered handler is the PLUGIN implementation,
    # not a same-named global/built-in. The plugin's handlers live in
    # modules under the code_intel package / hermes_plugins.code_intel.
    non_plugin = []
    for name in sorted(present):
        handler = by_name[name].get("function", {}).get("handler") \
            or by_name[name].get("function", {}).get("callable") \
            or by_name[name].get("handler")
        module = getattr(handler, "__module__", "") or ""
        if module and ("code_intel" not in module and "hermes_plugins" not in module):
            non_plugin.append((name, module))

    ok("registry", f"{len(present)}/{len(ALL_TOOLS)} tools active")
    if missing:
        issue("critical", "registry", f"MISSING TOOLS: {', '.join(sorted(missing))}")
    if non_plugin:
        issue("warning", "registry",
              f"{len(non_plugin)} tool(s) may resolve to a non-plugin handler: "
              + ", ".join(f"{n}@{m}" for n, m in non_plugin))


ERROR_PATTERNS = [
    (re.compile(r"LSPBridge.*has no attribute (\w+)", re.I), "attribute_error", "LSPBridge missing method"),
    (re.compile(r"code_\w+ dispatch error: (.+)", re.I), "tool_dispatch", "Tool dispatch crash"),
    (re.compile(r"Failed to persist symbol cache: (.+)", re.I), "cache_persist", "Symbol cache persist failure"),
    (re.compile(r"\[ERROR\].*lsp_bridge: (.+)", re.I), "lsp_error", "LSP bridge error"),
    (re.compile(r"\[WARNING\].*code_intel: (.+)", re.I), "code_intel_warn", "code_intel warning"),
    (re.compile(r"textDocument/diagnostic not supported", re.I), "pull_diag_unsupported", "Pull diagnostics unsupported"),
    (re.compile(r"No LSP bridge available for language=(\w+)", re.I), "no_lsp_lang", "No LSP for language"),
    (re.compile(r"tree.sitter.*Impossible pattern", re.I), "ts_query", "Impossible tree-sitter query"),
    (re.compile(r"SgNode.*Error|ast.grep.*error", re.I), "ast_grep", "ast-grep error"),
    (re.compile(r"TypeError.*tree_sitter", re.I), "ts_type_error", "tree-sitter type mismatch"),
    (re.compile(r"code_\w+.*timeout|timed out", re.I), "timeout", "Tool timeout"),
]


def scan_logs():
    if not LOG_FILE.exists():
        return
    try:
        mtime = datetime.fromtimestamp(LOG_FILE.stat().st_mtime)
        if mtime < CUTOFF:
            return
    except OSError:
        return
    try:
        lines = LOG_FILE.read_text(errors="replace").splitlines()
    except OSError:
        return

    findings = []
    for line in lines[-3000:]:
        for pattern, tag, desc in ERROR_PATTERNS:
            m = pattern.search(line)
            if m:
                detail = m.group(1) if m.lastindex else ""
                findings.append({"tag": tag, "desc": desc, "detail": detail[:100]})

    seen = set()
    unique = []
    for f in findings:
        key = (f["tag"], f["detail"][:80])
        if key not in seen:
            seen.add(key)
            unique.append(f)

    if unique:
        non_expected = [f for f in unique if f["tag"] != "pull_diag_unsupported"]
        if non_expected:
            issue("warning", "log_scan", f"{len(non_expected)} recent errors")
            for f in non_expected:
                detail = f" ({f['detail']})" if f["detail"] else ""
                issue("warning", f"log:{f['tag']}", f"{f['desc']}{detail}")


# ── Main ────────────────────────────────────────────────
def main():
    t0 = time.perf_counter()

    check_file_integrity()
    check_fast_tools()
    check_lsp()
    check_registry()
    scan_logs()

    total_ms = (time.perf_counter() - t0) * 1000
    n_critical = sum(1 for i in _issues if i["severity"] == "critical")
    n_warning = sum(1 for i in _issues if i["severity"] == "warning")
    n_info = sum(1 for i in _issues if i["severity"] == "info")

    # Silent when fully healthy
    if n_critical == 0 and n_warning == 0:
        print(f"✅ HEALTHY — {len(_ok)} checks passed ({total_ms:.0f}ms)")
        return 0

    header = f"🔬 code_intel health check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    if n_critical > 0:
        header += f"🔴 DEGRADED — {n_critical} critical, {n_warning} warnings"
    else:
        header += f"🟡 ATTENTION — {n_warning} warnings"

    print(header)
    if _issues:
        print()
        for i in _issues:
            icon = {"critical": "🔴", "warning": "🟡", "info": "ℹ️ "}.get(i["severity"], "  ")
            print(f"  {icon} [{i['component']}] {i['detail']}")

    if _ok:
        print(f"\n  ✅ {len(_ok)} checks passed")

    print(f"\n  Total: {total_ms:.0f}ms | passed={len(_ok)} "
          f"issues={len(_issues)} (critical={n_critical}, warning={n_warning}, info={n_info})")

    return 1 if n_critical > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
