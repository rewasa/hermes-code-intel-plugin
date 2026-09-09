"""Focused regression tests for code_diagnostics LSP behavior.

Covers the TSLS push-capability root cause and fallback-reason reporting:
- real LSP servers (typescript-language-server, pyright-langserver) must be
  present for the live tests; they are skipped with a clear message otherwise.
- clean files return method=lsp with zero diagnostics ([] vs None contract)
- error files return method=lsp with the intentional errors
- fallback_reason distinguishes unavailable / init-failed / no-support / no-response
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

from ci_diag_fixtures import ensure_fixtures  # noqa: E402

FIXTURE_DIR = PLUGIN_DIR / "tests" / ".fixtures"
VENV_PYTHON = Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python3"


def _venv_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _has_server(cmd: str) -> bool:
    from lsp_bridge import _resolve_command
    return _resolve_command(cmd) is not None


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return ensure_fixtures(str(FIXTURE_DIR))


def _diag(path: str) -> dict:
    from lsp_bridge import code_diagnostics_tool
    return json.loads(code_diagnostics_tool(path))


# ── live LSP tests (skipped when the server binary is missing) ──────────

@pytest.mark.skipif(not _has_server("typescript-language-server"),
                    reason="typescript-language-server not installed")
class TestTSDiagnostics:
    def test_clean_file_zero_diagnostics_push(self, fixtures):
        d = _diag(fixtures["clean.ts"])
        assert d["method"] == "lsp"
        assert d["lsp_server"] == "typescript-language-server"
        assert d["diagnostic_count"] == 0
        assert d["diagnostics"] == []

    def test_error_file_reports_intentional_errors(self, fixtures):
        d = _diag(fixtures["error.ts"])
        assert d["method"] == "lsp"
        assert d["diagnostic_count"] >= 1
        codes = {item.get("code") for item in d["diagnostics"]}
        assert 2322 in codes, f"expected TS2322 in {codes}"
        assert 2304 in codes, f"expected TS2304 in {codes}"

    def test_warm_reuse_still_pushes(self, fixtures):
        # second call exercises the didChange (already-open) path
        d = _diag(fixtures["error.ts"])
        assert d["method"] == "lsp"
        assert d["diagnostic_count"] >= 1


@pytest.mark.skipif(not _has_server("pyright-langserver"),
                    reason="pyright-langserver not installed")
class TestPythonDiagnostics:
    def test_clean_file_zero_diagnostics(self, fixtures):
        d = _diag(fixtures["clean.py"])
        assert d["method"] in ("lsp", "lsp-pull")
        assert d["diagnostic_count"] == 0

    def test_error_file_reports_intentional_error(self, fixtures):
        d = _diag(fixtures["error.py"])
        assert d["method"] in ("lsp", "lsp-pull")
        assert d["diagnostic_count"] >= 1
        assert any("not_defined_name" in item.get("message", "")
                   for item in d["diagnostics"])


# ── fallback-reason tests (no live server required for most) ────────────

class TestFallbackReasons:
    def test_unsupported_language(self, tmp_path):
        from lsp_bridge import code_diagnostics_tool
        md = tmp_path / "file.md"
        md.write_text("# plain markdown\n")
        d = json.loads(code_diagnostics_tool(str(md)))
        assert d["method"] == "ast_heuristic"
        assert d["fallback_reason"] == "unsupported_language"

    def test_reduced_path_still_finds_server(self, fixtures):
        """Cold subprocess with PATH=/usr/bin:/bin must still run real LSP."""
        code = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(PLUGIN_DIR)!r})\n"
            "from lsp_bridge import code_diagnostics_tool\n"
            "d = json.loads(code_diagnostics_tool(%r))\n"
            "print(json.dumps({'method': d.get('method'),"
            " 'count': d.get('diagnostic_count'),"
            " 'server': d.get('lsp_server')}))\n"
        ) % fixtures["error.ts"]
        env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
        proc = subprocess.run(
            [VENV_PYTHON if VENV_PYTHON.exists() else sys.executable, "-c", code],
            capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 0, proc.stderr[-300:]
        result_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
        d = json.loads(result_line)
        assert d["method"] == "lsp", proc.stdout[-300:]
        assert d["server"] == "typescript-language-server"
        assert d["count"] >= 1

    def test_silent_pull_capable_server_reason(self, fixtures):
        """Initialized server, push lost AND pull returns nothing →
        no_diagnostics_response (not the misleading 'server unavailable')."""
        import lsp_bridge
        from lsp_bridge import code_diagnostics_tool, get_lsp_manager

        mgr = get_lsp_manager()
        bridge = mgr.get_bridge("python", fixtures["error.py"])
        assert bridge.ensure_initialized()
        # pyright answers pull but doesn't advertise it — heuristic knows this
        orig_collect = lsp_bridge.LSPBridge.collect_diagnostics
        orig_send = lsp_bridge.LSPBridge._send_request
        lsp_bridge.LSPBridge.collect_diagnostics = lambda self, *a, **k: None
        lsp_bridge.LSPBridge._send_request = lambda self, m, p, timeout=10: None
        try:
            d = json.loads(code_diagnostics_tool(fixtures["error.py"]))
        finally:
            lsp_bridge.LSPBridge.collect_diagnostics = orig_collect
            lsp_bridge.LSPBridge._send_request = orig_send
        assert d["method"] == "ast_heuristic"
        assert d["fallback_reason"] == "no_diagnostics_response"

    def test_missing_binary_reason(self, fixtures, monkeypatch):
        """No resolvable server binary → lsp_server_unavailable.

        Uses a fresh LSPManager: the singleton would return an already-running
        bridge before the binary lookup ever runs.
        """
        import lsp_bridge
        from lsp_bridge import code_diagnostics_tool

        monkeypatch.setattr(lsp_bridge, "_resolve_command", lambda cmd: None)
        monkeypatch.setattr(lsp_bridge, "_lsp_manager", lsp_bridge.LSPManager())
        d = json.loads(code_diagnostics_tool(fixtures["error.ts"]))
        assert d["method"] == "ast_heuristic"
        assert d["fallback_reason"] == "lsp_server_unavailable"

    def test_empty_vs_none_contract(self, fixtures):
        """[] (authoritative clean) must not degrade to AST fallback."""
        d = _diag(fixtures["clean.ts"])
        assert d["method"] == "lsp"
        assert d["diagnostic_count"] == 0
        assert "fallback_reason" not in d or d["fallback_reason"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))