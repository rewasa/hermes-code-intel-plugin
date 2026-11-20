import importlib.util
import json
import sys
import types
from pathlib import Path


def _load(module_name, filename):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(module_name, root / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_registration_skips_existing_core_tool():
    ci = _load("code_intel_registration_test", "code_intel.py")

    class Registry:
        def __init__(self):
            self.calls = []

        def get_entry(self, name):
            return object() if name == "code_symbols" else None

        def register(self, **kwargs):
            self.calls.append(kwargs["name"])

    registry = Registry()
    setattr(ci, "registry", registry)
    assert ci._register_unless_present(name="code_symbols") is False
    assert ci._register_unless_present(name="plugin_only") is True
    assert registry.calls == ["plugin_only"]


def test_pnpm_workspace_globs_and_summary(tmp_path):
    package_root = tmp_path / "packages" / "shared"
    ignored = tmp_path / "packages" / "ignored"
    for path, private in ((tmp_path / "server", True), (tmp_path / "ui", True),
                          (tmp_path / "cli", True), (package_root, False), (ignored, False)):
        path.mkdir(parents=True)
        (path / "package.json").write_text(json.dumps({"name": path.name, "private": private}))
        (path / "src").mkdir()
        (path / "src" / "index.ts").write_text("export const value = 1\n")
    (tmp_path / "pnpm-workspace.yaml").write_text(
        "packages:\n  - 'server'\n  - 'ui'\n  - 'cli'\n  - 'packages/*'\n  - '!packages/ignored'\n"
    )

    lb = _load("code_intel_lsp_test", "lsp_bridge.py")
    folders = lb._find_workspace_folders(str(tmp_path))
    assert folders == sorted([str(tmp_path / "server"), str(tmp_path / "ui"),
                              str(tmp_path / "cli"), str(package_root)])

    ci = _load("code_intel_summary_test", "code_intel.py")
    package = types.ModuleType("code_intel")
    package.__path__ = [str(Path(__file__).parents[1])]
    sys.modules["code_intel"] = package
    sys.modules["code_intel.lsp_bridge"] = lb
    ci.__package__ = "code_intel"
    result = json.loads(ci.code_workspace_summary_tool(str(tmp_path)))
    assert {app["name"] for app in result["apps"]} == {"server", "ui", "cli"}
    assert {package["name"] for package in result["packages"]} == {"shared"}


def test_lsp_budget_stays_below_tool_timeout():
    lb = _load("code_intel_timeout_test", "lsp_bridge.py")
    assert lb._LSP_INIT_TIMEOUT + lb._LSP_REQUEST_TIMEOUT + 15 < 60
