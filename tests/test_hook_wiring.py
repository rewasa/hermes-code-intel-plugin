"""End-to-end wiring test: does register() actually install the coding default?

The unit tests cover ``code_intel_defaults`` in isolation; this one proves the
hook is registered by ``register(ctx)`` and that it responds to the *production*
pre_llm_call payload shape, which is the part that was silently dead.

Hermes runtime modules are stubbed so the plugin can be imported outside a
running agent. The stubs are installed by ``_install_runtime_stubs`` and are
shared by every test that calls ``register()`` — including the old-core
degradation test, which previously ran without the ``toolsets`` stub and died
with ``ModuleNotFoundError: No module named 'toolsets'`` for the wrong reason.

Section validation is asserted against ``tests/hermes_core_contract.py`` (the
checked record of the core's rules), and ``test_section_contract_matches_real_core``
compares that record against the LIVE core whenever ``hermes_cli`` is importable,
so the record cannot drift from the real validation rules.
"""

import importlib.util as _ilu
import json
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR.parent))

from hermes_core_contract import (  # noqa: E402
    CORE_CONTRACT,
    MAX_SYSTEM_PROMPT_SECTION_CHARS,
    PLUGIN_SECTIONS_END,
    PLUGIN_SECTIONS_START,
    SYSTEM_PROMPT_SECTION_POSITIONS,
    contract_drift,
    is_valid_system_prompt_section_id,
    validate_section,
)

pytest.importorskip("tree_sitter", reason="tree-sitter not installed")


class FakeRegistry:
    def __init__(self):
        self.registered = {}

    def register(self, **kwargs):
        self.registered[kwargs.get("name")] = kwargs

    def get_entry(self, name):
        return None


class FakeCtx:
    def __init__(self):
        self.hooks = {}
        self.commands = {}
        self.tools = []
        self.system_prompt_sections = {}

    def register_hook(self, name, handler):
        self.hooks.setdefault(name, []).append(handler)

    def register_command(self, name, handler=None, description=""):
        self.commands[name] = handler

    def register_skill(self, **kwargs):
        pass

    def register_tool(self, **kwargs):
        self.tools.append(kwargs.get("name"))

    def register_system_prompt_section(self, id, content, *, position="after_memory",
                                       max_chars=4000):
        assert position in {"after_memory"}
        assert 0 < max_chars <= 4000
        assert id not in self.system_prompt_sections
        self.system_prompt_sections[id] = {"content": content, "position": position,
                                           "max_chars": max_chars}


def _optional_import(module_name: str):
    """Import an optional Hermes-core module; return ``None`` when unavailable.

    Deliberately narrow: only ``ModuleNotFoundError`` counts as "core absent"
    (the standalone CI case). Any other failure is a real defect and propagates.

    ``sys.path`` is restored before returning: leaving the Hermes install dir on
    the path would make unrelated ``pytest.importorskip("tools.registry")``
    guards in this suite resolve the REAL Hermes runtime instead of skipping,
    turning those tests into order-dependent failures.
    """
    hermes_home = str(Path.home() / ".hermes" / "hermes-agent")
    added = hermes_home not in sys.path and Path(hermes_home).is_dir()
    if added:
        sys.path.insert(0, hermes_home)
    try:
        return __import__(module_name, fromlist=["*"])
    except ModuleNotFoundError:
        return None
    finally:
        if added:
            try:
                sys.path.remove(hermes_home)
            except ValueError:
                pass


def _install_runtime_stubs(monkeypatch):
    """Install the Hermes runtime modules the plugin touches during register().

    ``register()`` unconditionally does ``import toolsets`` and reads
    ``toolsets.TOOLSETS`` / ``toolsets._HERMES_CORE_TOOLS`` (real Hermes cores
    always ship those), and probes ``hermes_cli.plugins.VALID_HOOKS``. A test
    that calls ``register()`` without this stub cannot reach the code under test
    at all. Returns the fake toolset module and fake registry.
    """
    fake_registry = FakeRegistry()

    toolsets = types.ModuleType("toolsets")
    toolsets.TOOLSETS = {}
    toolsets._HERMES_CORE_TOOLS = []
    monkeypatch.setitem(sys.modules, "toolsets", toolsets)

    tools_mod = types.ModuleType("tools")
    registry_mod = types.ModuleType("tools.registry")
    registry_mod.registry = fake_registry
    tools_mod.registry = registry_mod
    monkeypatch.setitem(sys.modules, "tools", tools_mod)
    monkeypatch.setitem(sys.modules, "tools.registry", registry_mod)

    plugins_mod = types.ModuleType("hermes_cli.plugins")
    setattr(plugins_mod, "VALID_HOOKS", set(CORE_CONTRACT["REQUIRED_HOOKS"]))
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", plugins_mod)

    return toolsets, fake_registry


def _import_plugin_entrypoint(monkeypatch, name="code_intel"):
    """Exec the real plugin ``__init__.py`` as a package named *name*.

    The package alias is registered through ``monkeypatch`` so pytest restores
    ``sys.modules[name]`` to THIS repo's package on teardown — the repo root also
    ships a flat ``code_intel.py``, and a leaked alias would make every later
    ``from code_intel.<submodule> import ...`` fail with
    ``'code_intel' is not a package``.
    """
    for stale in [k for k in list(sys.modules) if k == name or k.startswith(name + ".")]:
        del sys.modules[stale]

    spec = _ilu.spec_from_file_location(
        name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    mod = _ilu.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def plugin(monkeypatch):
    """Import the plugin package with stubbed Hermes runtime modules."""
    _toolsets, fake_registry = _install_runtime_stubs(monkeypatch)
    mod = _import_plugin_entrypoint(monkeypatch)
    ctx = FakeCtx()
    mod.register(ctx)
    return mod, ctx, fake_registry


def _prod_payload(**over):
    payload = {
        "session_id": "wiring-sess",
        "task_id": "wiring-task",
        "turn_id": "t1",
        "user_message": "refactor the parser module",
        "conversation_history": [],
        "is_first_turn": True,
        "model": "m",
        "platform": "cli",
    }
    payload.update(over)
    return payload


def test_register_installs_pre_llm_call_hook(plugin):
    _mod, ctx, _reg = plugin
    assert "pre_llm_call" in ctx.hooks
    assert "transform_tool_result" in ctx.hooks
    assert "on_session_end" in ctx.hooks


def test_production_payload_yields_guidance(plugin):
    mod, ctx, _reg = plugin
    handler = ctx.hooks["pre_llm_call"][0]
    from code_intel.code_intel_defaults import forget_session

    forget_session("wiring-sess")
    out = handler(**_prod_payload())
    assert isinstance(out, str) and "code_symbols" in out
    # one-shot per session
    assert handler(**_prod_payload()) is None


def test_non_coding_payload_yields_nothing(plugin):
    mod, ctx, _reg = plugin
    handler = ctx.hooks["pre_llm_call"][0]
    from code_intel.code_intel_defaults import forget_session

    forget_session("wiring-sess")
    assert handler(**_prod_payload(user_message="what's the weather")) is None


def test_hook_injects_symbol_context_for_named_file(plugin, tmp_path):
    mod, ctx, _reg = plugin
    handler = ctx.hooks["pre_llm_call"][0]
    target = tmp_path / "svc.ts"
    target.write_text("export function compute(v: number): number { return v; }\n")
    from code_intel.code_intel_defaults import forget_session

    forget_session("ctx-sess")
    out = handler(**_prod_payload(session_id="ctx-sess",
                                  user_message="check %s for the bug" % target))
    assert out is not None
    assert "auto-context" in out and "compute" in out


def test_hook_never_raises_on_hostile_kwargs(plugin):
    mod, ctx, _reg = plugin
    handler = ctx.hooks["pre_llm_call"][0]
    assert handler() is None
    assert handler(user_message={"not": "a str"}, conversation_history="junk") is None
    assert handler(user_message="code", session_id=12345) is None


def test_context_budget_is_not_burned_without_file_refs(plugin):
    """Coding turns that name no file must not consume the context budget.

    Charging a slot before the file-reference scan left the *actual* symbol scans
    unbudgeted: three 'fix the bug' turns used up all three slots, then every
    later turn naming a real file got no auto-context.
    """
    mod, ctx, _reg = plugin
    handler = ctx.hooks["pre_llm_call"][0]
    from code_intel.code_intel_defaults import forget_session, state_sizes

    forget_session("budget-order")
    # Three coding turns with NO file reference — none may take a context slot.
    # (The first still returns the one-shot guidance; that is not budgeted.)
    first = handler(**_prod_payload(session_id="budget-order",
                                    user_message="fix the bug please"))
    assert first is not None and "code_symbols" in first
    for _ in range(2):
        handler(**_prod_payload(session_id="budget-order",
                               user_message="fix the bug please"))
    assert state_sizes()["context_sessions"] == 0

    # A turn that DOES name a file still gets its context.
    target = PLUGIN_DIR / "code_intel_defaults.py"
    out = handler(**_prod_payload(session_id="budget-order",
                                  user_message="check %s for the bug" % target))
    assert out is not None and "auto-context" in out
    assert state_sizes()["context_sessions"] == 1


def test_registered_tools_include_the_semantic_set(plugin):
    _mod, _ctx, reg = plugin
    from code_intel.code_intel import CODE_SYMBOLS_SCHEMA

    assert CODE_SYMBOLS_SCHEMA["name"] == "code_symbols"
    # registry stubbed, but the toolset injection must have happened
    assert "code_symbols" in sys.modules["toolsets"]._HERMES_CORE_TOOLS


def test_system_prompt_section_registered_and_valid(plugin):
    """The session-frozen section is the reliable default; it must satisfy the
    real core's validation rules (id charset, position, max_chars, no reserved
    persistence markers, non-empty after strip)."""
    _mod, ctx, _reg = plugin
    assert "code-intel.defaults" in ctx.system_prompt_sections
    section = ctx.system_prompt_sections["code-intel.defaults"]

    # Full core acceptance rules, mirrored from hermes_cli.plugins_dispatch and
    # drift-checked against the live core in test_section_contract_matches_real_core.
    content = validate_section("code-intel.defaults", section)

    assert is_valid_system_prompt_section_id("code-intel.defaults")
    assert section["position"] in SYSTEM_PROMPT_SECTION_POSITIONS
    assert 0 < len(content) <= section["max_chars"] <= MAX_SYSTEM_PROMPT_SECTION_CHARS
    assert PLUGIN_SECTIONS_START not in content and PLUGIN_SECTIONS_END not in content
    assert content.strip()


def test_section_contract_matches_real_core():
    """The recorded core contract must not drift from the live Hermes core.

    CI has no Hermes install, so the standalone section assertions use the
    record in ``hermes_core_contract``. This test is the guard that keeps that
    record honest: it runs wherever ``hermes_cli`` IS importable (the Hermes
    venv / a developer machine) and fails on any constant or id-charset drift.
    """
    core_dispatch = _optional_import("hermes_cli.plugins_dispatch")
    core_plugins = _optional_import("hermes_cli.plugins")
    if core_dispatch is None or core_plugins is None:
        pytest.skip("Hermes core (hermes_cli) not importable in this environment")

    drift = contract_drift(core_dispatch) + contract_drift(core_plugins)
    assert not drift, (
        "Hermes core no longer matches tests/hermes_core_contract.py:\n  "
        + "\n  ".join(sorted(set(drift)))
    )

    # Hook allowlist half of the contract (only hermes_cli.plugins has it).
    missing = set(CORE_CONTRACT["REQUIRED_HOOKS"]) - set(core_plugins.VALID_HOOKS)
    assert not missing, (
        "registered hooks absent from the live core's VALID_HOOKS: "
        f"{sorted(missing)} — they would be accepted and never invoked"
    )


def test_system_prompt_section_names_one_tool_per_job(plugin):
    _mod, ctx, _reg = plugin
    content = ctx.system_prompt_sections["code-intel.defaults"]["content"]
    for tool in ("code_symbols", "code_workspace_symbols", "code_definition",
                 "code_capsule", "code_references", "code_search", "code_diagnostics"):
        assert tool in content
    # bounded fallback must remain explicitly allowed — no hard block
    assert "Fallback is allowed" in content
    assert "bounded" in content


def test_system_prompt_section_degrades_on_old_core(monkeypatch):
    """A core without register_system_prompt_section must not abort register().

    Uses the same runtime stubs as the ``plugin`` fixture: ``register()`` may
    only fail for the *specific* old-core API it is meant to tolerate, never
    because of an unstubbed ``toolsets`` import.
    """
    _install_runtime_stubs(monkeypatch)
    mod = _import_plugin_entrypoint(monkeypatch)

    class OldCtx(FakeCtx):
        register_system_prompt_section = None  # attribute present but not callable

    ctx = OldCtx()
    mod.register(ctx)
    assert not ctx.system_prompt_sections
    assert "pre_llm_call" in ctx.hooks  # per-turn path still installed

    class AncientCtx:
        """Core predating the API entirely — no such attribute at all."""

        def __init__(self):
            self.hooks = {}
            self.commands = {}
            self.tools = []
            self.system_prompt_sections = {}

        def register_hook(self, name, handler):
            self.hooks.setdefault(name, []).append(handler)

        def register_command(self, name, handler=None, description=""):
            self.commands[name] = handler

        def register_skill(self, **kwargs):
            pass

        def register_tool(self, **kwargs):
            self.tools.append(kwargs.get("name"))

    ancient = AncientCtx()
    mod.register(ancient)
    assert not ancient.system_prompt_sections
    assert "pre_llm_call" in ancient.hooks
    assert "transform_tool_result" in ancient.hooks
