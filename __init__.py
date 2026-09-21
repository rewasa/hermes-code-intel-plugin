from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
from pathlib import Path
import os
import json

if TYPE_CHECKING:  # Hermes runtime only — not present in CI/standalone
    from hermes_cli.plugins import PluginContext

# Slash command handler
def _handle_code_intel_slash(raw_args: str) -> Optional[str]:
    from .code_intel import get_symbol_cache_stats, clear_symbol_cache

    argv = raw_args.strip().split()
    if not argv or argv[0] in ("help", "-h", "--help"):
        return (
            "/code-intel — AST code intelligence management\n\n"
            "Subcommands:\n"
            "  status   Show symbol cache, LSP health, workspace roots\n"
            "  clear    Clear the AST symbol cache to free memory\n"
        )

    sub = argv[0]
    if sub == "status":
        from .code_intel import get_symbol_cache_stats
        stats = get_symbol_cache_stats()
        lines = ["[code_intel] Status:"]
        lines.append(f"  Symbol cache: {stats['entries']} parsed AST files in memory.")

        # LSP health
        try:
            from .lsp_bridge import get_lsp_manager, _LANGUAGE_SERVERS, _find_workspace_root
            mgr = get_lsp_manager()
            active = []
            for lang_key, cfgs in _LANGUAGE_SERVERS.items():
                for cfg in cfgs:
                    cmd = cfg.get("command")
                    if cmd:
                        active.append(f"{lang_key} ({cmd})")
            bridge_count = len(mgr._bridges)
            lines.append(f"  LSP bridges: {bridge_count} active")
            lines.append(f"  Registered servers: {', '.join(active) if active else 'none'}")

            # Per-bridge details
            for bridge_id, bridge in mgr._bridges.items():
                info = bridge.get_server_info() if hasattr(bridge, 'get_server_info') else {}
                alive = "✓" if info.get("alive") else "✗"
                init = "init" if info.get("initialized") else "pending"
                diag = info.get("diagnostic_files", 0)
                lines.append(f"    {bridge_id}: {alive} {init} diag_files={diag}")

            # Workspace roots
            roots = set()
            for b in mgr._bridges.values():
                if getattr(b, "root_uri", None):
                    roots.add(b.root_uri)
            if roots:
                lines.append(f"  Workspace roots: {', '.join(roots)}")

            # Cache stats per bridge
            total_diag = sum(
                len(b._diagnostics_cache) if hasattr(b, '_diagnostics_cache') else 0
                for b in mgr._bridges.values()
            )
            if total_diag:
                lines.append(f"  Cached diagnostics: {total_diag} files across bridges")
        except Exception as exc:
            lines.append(f"  LSP info unavailable: {exc}")

        return "\n".join(lines)

    if sub == "clear":
        clear_symbol_cache()
        return "[code_intel] AST symbol cache cleared successfully."

    return f"Unknown subcommand: {sub}\nRun `/code-intel help` for usage."

# Hook handler
def _on_session_end(**kwargs: Any) -> None:
    """Persist AST caches to disk at session end, then clear memory."""
    from .code_intel import persist_symbol_cache, clear_symbol_cache
    saved = persist_symbol_cache()
    clear_symbol_cache()

    # Proactively drop bounded per-session steering state.
    session_id = kwargs.get("session_id")
    task_id = kwargs.get("task_id")
    from .code_intel_nudges import forget_session as forget_nudge_session
    from .code_intel_defaults import forget_session as forget_default_session
    forget_nudge_session(session_id or task_id)
    forget_default_session(session_id, task_id)


def register(ctx: PluginContext) -> None:
    import toolsets  # Hermes runtime only — imported lazily so CI/standalone can import this module

    # 0. Register plugin-provided skill (opt-in via skill_view("code_intel:native-code-intelligence"))
    _plugin_dir = Path(__file__).parent
    _skill_md = _plugin_dir / "skills" / "native-code-intelligence.md"
    if _skill_md.exists():
        ctx.register_skill(
            name="native-code-intelligence",
            path=_skill_md,
            description="Native tree-sitter + ast-grep code intelligence tools for Hermes agent. Replaces deprecated LSP MCP with in-process AST parsing.",
        )

    # 1. Register command & hooks
    ctx.register_command(
        "code-intel",
        handler=_handle_code_intel_slash,
        description="Manage AST-aware code intelligence and symbol caching."
    )
    ctx.register_hook("on_session_end", _on_session_end)

    # C1: frozen system-prompt section — the ONLY plugin surface that reaches
    # every session (main agent AND delegated subagents) unconditionally, once
    # per session, before any tool selection. Per-turn hooks (pre_llm_call /
    # transform_tool_result) only reach turns that already went wrong or that
    # happen to match a heuristic; this is the reliable default.
    # Bounded to a few hundred chars; older cores without the API are skipped.
    try:
        _register_section = getattr(ctx, "register_system_prompt_section", None)
        if callable(_register_section):
            _register_section(
                id="code-intel.defaults",
                content=(
                    "Code work default — semantic tools are installed and preferred:\n"
                    "- what is in a file: `code_symbols(path)` before reading it whole\n"
                    "- find a symbol: `code_workspace_symbols(query)`\n"
                    "- where it is defined: `code_definition(path, line)`; "
                    "`code_capsule(path, line)` for signature+doc+definition in one call\n"
                    "- who uses it: `code_references(path, line, group_by_file=True)` "
                    "before any rename/refactor\n"
                    "- structural search: `code_search(path, preset=...)` (AST-aware, "
                    "no comment/string false positives)\n"
                    "- after edits: `code_diagnostics(path)`\n"
                    "Fallback is allowed and never blocked: when a semantic tool is "
                    "unavailable, errors, or its output is insufficient, read targeted "
                    "line ranges or run a text/shell search bounded to specific "
                    "paths/globs. Do not stall on semantics."
                ),
                position="after_memory",
                max_chars=1200,
            )
    except Exception as e:
        import logging
        logging.getLogger("code_intel").warning(
            f"code_intel: system prompt section registration failed ({e}) — "
            f"relying on per-turn hooks only"
        )

    # C2: pre_llm_call hook — one compact default + bounded auto-context.
    def _pre_llm_call_inject_context(**kwargs: Any) -> Optional[str]:
        """Inject (a) a one-shot coding default and (b) bounded symbol context.

        Two real defects fixed here:
        * the previous body read only ``kwargs["messages"]``, which the
          production call site (``agent/turn_context.py``) never passes — it
          passes ``user_message`` + ``conversation_history``. The hook was a
          permanent no-op in real sessions while still passing its own tests.
        * context was rebuilt for every mentioned file on every turn with no
          session cap, so a long session re-paid the same symbol scan.

        Fail-safe by design: guidance never blocks a fallback tool, and any
        exception returns ``None`` (no injection) rather than touching the turn.
        """
        try:
            from .code_intel_defaults import (
                build_default_guidance,
                normalize_messages,
                take_context_slot,
            )

            session_id = kwargs.get("session_id")
            task_id = kwargs.get("task_id")
            messages = normalize_messages(
                user_message=kwargs.get("user_message"),
                messages=kwargs.get("messages"),
                conversation_history=kwargs.get("conversation_history"),
            )

            parts = []
            guidance = build_default_guidance(messages, session_id, task_id)
            if guidance:
                parts.append(guidance)

            last_msg = ""
            for m in reversed(messages):
                content = m.get("content")
                if m.get("role") == "user" and isinstance(content, str):
                    last_msg = content
                    break
            if not last_msg:
                return "\n".join(parts) if parts else None

            # Detect file paths in the message (simple heuristic)
            import re
            file_refs = re.findall(
                r'(?:^|[\s"\'])([\w/_.-]+\.(?:py|ts|tsx|js|jsx|rs|go|java))',
                last_msg
            )
            if not file_refs:
                return "\n".join(parts) if parts else None

            # Consume a context slot only once there IS work to do. Charging the
            # budget before this point burned all three slots on coding turns that
            # mentioned no file, leaving the actual symbol scans unbudgeted.
            if not take_context_slot(session_id, task_id):
                return "\n".join(parts) if parts else None

            # Limit to 3 files to keep context compact
            file_refs = file_refs[:3]

            from .code_intel import code_symbols_tool, detect_language
            for fref in file_refs:
                path = fref
                if not os.path.isabs(path):
                    path = os.path.join(os.getcwd(), path)
                if not os.path.exists(path):
                    continue
                lang = detect_language(path)
                if lang:
                    try:
                        symbols_json = code_symbols_tool(path=path, pattern="", include_body=False)
                        symbols = json.loads(symbols_json) if isinstance(symbols_json, str) else symbols_json
                        sym_list = symbols if isinstance(symbols, list) else symbols.get("symbols", [])
                        if sym_list:
                            summary = f"[auto-context] {fref}: {len(sym_list)} symbols"
                            # Top 8 symbols only
                            for s in sym_list[:8]:
                                name = s.get("name", "?")
                                kind = s.get("kind", "")
                                line = s.get("line", "")
                                summary += f"\n  L{line} {kind} {name}"
                            parts.append(summary)
                    except Exception:
                        pass

            return "\n".join(parts) if parts else None
        except Exception as e:
            import logging
            logging.getLogger("code_intel").debug(f"pre_llm_call hook error: {e}")
            return None

    ctx.register_hook("pre_llm_call", _pre_llm_call_inject_context)

    # M1: mechanical steering via transform_tool_result — fires on EVERY
    # handle_function_call regardless of spawn path (Paseo, delegate_task,
    # plain CLI), unlike the old _CODE_INTEL_STEERING text block below which
    # only reaches tools.delegate_tool subagents. See code_intel_nudges.py.
    def _transform_tool_result_nudge(**kwargs: Any) -> Optional[str]:
        try:
            from .code_intel_nudges import build_nudge
            tool_name = kwargs.get("tool_name", "")
            if tool_name not in ("read_file", "search_files", "terminal"):
                return None
            args = kwargs.get("args") or {}
            result = kwargs.get("result")
            session_id = kwargs.get("session_id")
            task_id = kwargs.get("task_id")
            hint = build_nudge(tool_name, args, result, session_id, task_id)
            if not hint or not isinstance(result, str):
                return None
            return result + hint
        except Exception as e:
            import logging
            logging.getLogger("code_intel").debug(f"nudge hook error: {e}")
            return None

    ctx.register_hook("transform_tool_result", _transform_tool_result_nudge)

    # B9: on an old Hermes core, an unknown hook name is silently accepted
    # (stored + warned) but never invoked — a stealth total outage for the
    # nudge feature. Detect that here so it fails loudly instead of quietly:
    # if the running core doesn't advertise transform_tool_result support,
    # log one unmistakable warning pointing at the real cause.
    try:
        from hermes_cli.plugins import VALID_HOOKS
        if "transform_tool_result" not in VALID_HOOKS:
            import logging
            logging.getLogger("code_intel").warning(
                "code_intel: this Hermes core's VALID_HOOKS does not include "
                "'transform_tool_result' — the code_intel nudge feature is "
                "registered but will NEVER fire. Upgrade Hermes or the nudge "
                "hook is silently dead."
            )
    except Exception as e:
        import logging
        logging.getLogger("code_intel").warning(
            f"code_intel: could not verify transform_tool_result hook support "
            f"on this Hermes core ({e}) — nudge feature may be silently dead."
        )

    # 2. Inject the code_intel toolset definition
    if "code_intel" not in toolsets.TOOLSETS:
        toolsets.TOOLSETS["code_intel"] = {
            "description": "AST-aware code intelligence: symbol extraction, structural search, safe refactoring, LSP go-to-definition and find-all-references (tree-sitter + ast-grep + LSP)",
            "tools": [
                "code_symbols", "code_search", "code_refactor",
                "code_definition", "code_references", "code_diagnostics",
                "code_callers", "code_callees", "code_capsule",
                "code_workspace_summary", "code_impact", "code_tests_for_symbol",
                "code_query", "code_rename", "code_workspace_symbols",
                "code_hover", "code_type_definition",
                "code_signatures", "code_action",
            ],
            "includes": []
        }

    # Inject into core platforms so it's globally available
    new_tools = [
        "code_symbols", "code_search", "code_refactor",
        "code_definition", "code_references", "code_diagnostics",
        "code_callers", "code_callees", "code_capsule",
        "code_workspace_summary", "code_impact", "code_tests_for_symbol",
        "code_query", "code_rename", "code_workspace_symbols",
        "code_hover", "code_type_definition",
        "code_signatures", "code_action",
    ]
    for t in new_tools:
        if t not in toolsets._HERMES_CORE_TOOLS:
            toolsets._HERMES_CORE_TOOLS.append(t)

    for preset in ["hermes-acp", "hermes-api-server"]:
        if preset in toolsets.TOOLSETS:
            tools = toolsets.TOOLSETS[preset]["tools"]
            for t in new_tools:
                if t not in tools:
                    tools.append(t)

    # Load our tools
    from . import code_intel

    # Register LSP-backed tools (definition, references, diagnostics, callers, callees).
    # These are NOT auto-registered at import time — must be invoked explicitly.
    try:
        from .lsp_bridge import register_lsp_tools
        register_lsp_tools()
    except Exception as e:
        import logging
        logging.getLogger("code_intel").warning(f"LSP tool registration failed: {e}")

    # Restore persisted symbol cache from disk (B5)
    loaded = code_intel.load_symbol_cache()
    if loaded:
        import logging
        logging.getLogger("code_intel").info(f"Restored {loaded} symbol cache entries from disk")

    # Inject steering hints directly into the registry schemas of the builtin tools!
    import tools.registry
    
    sf_entry = tools.registry.registry.get_entry("search_files")
    if sf_entry and "description" in sf_entry.schema:
        hint = (
            "\n\nFor AST-aware structural search inside source files "
            "(find function calls, imports, decorators, etc.), prefer code_search — "
            "it understands syntax and won't match comments or strings."
        )
        if hint not in sf_entry.schema["description"]:
            sf_entry.schema["description"] += hint

    rf_entry = tools.registry.registry.get_entry("read_file")
    if rf_entry and "description" in rf_entry.schema:
        hint = (
            "\n\nFor understanding what a file contains (list of functions, classes, "
            "methods with line numbers and signatures), prefer code_symbols — "
            "much more token-efficient than reading the entire file."
        )
        if hint not in rf_entry.schema["description"]:
            rf_entry.schema["description"] += hint

    p_entry = tools.registry.registry.get_entry("patch")
    if p_entry and "description" in p_entry.schema:
        hint = (
            "\n\nFor AST-aware structural replacement (rename patterns, wrap "
            "functions, add parameters across a file), prefer code_refactor — "
            "matches by syntax tree, not raw text. Dry-run by default."
        )
        if hint not in p_entry.schema["description"]:
            p_entry.schema["description"] += hint

    # Additional steering for new tools
    cd_entry = tools.registry.registry.get_entry("code_definition")
    if cd_entry and "description" in cd_entry.schema:
        hint = (
            "\n\nWhen you need to understand HOW a symbol is used across the project, "
            "call code_references AFTER code_definition. For a quick one-shot overview, use code_capsule instead."
        )
        if hint not in cd_entry.schema["description"]:
            cd_entry.schema["description"] += hint

    cr_entry = tools.registry.registry.get_entry("code_references")
    if cr_entry and "description" in cr_entry.schema:
        hint = (
            "\n\nBefore renaming or refactoring a symbol, always run code_references first "
            "to see all impacted files. Use group_by_file=True to save tokens on large codebases. "
            "For a compact summary, use code_capsule."
        )
        if hint not in cr_entry.schema["description"]:
            cr_entry.schema["description"] += hint

    cs_entry = tools.registry.registry.get_entry("code_symbols")
    if cs_entry and "description" in cs_entry.schema:
        hint = (
            "\n\nFor cross-file navigation, first use code_symbols on the current file to confirm "
            "the symbol exists, then use code_definition or code_references for deeper analysis."
        )
        if hint not in cs_entry.schema["description"]:
            cs_entry.schema["description"] += hint

    # ── Refresh delegate_task toolsets so code_intel appears in subagent toolset list ──
    # _SUBAGENT_TOOLSETS and _TOOLSET_LIST_STR are computed at import time (BEFORE this
    # plugin registers code_intel). We must recompute them so the delegate_task schema
    # correctly advertises code_intel as an available toolset.
    try:
        import tools.delegate_tool as dt
        # Upstream delegate_tool.py (2026-07 refactor) dropped the module-level
        # _EXCLUDED_TOOLSET_NAMES constant and moved exclusion into the runtime
        # _strip_blocked_tools() helper. getattr() keeps this plugin working on
        # both old and new Hermes builds instead of dying with AttributeError
        # (which used to abort this whole block and silently disable the
        # code_intel forcing/steering below).
        _excluded = getattr(
            dt, "_EXCLUDED_TOOLSET_NAMES",
            frozenset({"debugging", "safe", "delegation", "moa", "rl", "code_execution"}),
        )
        dt._SUBAGENT_TOOLSETS = sorted(
            name for name, defn in toolsets.TOOLSETS.items()
            if name not in _excluded
            and not name.startswith("hermes-")
            and not all(t in dt.DELEGATE_BLOCKED_TOOLS for t in defn.get("tools", []))
        )
        dt._TOOLSET_LIST_STR = ", ".join(f"'{n}'" for n in dt._SUBAGENT_TOOLSETS)

        # Also refresh the DELEGATE_TASK_SCHEMA toolset descriptions
        if "toolsets" in dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]:
            ts_prop = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]["toolsets"]
            ts_prop["description"] = (
                "Toolsets to enable for this subagent. "
                "Default: inherits your enabled toolsets. "
                f"Available toolsets: {dt._TOOLSET_LIST_STR}. "
                "Common patterns: ['terminal', 'file'] for code work, "
                "['web'] for research, ['browser'] for web interaction, "
                "['terminal', 'file', 'web'] for full-stack tasks."
            )
        if "tasks" in dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]:
            task_ts = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"].get("toolsets")
            if task_ts:
                task_ts["description"] = (
                    f"Toolsets for this specific task. Available: {dt._TOOLSET_LIST_STR}. "
                    "Use 'web' for network access, 'terminal' for shell, 'browser' for web interaction."
                )
        import logging
        logging.getLogger("code_intel").info(
            f"Refreshed delegate_task toolsets: {dt._TOOLSET_LIST_STR}"
        )

        # ── FORCE code_intel into every subagent + inject steering ──
        # Renato's rule: code_intel must be DEFAULT for ALL agents, and
        # subagents must KNOW which tools exist and how to use them.
        # Hermes 2026-09 moved DEFAULT_TOOLSETS from tools.delegate_tool to
        # tools.delegate_tool_toolsets (old path is a plugin-compat shim,
        # removed 2026-09-14). Fall back to the old location for older builds.
        try:
            import tools.delegate_tool_toolsets as dtt
        except ImportError:
            dtt = dt
        if "code_intel" not in dtt.DEFAULT_TOOLSETS:
            dtt.DEFAULT_TOOLSETS.append("code_intel")

        _CODE_INTEL_STEERING = (
            "\n\n## 🧠 Code Intelligence Tools (PREFER over read_file/grep/patch)\n"
            "You have native AST + LSP code-intel tools. USE THEM FIRST for any code task.\n\n"
            "**Discovery (instead of read_file on whole files):**\n"
            "- `code_workspace_summary(path)` — monorepo overview: apps, packages, entry points.\n"
            "- `code_symbols(path)` — list functions/classes/methods in a file with line numbers.\n"
            "- `code_workspace_symbols(query)` — fuzzy find a symbol across the entire workspace.\n\n"
            "**Navigation (instead of grep):**\n"
            "- `code_definition(path, line)` — jump to where a symbol is defined.\n"
            "- `code_references(path, line, group_by_file=True)` — find ALL usages of a symbol.\n"
            "- `code_callers(path, line)` / `code_callees(path, line)` — call graph.\n"
            "- `code_capsule(path, line)` — one-shot: signature + doc + definition + top refs.\n"
            "- `code_hover(path, line)` — type signature + docstring without reading source.\n"
            "- `code_signatures(path, line)` — parameter hints inside a call site.\n"
            "- `code_type_definition(path, line)` — jump to the TYPE shape (interface/class).\n\n"
            "**Search (instead of search_files for code):**\n"
            "- `code_search(path, preset='function_calls'|'imports'|'decorator_calls'|...)` — "
            "AST-aware, won't match comments/strings.\n\n"
            "**Refactoring (instead of patch + sed):**\n"
            "- `code_rename(path, line, new_name, dry_run=True)` — semantic rename across files.\n"
            "- `code_refactor(path, pattern, rewrite, dry_run=True)` — AST structural rewrite.\n"
            "- `code_action(path, line)` — quick-fixes / organize imports / source.fixAll.\n\n"
            "**Quality:**\n"
            "- `code_diagnostics(path)` — LSP errors/warnings. RUN AFTER editing code.\n"
            "- `code_impact(path, line)` — blast radius before refactor.\n"
            "- `code_tests_for_symbol(path, line)` — find tests covering a symbol.\n\n"
            "**Workflow:** capsule → references → impact → rename/refactor (dry_run) → apply → diagnostics.\n"
            "**Anti-pattern:** read_file on a 1000-line file when code_symbols would give you what you need in 50 tokens."
        )

        _orig_build_prompt = dt._build_child_system_prompt
        def _patched_build_prompt(*args, **kwargs):
            base = _orig_build_prompt(*args, **kwargs)
            if _CODE_INTEL_STEERING not in base:
                base = base + _CODE_INTEL_STEERING
            return base
        dt._build_child_system_prompt = _patched_build_prompt

        _orig_build_agent = dt._build_child_agent
        def _patched_build_agent(*args, **kwargs):
            ts = kwargs.get("toolsets")
            if ts is not None and "code_intel" not in ts:
                kwargs["toolsets"] = list(ts) + ["code_intel"]
            return _orig_build_agent(*args, **kwargs)
        dt._build_child_agent = _patched_build_agent

        logging.getLogger("code_intel").info(
            "code_intel: forced into DEFAULT_TOOLSETS + steering injected into child prompts"
        )
    except Exception as e:
        import logging
        logging.getLogger("code_intel").warning(f"Failed to refresh delegate_task toolsets: {e}")
