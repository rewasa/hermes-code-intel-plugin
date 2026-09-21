"""
Pytest bootstrap: package identity for the plugin (leak-proof).

The plugin is loaded by Hermes as the package ``code_intel`` (from
``~/.hermes/plugins/code_intel``), but a fresh clone of this repo may sit in a
directory named ``hermes-code-intel-plugin`` / ``code-intel-plugin``. Without an
alias, ``python -m pytest`` from the repo root fails with
``ModuleNotFoundError: 'code_intel' is not a package``.

Why the alias has to be *owned by a fixture*: the repo root also ships a flat
``code_intel.py`` (the AST engine module) and ``pyproject.toml`` puts the repo
root on ``sys.path`` (``pythonpath = ["."]``). The moment the ``code_intel``
alias disappears from ``sys.modules``, ``import code_intel`` resolves to that
flat module — which has no ``__path__`` — so every later
``from code_intel.<submodule> import ...`` dies with ``'code_intel' is not a
package``. One test that drops the alias (to re-import the entrypoint via
``spec_from_file_location``) therefore poisoned every test that ran after it.

The autouse fixture below makes that structure impossible: it repairs the
package alias before each test, and restores both the alias and the submodule
namespace afterwards.
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _install_code_intel_package() -> types.ModuleType:
    """Guarantee the ``code_intel`` package alias points at *ROOT*.

    Repairs in place whenever an importable package object already exists:
    replacing the module object would drop the submodule attributes
    (``code_intel.lsp_bridge``, ...) that ``import code_intel.x`` binds onto it,
    which breaks later ``monkeypatch.setattr("code_intel.lsp_bridge.…")`` calls.
    """
    # Make the checkout importable by its own directory name when it already is
    # one (the Hermes plugin dir) so ``code_intel`` resolves as a real package.
    if str(ROOT.parent) not in sys.path:
        sys.path.insert(0, str(ROOT.parent))

    existing = sys.modules.get("code_intel")
    if existing is not None and getattr(existing, "__path__", None):
        if str(ROOT) not in list(existing.__path__):
            existing.__path__ = [str(ROOT), *existing.__path__]
        if getattr(existing, "__file__", None) is None:
            existing.__file__ = str(ROOT / "__init__.py")
        return existing

    pkg = types.ModuleType("code_intel")
    pkg.__path__ = [str(ROOT)]
    pkg.__file__ = str(ROOT / "__init__.py")
    sys.modules["code_intel"] = pkg
    return pkg


def _rebind_submodule_attributes(pkg: types.ModuleType) -> None:
    """Re-attach ``code_intel.<sub>`` attributes onto *pkg* from ``sys.modules``.

    The import system binds every imported submodule as an attribute of its
    parent package (``import code_intel.lsp_bridge`` sets
    ``code_intel.lsp_bridge``). Restoring ``sys.modules`` without restoring that
    binding leaves the package *looking* like it has no submodules, which breaks
    ``monkeypatch.setattr("code_intel.lsp_bridge.…")``.
    """
    prefix = "code_intel."
    for name, mod in list(sys.modules.items()):
        if not name.startswith(prefix) or "." in name[len(prefix):]:
            continue
        setattr(pkg, name[len(prefix):], mod)


_install_code_intel_package()


@pytest.fixture(autouse=True)
def _code_intel_package_identity():
    """Guarantee ``code_intel`` is importable as a package for every test.

    Tests legitimately swap ``sys.modules['code_intel']`` (to exec the real
    plugin entrypoint), but that swap must never outlive the test.
    """
    _install_code_intel_package()
    _rebind_submodule_attributes(sys.modules["code_intel"])
    saved_submodules = {
        name: mod for name, mod in sys.modules.items() if name.startswith("code_intel.")
    }
    try:
        yield
    finally:
        for name in [n for n in sys.modules if n.startswith("code_intel.")]:
            if name not in saved_submodules:
                del sys.modules[name]
        sys.modules.update(saved_submodules)
        pkg = _install_code_intel_package()
        _rebind_submodule_attributes(pkg)
