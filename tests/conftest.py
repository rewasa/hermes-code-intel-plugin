"""
Pytest bootstrap making the plugin importable regardless of the checkout
directory name.

The plugin is loaded by Hermes as the package ``code_intel`` (from
``~/.hermes/plugins/code_intel``), but a fresh clone of this repo may sit in a
directory named ``hermes-code-intel-plugin`` / ``code-intel-plugin``. Without
this shim, ``python -m pytest`` from the repo root fails collection with
``ModuleNotFoundError: 'code_intel' is not a package``.

Two strategies are applied, so tests run from BOTH a clone and the Hermes
plugin dir:
  * put the parent directory on ``sys.path`` (works when the dir is already
    named ``code_intel``);
  * otherwise register the repo root as a namespace package named ``code_intel``
    so ``code_intel.code_intel`` and ``code_intel.lsp_bridge`` resolve here.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1) Make the checkout importable by its own directory name.
sys.path.insert(0, str(ROOT.parent))

# 2) If the checkout dir is not already named "code_intel", alias it so the
#    package-style imports used by the tests resolve to this repo root.
if ROOT.name != "code_intel" and "code_intel" not in sys.modules:
    pkg = types.ModuleType("code_intel")
    pkg.__path__ = [str(ROOT)]
    sys.modules["code_intel"] = pkg
