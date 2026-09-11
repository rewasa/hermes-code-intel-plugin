#!/usr/bin/env python3
"""code_intel diagnostics fixtures: TS + Python, clean + intentional error.

Used by code_intel_health_check.py and tests. Keep errors INTENTIONAL.
"""
import sys, os

FIXTURES = {
    "clean.ts": "const cleanValue: number = 1;\nconsole.log(cleanValue);\n",
    # Intentional TS errors: type mismatch (2322) + undefined name (2304)
    "error.ts": 'const badValue: number = "oops";\nconsole.log(notDefinedName);\n',
    "clean.py": "def clean_fn() -> int:\n    return 1\n",
    # Intentional Python error: undefined name
    "error.py": "def bad_fn() -> int:\n    return not_defined_name\n",
}

def ensure_fixtures(base_dir: str) -> dict:
    """Write fixtures under base_dir; return {relname: abs_path}."""
    os.makedirs(base_dir, exist_ok=True)
    out = {}
    for name, content in FIXTURES.items():
        p = os.path.join(base_dir, name)
        with open(p, "w") as f:
            f.write(content)
        out[name] = p
    return out

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/.hermes/scripts/.code_intel_fixtures")
    print(ensure_fixtures(base))