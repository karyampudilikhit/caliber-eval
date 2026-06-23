"""Verify each notebook's code cells execute end-to-end without error.

Used in CI / locally as a smoke check that the example notebooks aren't stale.
Doesn't run the notebooks through Jupyter — just concatenates the code cells
of each one and exec()s them in a fresh namespace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    full_source = "\n\n".join(_cell_source(c) for c in code_cells)
    namespace: dict[str, object] = {}
    try:
        exec(compile(full_source, str(path), "exec"), namespace)
    except Exception as e:
        print(f"FAIL {path.name}: {type(e).__name__}: {e}", file=sys.stderr)
        raise


def _cell_source(cell: dict[str, object]) -> str:
    src = cell["source"]
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def main() -> int:
    here = Path(__file__).parent
    notebooks = sorted(here.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1
    for nb in notebooks:
        run_notebook(nb)
        print(f"OK   {nb.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
