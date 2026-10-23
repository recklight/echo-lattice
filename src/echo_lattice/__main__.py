"""Run the command-line interface with ``python -m echo_lattice``.

Equivalent to the ``echo-lattice`` console script, and usable straight from a
checkout without installing (``PYTHONPATH=src``).
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess test
    main()
