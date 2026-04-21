#!/usr/bin/env python3
"""Full experiment-style demo on the 2D Styblinski-Tang benchmark.

Compatibility wrapper for the installable `ggfps demo` command.
"""

try:
    from ggfps_paper.cli import demo_main
except ModuleNotFoundError as exc:
    raise SystemExit("Install the package first with `python3 -m pip install -e .`.") from exc


if __name__ == "__main__":
    raise SystemExit(demo_main())
