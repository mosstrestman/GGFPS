#!/usr/bin/env python3
"""Compatibility wrapper for the installable `ggfps simple-demo` command."""

try:
    from ggfps_paper.cli import simple_demo_main
except ModuleNotFoundError as exc:
    raise SystemExit("Install the package first with `python3 -m pip install -e .`.") from exc


if __name__ == "__main__":
    raise SystemExit(simple_demo_main())
