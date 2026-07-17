#!/usr/bin/env python3
"""Compatibility entry point for the unified Python-main diagnostics."""

from rehab_engine import print_diagnostics, run_diagnostics


if __name__ == "__main__":
    result = run_diagnostics()
    print_diagnostics(result)
    raise SystemExit(1 if result.errors() else 0)
