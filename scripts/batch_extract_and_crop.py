#!/usr/bin/env python3
"""
CLI wrapper for nuxplore batch extraction + crop export.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_python_path() -> None:
    repo_python = Path(__file__).resolve().parents[1] / "python"
    repo_python_str = str(repo_python)
    if repo_python.exists() and repo_python_str not in sys.path:
        sys.path.insert(0, repo_python_str)


def main() -> int:
    _ensure_repo_python_path()
    from nuxplore.batch import main as batch_main

    return batch_main()


if __name__ == "__main__":
    raise SystemExit(main())
