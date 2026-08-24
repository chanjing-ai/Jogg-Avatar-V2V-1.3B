"""Ensure the repository's ``src/`` packages are importable."""
import os
import sys


def ensure_project_src_path() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
