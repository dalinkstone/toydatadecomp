"""Shared pytest fixtures for toydatadecomp tests."""

import sys
from pathlib import Path

# Ensure src/ is on the path so imports like `from generators.gen_customers import ...` work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
