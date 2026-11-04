import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lean_oracle  # noqa: E402
from js_checker import node_path  # noqa: E402


@pytest.fixture(scope="session")
def lean():
    path = lean_oracle.find_lean()
    if path is None:
        pytest.skip("no Lean 4 toolchain (set LEAN or put lean on PATH)")
    return path


@pytest.fixture(scope="session")
def node():
    path = node_path()
    if path is None:
        pytest.skip("node not found")
    return path
