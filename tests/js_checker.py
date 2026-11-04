"""Runs web/checker.js under Node from Python, for differential tests of the
Python checkers against the JavaScript ones."""

import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def node_path():
    return shutil.which("node")


def js_verify(pairs) -> list:
    """LeanCheck.verify(prompt, answer) for each (prompt, answer): True, False,
    or None when no JavaScript domain claims the prompt."""
    pairs = list(pairs)
    if not pairs:
        return []
    node = node_path()
    if node is None:
        raise RuntimeError("node not found")
    data = "\n".join(json.dumps([p, a], ensure_ascii=False) for p, a in pairs) + "\n"
    proc = subprocess.run([node, os.path.join(HERE, "js_bridge.js")], input=data, capture_output=True,
                          text=True, encoding="utf-8", check=True)
    out = [json.loads(line) for line in proc.stdout.splitlines() if line]
    if len(out) != len(pairs):
        raise RuntimeError(f"node returned {len(out)} verdicts for {len(pairs)} inputs: {proc.stderr[:500]}")
    for v in out:
        if isinstance(v, str):
            raise RuntimeError(f"checker.js threw: {v}")
    return out
