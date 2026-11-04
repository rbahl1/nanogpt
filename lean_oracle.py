"""Compiles declarations with a real Lean 4 toolchain.

The domains' Python checkers are fast stand-ins for Lean; this module is the
ground truth they are tested against, and evaluate.py --lean uses it to
compile every answer a model gives. All declarations of a batch go into one
file, so a batch of thousands costs one Lean start-up.

Lean is found through the LEAN environment variable (the path of the `lean`
binary) or on PATH. Only core Lean is used: no Mathlib, no Lake project.

    python lean_oracle.py --n 200     # compile 200 generated declarations per domain
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile


def find_lean():
    path = os.environ.get("LEAN") or shutil.which("lean")
    return path if path and os.path.exists(path) else None


# Declarations are text we did not necessarily write (fuzzed mutants, model
# samples). Lean runs code at elaboration time (`#eval`, custom elaborators,
# `run_tac`), and a batch shares one file, so one declaration could also change
# how the others are judged: an `axiom` or `notation` after its proof, or an
# unclosed comment or string swallowing its neighbours. So a declaration is
# compiled only if it is a single `example` with none of Lean's command
# keywords, comments, strings or escape hatches anywhere after its first word;
# everything else is rejected unread. This guards against accidents in
# generated text; it is not a sandbox for adversarial input.
_COMMANDS = (
    "import prelude open export namespace section end variable universe omit include set_option in "
    "axiom theorem lemma def abbrev instance example opaque structure class inductive coinductive mutual "
    "attribute notation infix infixl infixr prefix postfix macro macro_rules syntax elab elab_rules "
    "declare_syntax_cat initialize builtin_initialize deriving noncomputable unsafe partial private protected "
    "local scoped where add_decl_doc register_simp_attr simproc dsimproc binder_predicate grind_pattern "
    "unif_hint implemented_by extern sorry admit"
).split()
_FORBIDDEN = re.compile(r'#|@\[|/-|-/|--|"|\brun_|native_|\b(' + "|".join(_COMMANDS) + r")\b")


def is_plain_example(declaration: str) -> bool:
    return declaration.startswith("example ") and not _FORBIDDEN.search(declaration, len("example"))


# Every batch file starts with a declaration that compiles. Lean places a few
# errors at the very start of the file instead of in the declaration that caused
# them (an "unsolved goals" after a lexer error inside `by`, for one); such a
# message lands on this sentinel, and the batch is then split in halves until
# the declaration responsible is alone.
_SENTINEL = ["section", "example : True := trivial", "end"]


def check_declarations(declarations, lean: str = None, timeout: float = 900.0) -> list:
    """Compiles each declaration (e.g. `example (p : Prop) : p → p := fun h => h`)
    and returns one bool per declaration: True iff it is a plain `example` and
    Lean reported neither an error nor a use of `sorry` inside it.

    Each declaration is wrapped in `section ... end`, so an unfinished one (whose
    parse error Lean reports at the next token) is still blamed on itself."""
    lean = lean or find_lean()
    if lean is None:
        raise RuntimeError("no Lean 4 binary found: set LEAN or put `lean` on PATH")
    declarations = list(declarations)
    plain = [is_plain_example(d) for d in declarations]
    todo = [i for i, ok in enumerate(plain) if ok]
    verdict = _compile([declarations[i] for i in todo], lean, timeout)
    out = [False] * len(declarations)
    for i, ok in zip(todo, verdict):
        out[i] = ok
    return out


def _compile(declarations, lean: str, timeout: float) -> list:
    """Verdicts for plain declarations, compiled together in one file."""
    if not declarations:
        return []
    lines, spans = list(_SENTINEL), []
    for d in declarations:
        start = len(lines) + 1
        lines.append("section")
        lines.extend(d.split("\n"))
        lines.append("end")
        spans.append((start, len(lines)))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "Check.lean")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        # Lean stops reporting after maxErrors (default 100) errors, and everything
        # after that point would look as if it compiled; lift the limit.
        proc = subprocess.run([lean, "-DmaxErrors=1000000", "--json", path], capture_output=True, text=True,
                              timeout=timeout)
    if "maximum number of errors" in proc.stdout:
        raise RuntimeError("Lean stopped reporting errors partway through the batch; its verdicts are incomplete")
    bad, stray = set(), False
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Errors fail a declaration, and so does the `sorry` warning. Linter
        # warnings (an unused hypothesis, say) do not: the proof is still valid.
        if msg.get("severity") != "error" and msg.get("kind") != "hasSorry":
            continue
        # A message past a declaration's `end` (Lean reports some errors at the
        # next token) belongs to the last declaration that started before it.
        row = msg["pos"]["line"]
        owner = max((i for i, span in enumerate(spans) if span[0] <= row), default=None)
        if owner is None:
            stray = True
        else:
            bad.add(owner)
    if stray:
        if len(declarations) == 1:
            return [False]
        half = len(declarations) // 2
        return _compile(declarations[:half], lean, timeout) + _compile(declarations[half:], lean, timeout)
    if proc.returncode != 0 and not bad:
        raise RuntimeError(f"Lean failed without a located message:\n{proc.stdout}\n{proc.stderr}")
    return [i not in bad for i in range(len(declarations))]


def check_in_chunks(declarations, lean: str = None, chunk: int = 2000, timeout: float = 900.0) -> list:
    """check_declarations over chunks of `chunk` declarations (one Lean process
    each), for very large batches."""
    declarations = list(declarations)
    out = []
    for i in range(0, len(declarations), chunk):
        out.extend(check_declarations(declarations[i:i + chunk], lean, timeout))
    return out


def main(argv=None) -> int:
    import formal
    from data import sample_problems

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200, help="declarations per domain")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    lean = find_lean()
    if lean is None:
        print("no Lean 4 binary found: set LEAN or put `lean` on PATH")
        return 2
    failures = 0
    for name in formal.DOMAINS:
        problems = sample_problems(args.n, args.split, args.seed, domains=name, unique=True)
        decls = [p.prompt + " " + p.reference for p in problems]
        ok = check_declarations(decls, lean)
        bad = [d for d, o in zip(decls, ok) if not o]
        failures += len(bad)
        print(f"{name}: {sum(ok)}/{len(ok)} references compile")
        for d in bad[:5]:
            print("   ", d)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
