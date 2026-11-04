"""The linear domain: its generator, the Int semantics of the equations, split
keys, its checker against web/checker.js (the specification: they must agree
on every case, here references, every proof form and many kinds of mutated
answers and prompts), and against Lean 4 (soundness: every answer the checker
accepts compiles)."""

import collections
import functools
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time

import data
import formal
from formal import linear
from formal.linear import DOMAIN, HEAD, MAX_NESTING, PROOFS, TAIL
from js_checker import js_verify
import lean_oracle
from tokenizer import Tokenizer

N_PROBLEMS = 600
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@functools.lru_cache(maxsize=None)
def problems(n=N_PROBLEMS, seed=0) -> tuple:
    rng = random.Random(seed)
    return tuple(DOMAIN.sample(rng) for _ in range(n))


def root(prompt):
    return linear.solution_of(prompt)


def sides(prompt):
    """(left tree, right tree) of a claimed prompt."""
    return linear._parse(prompt)[1:]


def eq(left: str, right: str) -> str:
    return f"{HEAD}{left} = {right}{TAIL}"


# ---------------------------------------------------------------- generator

def test_references_verify_and_fit():
    tok = Tokenizer()
    for p in problems():
        assert p.domain == "linear" and DOMAIN.claims(p.prompt), p
        assert DOMAIN.verify(p.prompt, p.reference), p
        assert p.reference == f"{root(p.prompt)} := by omega", p
        for proof in PROOFS:
            assert DOMAIN.verify(p.prompt, f"{root(p.prompt)} := {proof}"), (p, proof)
        ids, _ = tok.example_ids(p.prompt, p.reference)
        assert len(ids) <= 96, p
        # The key is itself a prompt of the same problem, and its own key.
        key = DOMAIN.split_key(p.prompt)
        assert root(key) == root(p.prompt) and DOMAIN.split_key(key) == key, (p, key)
        assert formal.domain_for(p.prompt) is DOMAIN  # no other domain claims it


def test_documents_verify():
    rng = random.Random(5)
    proofs = collections.Counter()
    for p in problems():
        for _ in range(3):
            doc = DOMAIN.document(p, rng)
            assert doc.startswith(p.prompt + " "), doc
            answer = doc[len(p.prompt) + 1:]
            assert DOMAIN.verify(p.prompt, answer), doc
            proofs[answer.split(" := ", 1)[1]] += 1
    assert set(proofs) == set(PROOFS), proofs


_SAMPLE_IN_SUBPROCESS = """
import json, random
from formal.linear import DOMAIN
rng = random.Random(7)
ps = [DOMAIN.sample(rng) for _ in range(50)]
print(json.dumps([[p.prompt, p.reference, DOMAIN.document(p, rng)] for p in ps]))
"""


def test_generator_is_deterministic():
    assert list(problems(50, 7)) == list(problems.__wrapped__(50, 7))
    assert list(problems(50, 7)) != list(problems(50, 8))
    rng1, rng2 = random.Random(3), random.Random(3)
    assert [DOMAIN.document(p, rng1) for p in problems()] == [DOMAIN.document(p, rng2) for p in problems()]
    rng = random.Random(7)
    here = [[p.prompt, p.reference, DOMAIN.document(p, rng)] for p in [DOMAIN.sample(rng) for _ in range(50)]]
    for hash_seed in ("1", "2"):
        out = subprocess.run([sys.executable, "-c", _SAMPLE_IN_SUBPROCESS], cwd=ROOT, capture_output=True, text=True,
                             check=True, env={**os.environ, "PYTHONHASHSEED": hash_seed})
        assert json.loads(out.stdout) == here


def _form(prompt: str) -> str:
    left, right = prompt[len(HEAD):-len(TAIL)].split(" = ")
    if "(" in left:
        return "a(x+b)=c"
    return "ax+b=cx+d" if "x" in left and "x" in right else "ax+b=c" if "x" in left else "c=ax+b"


def test_distribution():
    """No constant answer scores well: the most common root covers at most 10%
    of problems."""
    ps = problems(5000, 1)
    roots = collections.Counter(root(p.prompt) for p in ps)
    top, top_count = roots.most_common(1)[0]
    share = top_count / len(ps)
    forms = collections.Counter(_form(p.prompt) for p in ps)
    shares = sorted(c / len(ps) for c in roots.values())
    print(f"linear roots: {len(roots)} distinct, {min(roots)}..{max(roots)}; negative "
          f"{sum(c for r, c in roots.items() if r < 0) / len(ps):.1%}, zero {roots[0] / len(ps):.1%}, positive "
          f"{sum(c for r, c in roots.items() if r > 0) / len(ps):.1%}; per root {shares[0]:.1%} to {shares[-1]:.1%}; "
          f"most common {', '.join(f'{r} ({c / len(ps):.1%})' for r, c in roots.most_common(5))}; forms "
          f"{', '.join(f'{k}: {v / len(ps):.0%}' for k, v in forms.most_common())}; "
          f"{len({p.prompt for p in ps})} distinct prompts of {len(ps)}")
    assert share <= 0.10, (top, share)
    assert set(roots) == set(range(-20, 21)), sorted(roots)
    assert all(forms[f] > 0.1 * len(ps) for f in ("a(x+b)=c", "ax+b=cx+d", "ax+b=c", "c=ax+b")), forms
    assert sum(" = -" in p.prompt for p in ps) > 0.1 * len(ps)  # negative constants
    assert sum(" - " in p.prompt for p in ps) > 0.2 * len(ps)
    assert len({p.prompt for p in ps}) > 0.8 * len(ps)
    test = data.sample_domain("linear", 400, "test", 0)
    test_roots = collections.Counter(root(p.prompt) for p in test)
    assert test_roots.most_common(1)[0][1] <= 0.10 * len(test), test_roots.most_common(3)
    # Nor does copying a number of the prompt, with either sign.
    guesses = collections.Counter()
    for p in ps:
        numbers, r = [int(n) for n in re.findall(r"\d+", p.prompt[len(HEAD):])], root(p.prompt)
        guesses.update(name for name, n in (("first", numbers[0]), ("last", numbers[-1]), ("largest", max(numbers)),
                                            ("smallest", min(numbers))) if n == abs(r))
    print("linear: copying a number of the prompt scores at most "
          + ", ".join(f"{k} {c / len(ps):.1%}" for k, c in guesses.items()))
    assert max(guesses.values()) <= 0.10 * len(ps), guesses


def test_sample_speed():
    rng = random.Random(11)
    start = time.perf_counter()
    for _ in range(5000):
        DOMAIN.sample(rng)
    per_sample = (time.perf_counter() - start) / 5000
    print(f"linear: {per_sample * 1e6:.0f} us per sample")
    assert per_sample < 0.0005


# ---------------------------------------------------------------- semantics, by hand

DEEP_PARENS = "(" * MAX_NESTING + "x" + ")" * MAX_NESTING
DEEP_NEG = "- " * MAX_NESTING + "x"

# (left, right, root as Lean's omega finds it); None when the checker must not claim it.
SEMANTICS = [
    ("3 * x + 5", "20", 5), ("x + 5", "0", -5), ("x", "- -5", 5), ("- -x", "3", 3), ("-x", "3", -3),
    ("-(x - 2)", "3", -1), ("2 * -x", "6", -3), ("-2 * x", "6", -3), ("x - -5", "0", -5), ("0", "x", 0),
    ("-x", "0", 0), ("x * 3", "9", 3), ("3 * (x + 1) * 2", "12", 1), ("x + x", "4", 2), ("x - x + x", "7", 7),
    ("2 * x", "2 * x + x - 4", 4), ("x", "2 * x + 3", -3), ("2 - 3", "x", -1), ("x", "3 - 5", -2),
    ("5 - x", "7", -2), ("-(-(-x))", "1", -1), ("x * -1", "5", -5), ("-x * -3", "9", 3), ("(-x) * (-3)", "9", 3),
    ("x + 1000000", "-1000000", -2000000), ("1000 * 1000 * x", "1000000", 1), ("x", "1000000", 1000000),
    ("x", "-1000000", -1000000), ("- - - x", "4", -4), ("2 * (3 * (x - 1))", "-12", -1), ("(x)", "(5)", 5),
    ("x+1", "0", -1), ("2*x-4", "0", 2), ("  x  ", "  1  ", 1), ("-(5)", "x", -5), ("-(-5)", "x", 5),
    ("x - 2 * x", "3", -3), ("4 - 2 * (x - 1)", "0", 3), (DEEP_PARENS, "1", 1), (DEEP_NEG, "1", 1),
    ("- " * (MAX_NESTING - 1) + "x", "1", -1),
    # No spaces, minus next to `*` and `(`, constant subterms whose Nat value
    # would differ (Lean elaborates them as Int because x is on one side).
    ("2*-x", "4", -2), ("-x*-2", "4", 2), ("x-(-2)", "0", -2), ("-(2-5)", "x", 3), ("x*(0-5)", "10", -2),
    ("(2-5)*(3-4)*x", "6", 2), ("x", "-(-(-(1-2)))", 1), ("-1-x", "0", -1), ("3-x", "5-2*x", 2),
    ("x*0+1", "x", 1), ("x", "1000000*(0-1)", -1000000), ("-x-x-x", "3", -1),
    # Not claimed: no root, or many; not linear; over the limits; not this grammar.
    ("2 * x", "3", None), ("x", "x + 1", None), ("x + 1", "1 + x", None), ("0 * x", "5", None),
    ("0 * x", "0", None), ("x - x", "0", None), ("2 * x + 1", "2 * x + 1", None), ("x * x", "4", None),
    ("x * (x + 1)", "2", None), ("(x - x) * x", "0", None), ("x * 0 * x", "0", None), ("x", "1000001", None),
    ("x", "10000000", None), ("x", "1000000 + 1 - 1", None), ("1000 * 1001 * x", "0", None),
    ("1001 * 1000 * x", "0", None), ("x * 1000000 * 2", "0", None), ("x --5", "0", None), ("--x", "3", None),
    ("x", "05", None), ("x", "00", None), ("y", "5", None), ("X", "5", None), ("x ^ 2", "4", None),
    ("x / 2", "3", None), ("x % 2", "1", None), ("2x", "4", None), ("x2", "4", None), ("x(2)", "4", None),
    ("2(x)", "4", None), ("٣ * x", "3", None), ("３ * x", "3", None), ("\U0001d7d1 * x", "3", None),
    ("x − 1", "0", None), ("x\t+ 1", "0", None), ("x + 1", "", None), ("", "1", None), ("x + ", "1", None),
    ("(x", "1", None), ("x)", "1", None), ("()", "1", None), ("x + + 1", "1", None), ("x - - 1", "1", -2 + 2),
    ("(" + DEEP_PARENS + ")", "1", None), ("- " + DEEP_NEG, "1", None), ("x = 1", "1", None),
    ("x.1", "1", None), ("1.5 * x", "3", None), ("+x", "1", None),
]


def test_semantics():
    for left, right, expected in SEMANTICS:
        prompt = eq(left, right)
        assert root(prompt) == expected, (left, right, root(prompt), expected)
        assert DOMAIN.claims(prompt) == (expected is not None), (left, right)
        if expected is not None:
            for proof in PROOFS:
                assert DOMAIN.verify(prompt, f"{expected} := {proof}")
            assert not DOMAIN.verify(prompt, f"{expected + 1} := by omega")


def test_exact_answers_only():
    prompt = eq("x + 5", "0")
    good = ["-5 := by omega", "-5 := by revert h; omega"]
    bad = ["-5", "-5 :=", "-5 := ", "-5 := by", "5 := by omega", "- 5 := by omega", "−5 := by omega",
           "--5 := by omega", "-(5) := by omega", "(-5) := by omega", "-05 := by omega", "-5.0 := by omega",
           "-５ := by omega", "-٥ := by omega", " -5 := by omega", "-5 := by omega ", "-5  := by omega",
           "-5 :=  by omega", "-5:= by omega", "-5 :=by omega", "-5 := by  omega", "-5 := by\tomega",
           "-5 := by omega\n", "-5 :=\n  by omega", "-5 = by omega", "-5 := by simp", "-5 := by linarith",
           "-5 := by omega;", "-5 := by revert h ; omega", "-5 := by revert h;omega", "-5 := by revert h; omega;",
           "-5 := by revert h\n  omega", "-5 := by intro; omega", "-5 := by decide", "-5 := rfl", "-5 := sorry",
           "-5 := by sorry", "-5 := by Omega", "-5 := by omega\U0001f600", "-4 := by omega", "-6 := by omega",
           "-5 := by exact by omega", None, -5, ""]
    for a in good:
        assert DOMAIN.verify(prompt, a), a
    for a in bad:
        assert not DOMAIN.verify(prompt, a), a
    # A root of zero is written 0, never -0 (JavaScript's -0 prints as 0 too).
    assert DOMAIN.verify(eq("-x", "0"), "0 := by omega") and not DOMAIN.verify(eq("-x", "0"), "-0 := by omega")


def test_checker_never_raises():
    weird_prompts = [None, 3, "", HEAD, TAIL, HEAD + TAIL, HEAD + "=" + TAIL, HEAD + " = " + TAIL,
                     eq("x", "1") + "\n", eq("x", "1\n"), eq("(" * 1000 + "x" + ")" * 1000, "1"),
                     eq("- " * 500 + "x", "1"), eq("x", "1 + " * 40 + "1"), eq("\ud800", "1"), eq("x", "1\udfff"),
                     eq("9" * 300, "x"), eq("1000000 * " * 30 + "x", "1"), eq("x" + " * x" * 30, "1"),
                     HEAD + "x = 1 = 1" + TAIL, HEAD + "x == 1" + TAIL]
    weird_answers = [None, 7, "", "\ud800", "5 := by omega\udfff", "5 := by omega", "x" * 500, "𝔸" * 250]
    for p in weird_prompts:
        assert DOMAIN.claims(p) in (True, False)
        if isinstance(p, str):
            assert isinstance(DOMAIN.split_key(p), str)
        for a in weird_answers:
            assert DOMAIN.verify(p, a) in (True, False)
    pad = 200 - len(eq("x", "1"))
    assert DOMAIN.claims(eq(" " * pad + "x", "1"))
    assert not DOMAIN.claims(eq(" " * (pad + 1) + "x", "1"))


# ---------------------------------------------------------------- split keys

def spell_side(t, rng, extra=0.15) -> str:
    """t printed with the parentheses Lean needs, redundant ones added at
    random, and random spacing (never `--`, which starts a comment)."""
    def toks(t):
        if isinstance(t, int):
            out = [str(t)]
        elif t == "x":
            out = ["x"]
        elif t[0] == "neg":
            inner = toks(t[1])
            if isinstance(t[1], tuple) and t[1][0] != "neg":
                inner = ["("] + inner + [")"]
            out = ["-"] + inner
        else:
            op, a, b = t
            left, right = toks(a), toks(b)
            if isinstance(a, tuple) and a[0] != "neg" and linear._PREC[a[0]] < linear._PREC[op]:
                left = ["("] + left + [")"]
            if isinstance(b, tuple) and b[0] != "neg" and linear._PREC[b[0]] <= linear._PREC[op]:
                right = ["("] + right + [")"]
            out = left + [op] + right
        while rng.random() < extra:
            out = ["("] + out + [")"]
        return out
    parts = toks(t)
    text = parts[0]
    for prev, tok in zip(parts, parts[1:]):
        text += (" " if prev == tok == "-" else rng.choice(("", " ", " ", "  "))) + tok
    return text


def spell(left, right, rng, extra=0.15) -> str:
    return f"{HEAD}{rng.choice(('', ' '))}{spell_side(left, rng, extra)}{rng.choice(('', ' ', '  '))}={rng.choice(('', ' ', '  '))}" \
           f"{spell_side(right, rng, extra)}{rng.choice(('', ' '))}{TAIL}"


def commute(t, rng):
    """t with the operands of + and * swapped and regrouped at random: the
    same side, the same problem."""
    if isinstance(t, int) or t == "x":
        return t
    if t[0] == "neg":
        return ("neg", commute(t[1], rng))
    op, a, b = t
    a, b = commute(a, rng), commute(b, rng)
    if op == "-":
        return (op, a, b)
    if rng.random() < 0.5:
        a, b = b, a
    if rng.random() < 0.5 and isinstance(a, tuple) and a[0] == op:  # (u + v) + b -> u + (v + b)
        return (op, a[1], (op, a[2], b))
    return (op, a, b)


def test_split_key_invariance():
    """Redundant parentheses, spacing, the sides swapped, the operands of + and
    * commuted or regrouped: the same equation, the same key."""
    rng = random.Random(2)
    for p in problems():
        key = DOMAIN.split_key(p.prompt)
        left, right = sides(p.prompt)
        for i in range(8):
            l, r = (commute(left, rng), commute(right, rng)) if i % 2 else (left, right)
            other = spell(l, r, rng, extra=0.3) if i % 4 < 2 else spell(r, l, rng, extra=0.3)
            if not DOMAIN.claims(other):  # too long or too deep
                assert len(other) > 200 or other.count("(") > MAX_NESTING, other
                continue
            assert DOMAIN.split_key(other) == key, (p.prompt, other)
            assert root(other) == root(p.prompt)
            assert data.split_of(other) == data.split_of(p.prompt)
    assert DOMAIN.split_key(eq("((3) * x)", "(-(6))")) == eq("-6", "3 * x")
    assert DOMAIN.split_key(eq("-(-x)", "((2 * x) + 1) + 2")) == eq("- -x", "1 + 2 + 2 * x")
    assert DOMAIN.split_key(eq("x - (1 - 2)", "(-3) * x + 5")) == eq("-3 * x + 5", "x - (1 - 2)")
    assert DOMAIN.split_key(eq("(-(3 * x))", "3")) == eq("-(3 * x)", "3")
    assert DOMAIN.split_key(eq("x", "05")) == eq("x", "05")  # unclaimed: the prompt itself


def test_swapped_twins_share_a_split():
    """Regression: keys used to keep the sides in the order written, and the
    generator writes both `3 * x + 5 = 20` (form ax+b=c) and `20 = 3 * x + 5`
    (form c=ax+b, the same equation swapped). Of the problems a 40,000-draw
    stream (pretraining's share per domain) put in the test split, 15% had
    the swapped equation, with its answer, in the training split."""
    for a, b in ((("3 * x + 5", "20"), ("20", "3 * x + 5")), (("2 * x + 3", "5 * x - 6"), ("5 * x - 6", "2 * x + 3")),
                 (("x * 3 + 5", "20"), ("5 + 3 * x", "20")), (("4 * (x - 2)", "8"), ("8", "(x - 2) * 4"))):
        assert DOMAIN.split_key(eq(*a)) == DOMAIN.split_key(eq(*b)), (a, b)
    rng, train, test = data._domain_rng(0, "linear"), set(), []
    for _ in range(20_000):
        p = DOMAIN.sample(rng)
        split = data.split_of_key(DOMAIN.split_key(p.prompt))
        if split == "train":
            train.add(p.prompt)
        elif split == "test":
            test.append(p.prompt)
    twins = [q for q in test if eq(*q[len(HEAD):-len(TAIL)].split(" = ")[::-1]) in train]
    assert len(test) > 800 and not twins, twins[:5]


def test_distinct_problems_have_distinct_keys():
    """Prompts share a key only if they are the same equation up to the order
    of its sides and the order and grouping of the operands of + and *; then
    they have one root."""
    ps = problems(5000, 3)
    rng = random.Random(6)
    prompts = [p.prompt for p in ps] + [spell(_random_side(rng, rng.randint(0, 5)), _random_side(rng, rng.randint(0, 5)),
                                              rng, extra=0) for _ in range(20_000)]
    by_key = {}
    for p in prompts:
        if DOMAIN.claims(p):
            by_key.setdefault(DOMAIN.split_key(p), []).append(p)
    for key, same in by_key.items():
        assert len({frozenset(linear.show(linear.canonical(s)) for s in sides(p)) for p in same}) == 1, same
        assert len({root(p) for p in same}) == 1, same
    assert sum(len(same) > 1 for same in by_key.values()) > 10  # the check above had something to check
    # The same root does not make the same problem: `2 * x = 4` and
    # `x + x = 4` are different equations; so are both sides negated, or
    # moved across.
    keys = {DOMAIN.split_key(eq(a, b)) for a, b in (("2 * x", "4"), ("x * 2", "4"), ("x + x", "4"), ("4", "2 * x"),
                                                     ("(2 * x)", "4"), ("2 * x", "(4)"), ("x", "2"), ("-2 * x", "-4"),
                                                     ("2 * x - 4", "0"), ("x - 2", "0"), ("2 - x", "0"))}
    assert len(keys) == 7


# ---------------------------------------------------------------- differential against web/checker.js

ARABIC, FULLWIDTH, DEVANAGARI = "٠١٢٣٤٥٦٧٨٩", "０１２３４５６７８９", "०१२३४५६७८९"
BOLD = "".join(chr(0x1D7CE + i) for i in range(10))  # astral digits, two UTF-16 code units each
ASTRAL = ("𝔸", "\U0001f600", "\U0001d7d5", "\U00010000")
WRONG_PROOFS = ("by simp", "by linarith", "by omega;", "by  omega", "by\tomega", "by revert h ; omega",
                "by revert h;omega", "by revert h; omega;", "by revert h\n  omega", "by intro; omega", "by decide",
                "rfl", "by rfl", "by omega\n", "by\n  omega", "by Omega", "by omega!", "by exact h", "by subst h; rfl",
                "by revert h; simp", "by revert x; omega", "by revert h; omega; omega", "by clear h; omega",
                "by omega <;> rfl", "", "by", "omega", "by revert h", "by revert h;", "by omeg", "by revert  h; omega",
                "by revert h;  omega", "sorry", "by sorry", "by norm_num", "by revert h; norm_num")


def _digits(s: str, script: str, rng, every: bool) -> str:
    idx = [i for i, c in enumerate(s) if c.isdigit() and c < "\x80"]
    if not idx:
        return s
    chosen = idx if every else [rng.choice(idx)]
    return "".join(script[int(c)] if i in chosen else c for i, c in enumerate(s))


def mutate_answer(answer: str, r: int, rng: random.Random) -> str:
    """A near miss of a correct answer: the root slightly wrong, its sign
    wrong or misspelled, or the proof misspelled."""
    head, proof = answer.split(" := ", 1)
    kind = rng.randrange(17)
    if kind == 0:  # one digit changed
        idx = [i for i, c in enumerate(head) if c.isdigit()] or [0]
        i = rng.choice(idx)
        return head[:i] + rng.choice([d for d in "0123456789" if d != head[i:i + 1]]) + head[i + 1:] + " := " + proof
    if kind == 1:
        return f"{r + rng.choice((-10, -2, -1, 1, 2, 10, 1000000))} := {proof}"
    if kind == 2:  # the sign: flipped, doubled, spaced, parenthesised, a Unicode minus
        a = abs(r)
        return rng.choice((f"{-r}", f"-{r}", f"- {r}", f"- -{a}", f"--{a}", f"−{a}", f"+{r}", f"(-{a})", f"-({a})",
                           f"-0{a}", f"(0 - {a})", f"-{a}" if r > 0 else f"{a}")) + " := " + proof
    if kind == 3:
        return f"{head} := {rng.choice(WRONG_PROOFS)}"
    if kind == 4:
        i = rng.randrange(len(answer) + 1)
        return answer[:i] + rng.choice((" ", "  ", "\t", "\n", "\u00a0")) + answer[i:]
    if kind == 5:
        return answer[:rng.randrange(len(answer))]
    if kind == 6:
        return _digits(answer, rng.choice((ARABIC, FULLWIDTH, DEVANAGARI, BOLD)), rng, rng.random() < 0.5)
    if kind == 7:
        i = rng.randrange(len(answer) + 1)
        return answer[:i] + rng.choice(ASTRAL) + answer[i:]
    if kind == 8:  # a leading zero
        return (f"-0{-r}" if r < 0 else f"0{r}") + " := " + proof
    if kind == 9:
        return rng.choice((f"{head}.0", f"({head})", f"{head}e0", f"{head}_0", f"{head}x", f"{head} * 1",
                           f"{head} + 0")) + " := " + proof
    if kind == 10:
        i = rng.randrange(len(answer))
        return answer[:i] + answer[i + 1:]
    if kind == 11:
        return head + rng.choice((" = ", " :=", ":= ", ":=", " : = ", " ≔ ", " :== ")) + proof
    if kind == 12:
        return answer + rng.choice((" ", "\n", ";", " " + answer, "\n" + answer, "; omega"))
    if kind == 13:
        return f"{head} := {rng.choice(PROOFS)}"  # another proof form: still correct
    if kind == 14:
        return answer + " " * rng.choice((370, 400))  # over the length limit
    if kind == 15:
        i, j = sorted(rng.sample(range(len(answer)), 2))
        return answer[:i] + answer[j] + answer[i + 1:j] + answer[i] + answer[j + 1:]
    return answer.replace(" ", "")


def _const(c: int):
    return c if c >= 0 else ("neg", -c)


def _paths(t, path=()):
    """Paths to every subtree of t, t itself first."""
    yield path, t
    if isinstance(t, tuple):
        for i, u in enumerate(t[1:], 1):
            yield from _paths(u, path + (i,))


def _replace(t, path, new):
    if not path:
        return new
    i = path[0]
    return t[:i] + (_replace(t[i], path[1:], new),) + t[i + 1:]


def _random_side(rng: random.Random, n_ops: int):
    """A random side, linear or not, with constants of every size."""
    if n_ops == 0:
        return rng.choice(("x", "x", rng.randint(0, 9), rng.randint(0, 100), rng.choice((999999, 1000000))))
    if rng.random() < 0.15:
        return ("neg", _random_side(rng, n_ops - 1))
    k = rng.randint(0, n_ops - 1)
    return (rng.choice("+-*"), _random_side(rng, k), _random_side(rng, n_ops - 1 - k))


_SOUP = list("0123456789") * 2 + list("x+-*()") * 3 + [" "] * 8 + ["=", "00", "1000000", "--", "- -", "y", "^",
                                                                    "/", "٣", "３", "\U0001d7d1", "−", "\t"]


def soup(rng: random.Random) -> str:
    return "".join(rng.choice(_SOUP) for _ in range(rng.randint(1, 30)))


def mutate_prompt(prompt: str, rng: random.Random) -> str:
    left, right = sides(prompt)
    body = prompt[len(HEAD):-len(TAIL)]
    kind = rng.randrange(19)
    if kind == 0:  # respelled
        return spell(left, right, rng, extra=rng.choice((0.1, 0.3, 0.6)))
    if kind == 1:
        return spell(right, left, rng, extra=0.1)
    if kind == 2:  # the same term on both sides
        t = rng.choice(("x", 3, ("*", 2, "x"), ("neg", "x"), ("*", 1000, "x"), 999999))
        op = rng.choice("+-")
        return spell((op, left, t), (op, right, t), rng, extra=0.1)
    if kind == 3:
        return spell(("neg", left), ("neg", right), rng, extra=0.1)
    if kind == 4:  # both sides scaled: the root stays, the coefficients may pass the limit
        c = rng.choice((2, 3, ("neg", 1), 0, 1000, 100000, 1000000))
        return spell(("*", c, left), ("*", c, right), rng, extra=0.1)
    if kind in (5, 6, 7, 8):
        side, other = (left, right) if rng.random() < 0.5 else (right, left)
        paths = list(_paths(side))
        if kind == 5:  # a constant changed
            ints = [p for p, u in paths if isinstance(u, int)] or [()]
            side = _replace(side, rng.choice(ints), rng.choice((0, 1, 2, 7, 13, 30, rng.randint(0, 99))))
        elif kind == 6:  # a sign flipped
            path, u = rng.choice(paths)
            side = _replace(side, path, u[1] if isinstance(u, tuple) and u[0] == "neg" else ("neg", u))
        elif kind == 7:  # an operator changed (x * x among the results)
            ops = [(p, u) for p, u in paths if isinstance(u, tuple) and u[0] != "neg"]
            if ops:
                path, u = rng.choice(ops)
                side = _replace(side, path, (rng.choice("+-*"),) + u[1:])
        else:  # x where a constant was, or a side multiplied by x
            ints = [p for p, u in paths if isinstance(u, int)]
            side = _replace(side, rng.choice(ints), "x") if ints and rng.random() < 0.5 else ("*", side, "x")
        return spell(side, other, rng, extra=0.1) if rng.random() < 0.5 else spell(other, side, rng, extra=0.1)
    if kind == 9:  # numbers at and above the limits, leading zeros
        big = rng.choice(("999999", "1000000", "1000001", "9999999", "10000000", "05", "00", "007", "0"))
        if rng.random() < 0.5:
            before, after = rng.choice((f"{big} + ", f"{big} * ", "")), rng.choice((f" + {big}", f" - {big}", f" * {big}", ""))
            return f"{HEAD}{before}{body}{after}{TAIL}"
        return re.sub(r"\d+", lambda m: big if rng.random() < 0.4 else m.group(), prompt)
    if kind == 10:  # no root or every integer a root
        c = rng.choice((0, 0, 1, 2, 5))
        if rng.random() < 0.5:
            return spell(("+", left, c), left, rng, extra=0.1)
        return spell(("*", 2, left), ("+", ("*", 2, right), rng.choice((1, 3))), rng, extra=0.1)
    if kind == 11:  # `--` starts a comment; `- -` is double negation
        text = spell(left, right, rng, extra=0.1)
        i = rng.randrange(len(HEAD), len(text) - len(TAIL) + 1)
        return text[:i] + rng.choice(("--", "- -", "-- ", " - - ", "--x", "- -x")) + text[i:]
    if kind == 12:  # nesting at the limit: parentheses or prefix minus
        d = rng.choice((MAX_NESTING - 1, MAX_NESTING, MAX_NESTING + 1))
        wrap = rng.choice((lambda s: "(" * d + s + ")" * d, lambda s: "- " * d + s, lambda s: "-(" * d + s + ")" * d))
        return eq(wrap(spell_side(left, rng, 0)), spell_side(right, rng, 0))
    if kind == 13:  # a character inserted, deleted or swapped
        i = rng.randrange(len(body) + 1)
        op = rng.randrange(3)
        if op == 0:
            body = body[:i] + rng.choice(_SOUP) + body[i:]
        elif op == 1 and len(body) > 1:
            body = body[:i] + body[i + 1:]
        elif len(body) > 2:
            i = min(i, len(body) - 2)
            body = body[:i] + body[i + 1] + body[i] + body[i + 2:]
        return f"{HEAD}{body}{TAIL}"
    if kind == 14:  # the header changed
        return rng.choice(("example (x : Nat) (h : ", "example (y : Int) (h : ", "example (x : ℤ) (h : ",
                           "example (x : Int) (h: ", "example (x : Int) (h  : ", "example (x : Int)(h : ",
                           "example (x : Int) (h : (", HEAD + HEAD)) + body \
            + rng.choice((TAIL, ") : x = ", ") : y =", ") : x =\n", "): x =", ") : x :=", ") : x = 0", "))" + TAIL))
    if kind == 15:  # spaces up to and past the length limit
        pad = 200 - len(prompt) + rng.choice((-1, 0, 0, 1))
        return f"{HEAD}{' ' * max(pad, 0)}{body}{TAIL}"
    if kind == 16:
        if rng.random() < 0.5:
            return f"{HEAD}{_digits(body, rng.choice((ARABIC, FULLWIDTH, DEVANAGARI, BOLD)), rng, False)}{TAIL}"
        i = rng.randrange(len(body) + 1)
        return f"{HEAD}{body[:i]}{rng.choice(ASTRAL)}{body[i:]}{TAIL}"
    if kind == 17:
        if rng.random() < 0.5:
            return f"{HEAD}{soup(rng)}{TAIL}"
        return spell(_random_side(rng, rng.randint(0, 5)), _random_side(rng, rng.randint(0, 5)), rng, extra=0.1)
    # two problems added together
    l2, r2 = sides(rng.choice(problems()).prompt)
    return spell(("+", left, l2), ("+", right, r2), rng, extra=0.1)


@functools.lru_cache(maxsize=None)
def differential_cases() -> tuple:
    """(prompt, answer, origin): references and every proof form, near misses
    of them, mutated prompts with the root Python computes for them (so a
    disagreement on a root shows), and random strings."""
    rng = random.Random(0)
    cases = []
    ps = problems()
    for p in ps:
        r = root(p.prompt)
        cases += [(p.prompt, f"{r} := {proof}", "valid") for proof in PROOFS]
        for _ in range(16):
            a = f"{r} := {rng.choice(PROOFS)}"
            for _ in range(rng.choice((1, 1, 1, 2))):
                if " := " not in a:
                    break
                a = mutate_answer(a, r, rng)
            cases.append((p.prompt, a, "answer mutant"))
        cases.append((p.prompt, rng.choice(ps).reference, "other answer"))
        cases.append((p.prompt, soup(rng), "random answer"))
    for p in ps:
        for _ in range(12):
            q = mutate_prompt(p.prompt, rng)
            r = root(q)
            if r is None:
                cases.append((q, p.reference, "prompt mutant"))
            else:
                cases.append((q, f"{r} := {rng.choice(PROOFS)}", "prompt mutant"))
                cases.append((q, f"{rng.choice((r + 1, r - 1, -r if r else 1))} := by omega", "prompt mutant"))
    for left, right, expected in SEMANTICS:
        for proof in PROOFS:
            cases.append((eq(left, right), f"{expected if expected is not None else 0} := {proof}", "by hand"))
    # Shapes and spellings the generator never makes: up to six operators,
    # prefix minus anywhere, products of constant subterms, sides without x,
    # often no spaces at all. The ones the checker accepts go to Lean too.
    rng = random.Random(1)
    for _ in range(600):
        texts = []
        for _ in range(2):
            text = spell_side(_linear_side(rng, rng.randint(0, 6)), rng, extra=rng.choice((0, 0.3)))
            if rng.random() < 0.5:
                text = text.replace(" ", "")
                while "--" in text:
                    text = text.replace("--", "- -")
            texts.append(text)
        q = eq(*texts)
        r = root(q)
        cases.append((q, f"{0 if r is None else r} := {rng.choice(PROOFS)}", "grammar"))
    return tuple(cases)


def _linear_side(rng: random.Random, n_ops: int, x: bool = True):
    """A random side that stays linear: one factor of every product is free of x."""
    if n_ops == 0:
        return rng.choice(("x", "x", 0, 1, 2, 7, rng.randint(0, 30), 1000, 999999, 1000000)) if x \
            else rng.choice((0, 1, 2, 7, rng.randint(0, 30), 1000))
    if rng.random() < 0.25:
        return ("neg", _linear_side(rng, n_ops - 1, x))
    k = rng.randint(0, n_ops - 1)
    op = rng.choice("+-*")
    if op == "*":
        if rng.random() < 0.5:
            return ("*", _linear_side(rng, k, x), _linear_side(rng, n_ops - 1 - k, False))
        return ("*", _linear_side(rng, k, False), _linear_side(rng, n_ops - 1 - k, x))
    return (op, _linear_side(rng, k, x), _linear_side(rng, n_ops - 1 - k, x))


def _python_verdict(prompt, answer):
    return DOMAIN.verify(prompt, answer) if DOMAIN.claims(prompt) else None


def test_differential_against_javascript(node):
    cases = differential_cases()
    assert len(cases) >= 20_000
    js = js_verify([(p, a) for p, a, _ in cases])
    py = [_python_verdict(p, a) for p, a, _ in cases]
    mismatches = [(c, j, q) for c, j, q in zip(cases, js, py) if j != q]
    counts = {}
    for (_, _, origin), verdict in zip(cases, py):
        counts.setdefault(origin, [0, 0, 0])[{True: 0, False: 1, None: 2}[verdict]] += 1
    print(f"linear differential: {len(cases)} cases, {len(mismatches)} disagreements; "
          + "; ".join(f"{o}: {a} accepted, {r} rejected, {u} unclaimed" for o, (a, r, u) in counts.items()))
    assert not mismatches, mismatches[:10]
    assert all(v is True for (_, _, origin), v in zip(cases, py) if origin == "valid")
    for origin in ("answer mutant", "prompt mutant"):
        accepted, rejected, unclaimed = counts[origin]
        assert accepted > 500 and rejected > 500, (origin, counts[origin])
    assert counts["prompt mutant"][2] > 1000
    assert counts["grammar"][0] > 100 and counts["grammar"][2] > 50, counts["grammar"]


def test_every_short_body_against_javascript(node):
    """Every body of up to five symbols over the grammar's characters (`=`
    included, so none, one or several), and every side of up to three symbols
    against a few fixed sides, with the root Python computes: the corner of
    the input space (lone operators, `- -`, empty sides) that random mutants of
    real problems reach only by chance."""
    symbols = list("x02+-*() =")
    bodies = ["".join(s) for n in range(1, 6) for s in itertools.product(symbols, repeat=n)]
    short = ["".join(s) for n in range(0, 4) for s in itertools.product(symbols[:-1], repeat=n)]
    bodies += [b for s in short for other in ("1", "x", "-x", "2 - 5") for b in (f"{s}={other}", f"{other}={s}")]
    pairs = []
    for b in bodies:
        r = root(HEAD + b + TAIL)
        pairs.append((HEAD + b + TAIL, f"{0 if r is None else r} := by omega"))
    js = js_verify(pairs)
    py = [_python_verdict(p, a) for p, a in pairs]
    mismatches = [(c, j, q) for c, j, q in zip(pairs, js, py) if j != q]
    claimed = sum(v is not None for v in py)
    print(f"linear short bodies: {len(pairs)} prompts, {claimed} claimed, {len(mismatches)} disagreements")
    assert not mismatches, mismatches[:10]
    assert claimed > 500


def js_verify_escaped(node, pairs) -> list:
    """js_verify for strings that UTF-8 cannot carry (lone surrogates, which a
    JavaScript string can hold): the JSON goes to Node ASCII-escaped."""
    data = "".join(json.dumps([p, a]) + "\n" for p, a in pairs)
    out = subprocess.run([node, os.path.join(ROOT, "tests", "js_bridge.js")], input=data, capture_output=True,
                         text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line]


def test_lone_surrogates_against_javascript(node):
    """A lone surrogate is never in the grammar, so every case is rejected or
    unclaimed; the point is that Python neither raises nor disagrees."""
    rng = random.Random(4)
    cases = []
    for p in problems()[:100]:
        body = p.prompt[len(HEAD):-len(TAIL)]
        for s in ("\ud800", "\udfff", "\udbff\udbff"):
            i = rng.randrange(len(body) + 1)
            cases += [(p.prompt, p.reference + s), (p.prompt, s + p.reference), (p.prompt + s, p.reference),
                      (f"{HEAD}{body[:i]}{s}{body[i:]}{TAIL}", p.reference),
                      (f"{HEAD}{' ' * (200 - len(p.prompt))}{body}{TAIL}{s}", p.reference)]
    js = js_verify_escaped(node, cases)
    py = [_python_verdict(p, a) for p, a in cases]
    assert len(js) == len(cases) and js == py, [(c, j, q) for c, j, q in zip(cases, js, py) if j != q][:10]


# ---------------------------------------------------------------- against Lean

def _stated_value(answer: str):
    """The integer an answer states, read as Lean reads it (prefix minus,
    `+ 0`, parentheses, leading zeros, hexadecimal, digit separators), or
    None. Only digits, `+ - * ( )` and spaces reach eval, so it computes
    integer arithmetic and nothing else."""
    s = answer.split(":=", 1)[0].replace("_", "").strip()
    s = re.sub(r"0[xX][0-9a-fA-F]+", lambda m: str(int(m.group(), 16)), s)
    s = re.sub(r"\b0+(?=[0-9])", "", s)
    if not s or "--" in s or not re.fullmatch(r"[-+*() 0-9]+", s):
        return None
    try:
        return eval(s, {"__builtins__": {}})
    except (SyntaxError, TypeError):
        return None


def test_soundness_against_lean(lean):
    """Both proof forms of 500 references, the semantic edge cases (negative
    roots, `- -5`, Int subtraction), and every mutant the checker accepts
    compile. Rejected answers that Lean accepts must state the checker's root:
    the checker may refuse a respelling (`(-5)`, `-05`), never a root Lean
    proves different."""
    valid = [f"{p.prompt} {root(p.prompt)} := {proof}" for p in problems()[:500] for proof in PROOFS]
    valid += [f"{eq(left, right)} {r} := {proof}" for left, right, r in SEMANTICS if r is not None for proof in PROOFS]
    accepted, rejected = set(), {}
    for prompt, answer, origin in differential_cases():
        if DOMAIN.claims(prompt):
            if DOMAIN.verify(prompt, answer):
                accepted.add(prompt + " " + answer)
            else:
                rejected[prompt + " " + answer] = (prompt, answer)
    accepted -= set(valid)
    sample = sorted(r for r in rejected if lean_oracle.is_plain_example(r))
    sample = random.Random(0).sample(sample, min(len(sample), 800))
    core = list(dict.fromkeys(valid)) + sorted(accepted)
    verdicts = dict(zip(core, lean_oracle.check_in_chunks(core, lean, chunk=2000)))
    # A garbage answer can make Lean report an error without a position (after
    # the lexer error in `:= by 0evert h; omega` comes "unsolved goals" at line
    # 1, column 0), which check_declarations blames on the first declaration
    # of the batch. So the rejected answers go in a batch of their own, behind
    # a declaration that compiles.
    lean_accepts = [d for d, ok in zip(sample, lean_oracle.check_declarations([core[0]] + sample, lean)[1:]) if ok]
    failed_valid = [d for d in dict.fromkeys(valid) if not verdicts[d]]
    failed_accepted = [d for d in sorted(accepted) if not verdicts[d]]
    print(f"linear vs Lean: {len(set(valid))} references, proof forms and edge cases, {len(accepted)} accepted "
          f"mutants; of {len(sample)} rejected answers Lean accepts {len(lean_accepts)}")
    for d in lean_accepts[:10]:
        print("   checker rejects, Lean accepts:", repr(d))
    assert not failed_valid, failed_valid[:10]
    assert not failed_accepted, failed_accepted[:10]
    wrong = [d for d in lean_accepts if _stated_value(rejected[d][1]) != root(rejected[d][0])]
    assert not wrong, wrong[:10]
    assert len(lean_accepts) < len(sample) - 400
