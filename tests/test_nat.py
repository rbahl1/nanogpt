"""The nat domain: its generator, Lean's Nat semantics, split keys, its checker
against web/checker.js (the specification: they must agree on every case, here
references, every proof form and many kinds of mutated answers and prompts),
and against Lean 4 (soundness: every answer the checker accepts compiles)."""

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
from formal import nat
from formal.nat import DOMAIN, HEAD, MAX_NESTING, PROOFS, TAIL
from js_checker import js_verify
import lean_oracle
from tokenizer import Tokenizer

N_PROBLEMS = 600
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@functools.lru_cache(maxsize=None)
def problems(n=N_PROBLEMS, seed=0) -> tuple:
    rng = random.Random(seed)
    return tuple(DOMAIN.sample(rng) for _ in range(n))


def value(prompt):
    return nat.value_of(prompt)


def tree(prompt):
    return nat._parse(prompt)[1]


# ---------------------------------------------------------------- generator

def test_references_verify_and_fit():
    tok = Tokenizer()
    for p in problems():
        assert p.domain == "nat" and DOMAIN.claims(p.prompt), p
        assert DOMAIN.verify(p.prompt, p.reference), p
        assert p.reference == f"{value(p.prompt)} := by decide", p
        for proof in PROOFS:
            assert DOMAIN.verify(p.prompt, f"{value(p.prompt)} := {proof}"), (p, proof)
        ids, _ = tok.example_ids(p.prompt, p.reference)
        assert len(ids) <= 64, p
        # The key is itself a prompt of the same problem, and its own key.
        key = DOMAIN.split_key(p.prompt)
        assert value(key) == value(p.prompt) and DOMAIN.split_key(key) == key, (p, key)
        assert formal.domain_for(p.prompt) is DOMAIN  # no other domain claims it


def test_documents_verify():
    rng = random.Random(5)
    proofs = collections.Counter()
    for p in problems():
        for _ in range(4):
            doc = DOMAIN.document(p, rng)
            assert doc.startswith(p.prompt + " "), doc
            answer = doc[len(p.prompt) + 1:]
            assert DOMAIN.verify(p.prompt, answer), doc
            proofs[answer.split(" := ", 1)[1]] += 1
    assert set(proofs) == set(PROOFS), proofs  # every proof form appears in pretraining


_SAMPLE_IN_SUBPROCESS = """
import json, random
from formal.nat import DOMAIN
rng = random.Random(7)
ps = [DOMAIN.sample(rng) for _ in range(50)]
print(json.dumps([[p.prompt, p.reference, DOMAIN.document(p, rng)] for p in ps]))
"""


def test_generator_is_deterministic():
    assert list(problems(50, 7)) == list(problems.__wrapped__(50, 7))
    assert list(problems(50, 7)) != list(problems(50, 8))
    rng1, rng2 = random.Random(3), random.Random(3)
    assert [DOMAIN.document(p, rng1) for p in problems()] == [DOMAIN.document(p, rng2) for p in problems()]
    # Across processes too, whatever the string hash seed: data.py relies on it.
    rng = random.Random(7)
    here = [[p.prompt, p.reference, DOMAIN.document(p, rng)] for p in [DOMAIN.sample(rng) for _ in range(50)]]
    for hash_seed in ("1", "2"):
        out = subprocess.run([sys.executable, "-c", _SAMPLE_IN_SUBPROCESS], cwd=ROOT, capture_output=True, text=True,
                             check=True, env={**os.environ, "PYTHONHASHSEED": hash_seed})
        assert json.loads(out.stdout) == here


def _ops(t) -> int:
    return 0 if isinstance(t, int) else 1 + _ops(t[1]) + _ops(t[2])


def test_distribution():
    """No constant answer scores well: the most common value covers at most
    10% of problems (truncated subtraction makes 0 the value of ~30% of random
    expressions, so the generator has to thin them out)."""
    ps = problems(5000, 1)
    values = collections.Counter(value(p.prompt) for p in ps)
    top, top_count = values.most_common(1)[0]
    share = top_count / len(ps)
    buckets = collections.Counter()
    for v, c in values.items():
        buckets["0" if v == 0 else "1" if v == 1 else "2-9" if v < 10 else "10-99" if v < 100
                else "100-999" if v < 1000 else "1000+"] += c
    ops = collections.Counter(_ops(tree(p.prompt)) for p in ps)
    print(f"nat values: {', '.join(f'{k}: {buckets[k] / len(ps):.1%}' for k in ('0', '1', '2-9', '10-99', '100-999', '1000+'))}; "
          f"most common {', '.join(f'{v} ({c / len(ps):.1%})' for v, c in values.most_common(6))}; "
          f"{len(values)} distinct values, max {max(values)}; operators "
          f"{', '.join(f'{k}: {ops[k] / len(ps):.0%}' for k in sorted(ops))}; "
          f"{len({p.prompt for p in ps})} distinct prompts of {len(ps)}")
    assert share <= 0.10, (top, share)
    assert len(values) > 200 and buckets["1000+"] > 0 and buckets["100-999"] > 0.05 * len(ps), buckets
    for op in "+-*/%":
        assert sum(op in p.prompt for p in ps) > 0.05 * len(ps), op
    assert sum("(" in p.prompt for p in ps) > 0.1 * len(ps)
    # The edge cases of Lean's semantics occur.
    assert sum(" / 0" in p.prompt or " % 0" in p.prompt for p in ps) > 20
    # Nor in a split: the held-out problems are no easier to guess.
    test = data.sample_domain("nat", 400, "test", 0)
    test_values = collections.Counter(value(p.prompt) for p in test)
    assert test_values.most_common(1)[0][1] <= 0.10 * len(test), test_values.most_common(3)
    # Nor does copying a number of the prompt score well. Before the generator
    # thinned them, the first number was the value of 17% of problems
    # (`5 % 7`, `5 + 0`, `5 * 1`, `5 - 0`): a free 17% for "copy the first
    # number".
    guesses = collections.Counter()
    for p in ps:
        numbers, v = [int(n) for n in re.findall(r"\d+", p.prompt)], value(p.prompt)
        guesses.update(name for name, n in (("first", numbers[0]), ("second", numbers[1]), ("last", numbers[-1]),
                                            ("largest", max(numbers)), ("smallest", min(numbers))) if n == v)
    print("nat: copying a number of the prompt scores " + ", ".join(f"{k} {c / len(ps):.1%}" for k, c in guesses.items()))
    assert max(guesses.values()) <= 0.10 * len(ps), guesses


def test_sample_speed():
    rng = random.Random(11)
    start = time.perf_counter()
    for _ in range(5000):
        DOMAIN.sample(rng)
    per_sample = (time.perf_counter() - start) / 5000
    print(f"nat: {per_sample * 1e6:.0f} us per sample")
    assert per_sample < 0.0005


# ---------------------------------------------------------------- Lean's semantics, by hand

# (expression, value as Lean computes it); None when the checker must not claim it.
SEMANTICS = [
    ("3 + 4", 7), ("2 + 3 * 4", 14), ("(2 + 3) * 4", 20), ("10 - 3 - 2", 5), ("10 - (3 - 2)", 9),
    ("100 / 10 / 5", 2), ("100 / (10 / 5)", 50), ("7 % 3 * 2", 2), ("7 % (3 * 2)", 1), ("2 * 7 % 4", 2),
    ("3 - 7", 0), ("3 - 7 + 2", 2), ("2 - 3 * 4", 0), ("(3 - 7) * 5", 0), ("0 - 0", 0),
    ("7 / 2", 3), ("1 / 2", 0), ("9 % 4", 1), ("4 % 9", 4),
    ("7 / 0", 0), ("7 % 0", 7), ("0 / 0", 0), ("0 % 0", 0), ("5 / (2 - 3)", 0), ("5 % (2 - 3)", 5),
    ("12 % (3 - 3) / 5", 2), ("10000", 10000), ("0", 0), ("100 * 100", 10000), ("9999 + 1", 10000),
    ("10000 / 0", 0), ("10000 % 0", 10000), ("10000 - 10000", 0), ("(((((1)))))", 1),
    ("3+4*2", 11), ("  3  +  4  ", 7), ("3 +4", 7), ("(3)*(4)", 12),
    ("(" * MAX_NESTING + "1" + ")" * MAX_NESTING, 1),
    ("(" * MAX_NESTING + "1" + ")" * MAX_NESTING + " + 2", 3),
    # No spaces at all, chains of division and modulo by zero.
    ("5%0%0", 5), ("5/0/0", 0), ("8/2/2", 2), ("2%3*4", 8), ("7%100*100", 700), ("(0-5)*7", 0), ("99*101", 9999),
    ("1%(2-3)", 1), ("9999%(1-1)+1", 10000), ("+".join(["1"] * 40), 40), ("10000%10000", 0),
    # Not claimed: over the limits, outside the grammar, or not Lean Nat syntax.
    ("10001", None), ("99999", None), ("100000", None), ("9999 + 2", None), ("10000 + 1 - 1", None),
    ("101 * 100 - 100", None), ("10000 * 2 / 2", None), ("(10000 + 1) % 2", None), ("10000 * 10000 % 1", None),
    ("007", None), ("00", None), ("01 + 1", None), ("-3 + 5", None), ("- 3", None), ("3 - -4", None),
    ("3 -- 4", None), ("3 - - 4", None), ("x * x", None), ("2 ^ 3", None), ("3 + ", None), ("", None),
    (" ", None), ("()", None), ("(3", None), ("3)", None), ("3 4", None), ("3 + + 4", None), ("٣ + 4", None),
    ("３ + 4", None), ("\U0001d7d1 + 4", None), ("3 − 4", None), ("3 × 4", None), ("3\t+ 4", None),
    ("3.0 + 4", None), ("0x10", None), ("1_000", None), ("3 + 4 = 7", None),
    ("(" * (MAX_NESTING + 1) + "1" + ")" * (MAX_NESTING + 1), None),
]


def test_lean_semantics():
    for expr, expected in SEMANTICS:
        prompt = f"{HEAD}{expr}{TAIL}"
        assert value(prompt) == expected, (expr, value(prompt), expected)
        assert DOMAIN.claims(prompt) == (expected is not None), expr
        if expected is not None:
            for proof in PROOFS:
                assert DOMAIN.verify(prompt, f"{expected} := {proof}")
            assert not DOMAIN.verify(prompt, f"{expected + 1} := by decide")


def test_exact_answers_only():
    prompt = "example : 3 + 4 ="
    good = ["7 := by decide", "7 := rfl", "7 := by rfl", "7 := by simp"]
    bad = ["7", "7 :=", "7 := ", "7 := by", "07 := rfl", "+7 := rfl", "-7 := rfl", "7.0 := rfl", "７ := rfl",
           "٧ := rfl", "\U0001d7d5 := rfl", " 7 := rfl", "7 := rfl ", "7  := rfl", "7 :=  rfl", "7:= rfl",
           "7 :=rfl", "7 := by  decide", "7 := by\tdecide", "7 := by decide\n", "7 :=\n  rfl", "7 = rfl",
           "7 := (rfl)", "7 := by omega", "7 := by norm_num", "7 := decide", "7 := by native_decide",
           "7 := sorry", "7 := by sorry", "7 := Rfl", "7 := by Decide", "7 := by decide; rfl",
           "7 := by simp only []", "7 := rfl\U0001f600", "\U0001f6007 := rfl", "8 := rfl", "6 := by decide",
           "(7) := rfl", "7 := by exact rfl", "7 := Eq.refl 7", None, 7, ""]
    for a in good:
        assert DOMAIN.verify(prompt, a), a
    for a in bad:
        assert not DOMAIN.verify(prompt, a), a


def test_checker_never_raises():
    weird_prompts = [None, 3, "", HEAD, TAIL, HEAD + TAIL, "example : =", "example :  =", "example : 3 + 4 =\n",
                     HEAD + "(" * 1000 + "1" + ")" * 1000 + TAIL, HEAD + "1 + " * 60 + "1" + TAIL,
                     HEAD + "\ud800" + TAIL, HEAD + "3 + \udfff4" + TAIL, HEAD + "9" * 300 + TAIL,
                     HEAD + "9999 * " * 30 + "9999" + TAIL, HEAD + "1 / 0 % 0" * 20 + TAIL]
    weird_answers = [None, 7, "", "\ud800", "7 := rfl\udfff", "7 := rfl", "x" * 500, "𝔸" * 250, "7 := " + "(" * 300]
    for p in weird_prompts:
        assert DOMAIN.claims(p) in (True, False)
        if isinstance(p, str):
            assert isinstance(DOMAIN.split_key(p), str)
        for a in weird_answers:
            assert DOMAIN.verify(p, a) in (True, False)
    # The prompt length limit (200 UTF-16 code units, as in JavaScript), at its edge.
    pad = 200 - len(HEAD + "1" + TAIL)
    assert DOMAIN.claims(HEAD + " " * pad + "1" + TAIL)
    assert not DOMAIN.claims(HEAD + " " * (pad + 1) + "1" + TAIL)


# ---------------------------------------------------------------- split keys

def spell(t, rng, extra=0.15) -> str:
    """t printed with the parentheses Lean needs, redundant ones added at
    random, and random spacing: another spelling of the same problem."""
    def toks(t):
        if isinstance(t, int):
            out = [str(t)]
        else:
            op, a, b = t
            left, right = toks(a), toks(b)
            if not isinstance(a, int) and nat._PREC[a[0]] < nat._PREC[op]:
                left = ["("] + left + [")"]
            if not isinstance(b, int) and nat._PREC[b[0]] <= nat._PREC[op]:
                right = ["("] + right + [")"]
            out = left + [op] + right
        while rng.random() < extra:
            out = ["("] + out + [")"]
        return out
    parts = toks(t)
    text = parts[0]
    for tok in parts[1:]:
        text += rng.choice(("", " ", " ", "  ")) + tok
    return f"{HEAD}{text}{TAIL}"


def commute(t, rng):
    """t with the operands of + and * swapped and regrouped at random
    (`a + b` to `b + a`, `(a + b) + c` to `a + (b + c)`): the same sum or
    product, the same problem."""
    if isinstance(t, int):
        return t
    op, a, b = t
    a, b = commute(a, rng), commute(b, rng)
    if op not in "+*":
        return (op, a, b)
    if rng.random() < 0.5:
        a, b = b, a
    if rng.random() < 0.5 and isinstance(a, tuple) and a[0] == op:  # (u + v) + b -> u + (v + b)
        return (op, a[1], (op, a[2], b))
    if rng.random() < 0.5 and isinstance(b, tuple) and b[0] == op:  # a + (u + v) -> (a + u) + v
        return (op, (op, a, b[1]), b[2])
    return (op, a, b)


def test_split_key_invariance():
    rng = random.Random(2)
    for p in problems():
        key = DOMAIN.split_key(p.prompt)
        for i in range(8):
            t = commute(tree(p.prompt), rng) if i % 2 else tree(p.prompt)
            other = spell(t, rng, extra=0.3)
            if not DOMAIN.claims(other):  # too long or too deep, or regrouped past MAX_VALUE
                assert len(other) > 200 or other.count("(") > MAX_NESTING or nat._evaluate(t) is None, other
                continue
            assert DOMAIN.split_key(other) == key, (p.prompt, other)
            assert value(other) == value(p.prompt)
            assert data.split_of(other) == data.split_of(p.prompt)
    assert DOMAIN.split_key("example : ((3)) +  (4 * 5) =") == "example : 3 + 4 * 5 ="
    assert DOMAIN.split_key("example : 4 * 5 + 3 =") == "example : 3 + 4 * 5 ="
    assert DOMAIN.split_key("example : (3 + 4) + 5 =") == "example : 3 + 4 + 5 ="
    assert DOMAIN.split_key("example : 3 + (5 + 4) =") == "example : 3 + 4 + 5 ="
    assert DOMAIN.split_key("example : 3 - 07 =") == "example : 3 - 07 ="  # unclaimed: the prompt itself


def test_commuted_twins_share_a_split():
    """Regression: keys used to keep the operands in the order written, and the
    generator writes both `3 + 4` and `4 + 3`. Of the problems a 40,000-draw
    stream (pretraining's share per domain) put in the test split, 22% had a
    commuted twin, with its answer, in the training split."""
    for a, b in (("3 + 4", "4 + 3"), ("2 * 7 - 1", "7 * 2 - 1"), ("(1 + 2) + 3", "1 + (2 + 3)"),
                 ("6 / 2 * 3", "3 * (6 / 2)"), ("(5 % 3 + 1) * (2 + 8)", "(8 + 2) * (1 + 5 % 3)")):
        assert DOMAIN.split_key(f"{HEAD}{a}{TAIL}") == DOMAIN.split_key(f"{HEAD}{b}{TAIL}"), (a, b)
    def commuted(t):
        """Every tree that swapping the operands of some + and * nodes makes."""
        if isinstance(t, int):
            return {t}
        op, a, b = t
        return {u for x in commuted(a) for y in commuted(b) for u in ((op, x, y), (op, y, x))[:1 + (op in "+*")]}

    rng, train, test = data._domain_rng(0, "nat"), set(), []
    for _ in range(20_000):
        p = DOMAIN.sample(rng)
        split = data.split_of_key(DOMAIN.split_key(p.prompt))
        if split == "train":
            train.add(p.prompt)
        elif split == "test":
            test.append(p.prompt)
    twins = [q for q in test if any(f"{HEAD}{nat.show(t)}{TAIL}" in train for t in commuted(tree(q)))]
    assert len(test) > 800 and not twins, twins[:5]


def test_distinct_problems_have_distinct_keys():
    """Prompts share a key only if they are the same sum or product up to the
    order and grouping of its operands; then Lean gives them one value."""
    ps = problems(5000, 3)
    rng = random.Random(6)
    prompts = [p.prompt for p in ps] + [spell(_big_tree(rng, rng.randint(1, 6)), rng, 0) for _ in range(5000)]
    by_key = {}
    for p in prompts:
        if DOMAIN.claims(p):
            by_key.setdefault(DOMAIN.split_key(p), []).append(p)
    for key, same in by_key.items():
        assert len({nat.show(nat.canonical(tree(p))) for p in same}) == 1, same
        assert len({value(p) for p in same}) == 1, same
    assert len({DOMAIN.split_key(p.prompt) for p in ps}) == len({repr(nat.canonical(tree(p.prompt))) for p in ps})
    # Different expressions with the same value are different problems; so are
    # the operands of - / % in the other order, and another grouping of them.
    exprs = ("3 + 4", "7", "14 / 2", "3 + 4 + 0", "3 + 4 * 1", "5 - 3", "3 - 5", "8 / 2 / 2", "8 / (2 / 2)",
             "6 / (2 * 3)", "6 / 2 * 3", "2 * (3 + 4)", "2 * 3 + 4", "7 % 4 % 2", "7 % (4 % 2)", "9 - 2 - 1", "9 - (2 - 1)")
    assert len({DOMAIN.split_key(f"{HEAD}{e}{TAIL}") for e in exprs}) == len(exprs)


# ---------------------------------------------------------------- differential against web/checker.js

ARABIC, FULLWIDTH, DEVANAGARI = "٠١٢٣٤٥٦٧٨٩", "０１２３４５６７８９", "०१२३४५६७८९"
BOLD = "".join(chr(0x1D7CE + i) for i in range(10))  # astral digits, two UTF-16 code units each
ASTRAL = ("𝔸", "\U0001f600", "\U0001d7d5", "\U00010000")
WRONG_PROOFS = ("by omega", "by norm_num", "decide", "by decide!", "by native_decide", "Rfl", "by Rfl", "by  decide",
                "by\tdecide", "by decide;", "by simp only []", "by simp_arith", "by exact rfl", "(rfl)", "by trivial",
                "sorry", "by sorry", "rfl rfl", "by decide\n", "by\n  decide", "Eq.refl _", "by simp; rfl", "",
                "by", "by rfl rfl", "by decide <;> rfl", "rfl\n", " rfl", "by simp ", "by decid", "by simpp")


def _digits(s: str, script: str, rng, every: bool) -> str:
    idx = [i for i, c in enumerate(s) if c.isdigit() and c < "\x80"]
    if not idx:
        return s
    chosen = idx if every else [rng.choice(idx)]
    return "".join(script[int(c)] if i in chosen else c for i, c in enumerate(s))


def mutate_answer(answer: str, v: int, rng: random.Random) -> str:
    """A near miss of a correct answer: what a model that almost knows the value
    or the proof syntax writes."""
    head, proof = answer.split(" := ", 1)
    kind = rng.randrange(17)
    if kind == 0:  # one digit changed
        i = rng.randrange(len(head))
        return head[:i] + rng.choice([d for d in "0123456789" if d != head[i]]) + head[i + 1:] + " := " + proof
    if kind == 1:  # off by a little, or a lot
        return f"{max(v + rng.choice((-10, -2, -1, 1, 2, 10, v, 10000)), 0)} := {proof}"
    if kind == 2:  # sign
        return rng.choice(("-", "+", "−", "- ", "--", "(-")) + answer
    if kind == 3:
        return f"{head} := {rng.choice(WRONG_PROOFS)}"
    if kind == 4:  # a space somewhere, or a doubled one
        i = rng.randrange(len(answer) + 1)
        return answer[:i] + rng.choice((" ", "  ", "\t", "\n", "\u00a0")) + answer[i:]
    if kind == 5:
        return answer[:rng.randrange(len(answer))]
    if kind == 6:
        return _digits(answer, rng.choice((ARABIC, FULLWIDTH, DEVANAGARI, BOLD)), rng, rng.random() < 0.5)
    if kind == 7:
        i = rng.randrange(len(answer) + 1)
        return answer[:i] + rng.choice(ASTRAL) + answer[i:]
    if kind == 8:
        return rng.choice(("0", "00")) + answer
    if kind == 9:
        return rng.choice((f"{head}.0", f"({head})", f"{head}e0", f"0x{v:x}", f"{head}_0", f"{head}u")) + " := " + proof
    if kind == 10:
        i = rng.randrange(len(answer))
        return answer[:i] + answer[i + 1:]
    if kind == 11:
        return head + rng.choice((" = ", " :=", ":= ", ":=", " : = ", " ≔ ", " :== ")) + proof
    if kind == 12:
        return answer + rng.choice((" ", "\n", ";", " " + answer, "\n" + answer))
    if kind == 13:
        return f"{head} := {rng.choice(PROOFS)}"  # another proof form: still correct
    if kind == 14:
        return answer + " " * rng.choice((380, 400))  # over the length limit
    if kind == 15:
        i, j = sorted(rng.sample(range(len(answer)), 2))
        return answer[:i] + answer[j] + answer[i + 1:j] + answer[i] + answer[j + 1:]
    return f"{head} := {rng.choice(PROOFS)}".replace(" ", "")


_SOUP = list("0123456789") * 3 + list("+-*/%()") * 2 + [" "] * 8 + ["00", "10000", "9999", "--", "- -", "x", "^",
                                                                       "٣", "３", "\U0001d7d1", "−", "×", "÷", "\t"]


def soup(rng: random.Random) -> str:
    """A random string of the grammar's characters: what an untrained model writes."""
    return "".join(rng.choice(_SOUP) for _ in range(rng.randint(1, 30)))


def _big_tree(rng: random.Random, n_ops: int):
    """A random tree with larger numbers than the generator's, so that values
    and intermediate results cross MAX_VALUE."""
    if n_ops == 0:
        return rng.choice((0, 1, 2, rng.randint(0, 100), rng.randint(0, 10000), rng.choice((9999, 10000))))
    k = rng.randint(0, n_ops - 1)
    return (rng.choice("+-*/%"), _big_tree(rng, k), _big_tree(rng, n_ops - 1 - k))


def _replace_leaf(t, rng, new):
    """t with one leaf, chosen at random, replaced by new(leaf) (a string of
    expression text, so it may be anything); returns the spelled expression."""
    leaves = []

    def walk(u, path):
        if isinstance(u, int):
            leaves.append(path)
        else:
            walk(u[1], path + (1,))
            walk(u[2], path + (2,))
    walk(t, ())
    target = rng.choice(leaves)

    def show(u, path):
        if isinstance(u, int):
            return new(u) if path == target else str(u)
        op, a, b = u
        left, right = show(a, path + (1,)), show(b, path + (2,))
        return f"({left}) {op} ({right})"
    return show(t, ())


def mutate_prompt(prompt: str, rng: random.Random) -> str:
    t = tree(prompt)
    body = prompt[len(HEAD):-len(TAIL)]
    kind = rng.randrange(16)
    if kind == 0:  # respelled: redundant parentheses, spacing
        return spell(t, rng, extra=rng.choice((0.1, 0.3, 0.6)))
    if kind == 1:
        return f"{HEAD}{_replace_leaf(t, rng, lambda n: rng.choice(('0', '00')) + str(n))}{TAIL}"
    if kind == 2:  # numbers at and above the limit
        return f"{HEAD}{_replace_leaf(t, rng, lambda n: str(rng.choice((9999, 10000, 10001, 99999, 100000, 123456))))}{TAIL}"
    if kind == 3:  # intermediate results above MAX_VALUE
        big = rng.choice(("10000", "9999", "5001", "5000", "101", "100"))
        forms = (f"({body}) * {big} / {big}", f"{big} * ({body}) - {big}", f"{big} + ({body}) - {big}",
                 f"({body}) * {big} % 7", f"{big} * {big} / ({body})")
        return f"{HEAD}{rng.choice(forms)}{TAIL}"
    if kind == 4:  # division and modulo by zero, also by a zero that truncated subtraction makes
        zero = rng.choice(("0", "(0)", "(2 - 5)", "(3 - 3)", "(0 * 7)", "(1 / 2)"))
        return f"{HEAD}{_replace_leaf(t, rng, lambda n: f'{n} {rng.choice(chr(47) + chr(37))} {zero}')}{TAIL}"
    if kind == 5:  # minus signs: `--` starts a comment, `- -` and prefix minus are not Nat syntax
        return f"{HEAD}{_replace_leaf(t, rng, lambda n: rng.choice(('-', '- ', '- -', '--', '(-')) + str(n))}{TAIL}"
    if kind == 6:  # nesting at the limit
        d = rng.choice((MAX_NESTING - 1, MAX_NESTING, MAX_NESTING + 1))
        return f"{HEAD}{'(' * d}{body}{')' * d}{TAIL}"
    if kind == 7:  # a character inserted, deleted or swapped
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
    if kind == 8:  # an operator changed
        ops = [i for i, c in enumerate(body) if c in "+-*/%"]
        if ops:
            i = rng.choice(ops)
            body = body[:i] + rng.choice("+-*/%") + body[i + 1:]
        return f"{HEAD}{body}{TAIL}"
    if kind == 9:  # the header changed
        return rng.choice(("example :", "example: ", "example :  ", "theorem t : ", "example (n : Nat) : ",
                           "example : (", "Example : ", "example  : ")) + body + rng.choice((TAIL, " = ", "=", " :=", " = 0", ")" + TAIL))
    if kind == 10:  # spaces up to and past the length limit
        pad = 200 - len(prompt) + rng.choice((-1, 0, 0, 1))
        return f"{HEAD}{' ' * max(pad, 0)}{body}{TAIL}"
    if kind == 11:  # digits from other scripts
        return f"{HEAD}{_digits(body, rng.choice((ARABIC, FULLWIDTH, DEVANAGARI, BOLD)), rng, False)}{TAIL}"
    if kind == 12:  # an astral character
        i = rng.randrange(len(body) + 1)
        return f"{HEAD}{body[:i]}{rng.choice(ASTRAL)}{body[i:]}{TAIL}"
    if kind == 13:
        return f"{HEAD}{show_big(rng)}{TAIL}"
    if kind == 14:
        return f"{HEAD}{soup(rng)}{TAIL}"
    # combined with another problem
    other = tree(rng.choice(problems()).prompt)
    return spell((rng.choice("+-*/%"), t, other), rng, extra=0.1)


def show_big(rng: random.Random) -> str:
    return spell(_big_tree(rng, rng.choice((1, 2, 3, 4, 6))), rng, extra=0.1)[len(HEAD):-len(TAIL)]


@functools.lru_cache(maxsize=None)
def differential_cases() -> tuple:
    """(prompt, answer, origin): references and every proof form, near misses
    of them, mutated prompts with the answer Python computes for them (so a
    disagreement on a value shows), and random strings."""
    rng = random.Random(0)
    cases = []
    ps = problems()
    for p in ps:
        v = value(p.prompt)
        cases += [(p.prompt, f"{v} := {proof}", "valid") for proof in PROOFS]
        for _ in range(16):
            a = f"{v} := {rng.choice(PROOFS)}"
            for _ in range(rng.choice((1, 1, 1, 2))):
                if " := " not in a:
                    break
                a = mutate_answer(a, v, rng)
            cases.append((p.prompt, a, "answer mutant"))
        cases.append((p.prompt, rng.choice(ps).reference, "other answer"))
        cases.append((p.prompt, soup(rng), "random answer"))
    for p in ps:
        for _ in range(12):
            q = mutate_prompt(p.prompt, rng)
            v = value(q)
            if v is None:
                cases.append((q, p.reference, "prompt mutant"))
            else:
                cases.append((q, f"{v} := {rng.choice(PROOFS)}", "prompt mutant"))
                cases.append((q, f"{v + rng.choice((1, -1))} := by decide", "prompt mutant"))
    for expr, expected in SEMANTICS:
        prompt = f"{HEAD}{expr}{TAIL}"
        for proof in PROOFS:
            cases.append((prompt, f"{expected if expected is not None else 0} := {proof}", "by hand"))
    # Shapes and spellings the generator never makes: up to ten operators,
    # numbers up to the limit, no spaces at all or many redundant parentheses.
    # The ones the checker accepts go to Lean with the rest.
    rng = random.Random(1)
    for _ in range(500):
        body = spell(_big_tree(rng, rng.randint(1, 10)), rng, extra=rng.choice((0, 0.3)))[len(HEAD):-len(TAIL)]
        q = f"{HEAD}{body.replace(' ', '') if rng.random() < 0.5 else body}{TAIL}"
        v = value(q)
        cases.append((q, f"{0 if v is None else v} := {rng.choice(PROOFS)}", "grammar"))
    return tuple(cases)


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
    print(f"nat differential: {len(cases)} cases, {len(mismatches)} disagreements; "
          + "; ".join(f"{o}: {a} accepted, {r} rejected, {u} unclaimed" for o, (a, r, u) in counts.items()))
    assert not mismatches, mismatches[:10]
    assert all(v is True for (_, _, origin), v in zip(cases, py) if origin == "valid")
    # The mutants exercise both sides of every check.
    for origin in ("answer mutant", "prompt mutant"):
        accepted, rejected, unclaimed = counts[origin]
        assert accepted > 500 and rejected > 500, (origin, counts[origin])
    assert counts["prompt mutant"][2] > 1000
    assert counts["grammar"][0] > 100 and counts["grammar"][2] > 100, counts["grammar"]


def test_every_short_body_against_javascript(node):
    """Every body of up to four symbols over the grammar's characters and the
    numbers at the limit (30,940 prompts), with the value Python computes: the
    corner of the input space (lone operators, empty parentheses, `0 % 0`)
    that random mutants of real problems reach only by chance."""
    symbols = list("019+-*/%() ") + ["10000", "9999"]
    prompts = [HEAD + "".join(s) + TAIL for n in range(1, 5) for s in itertools.product(symbols, repeat=n)]
    pairs = [(p, f"{0 if value(p) is None else value(p)} := rfl") for p in prompts]
    js = js_verify(pairs)
    py = [_python_verdict(p, a) for p, a in pairs]
    mismatches = [(c, j, q) for c, j, q in zip(pairs, js, py) if j != q]
    claimed = sum(v is not None for v in py)
    print(f"nat short bodies: {len(pairs)} prompts, {claimed} claimed, {len(mismatches)} disagreements")
    assert not mismatches, mismatches[:10]
    assert claimed > 1000


def js_verify_escaped(node, pairs) -> list:
    """js_verify for strings that UTF-8 cannot carry (lone surrogates, which a
    JavaScript string can hold): the JSON goes to Node ASCII-escaped."""
    data = "".join(json.dumps([p, a]) + "\n" for p, a in pairs)
    out = subprocess.run([node, os.path.join(ROOT, "tests", "js_bridge.js")], input=data, capture_output=True,
                         text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line]


def test_lone_surrogates_against_javascript(node):
    """A lone surrogate is never in the grammar, so every case is rejected or
    unclaimed; the point is that Python neither raises nor counts lengths
    differently (a prompt at the limit plus one code unit)."""
    rng = random.Random(4)
    cases = []
    for p in problems()[:100]:
        v = value(p.prompt)
        body = p.prompt[len(HEAD):-len(TAIL)]
        for s in ("\ud800", "\udfff", "\udbff\udbff"):
            i = rng.randrange(len(body) + 1)
            cases += [(p.prompt, f"{v} := rfl{s}"), (p.prompt, f"{s}{v} := rfl"), (p.prompt + s, p.reference),
                      (f"{HEAD}{body[:i]}{s}{body[i:]}{TAIL}", p.reference),
                      (f"{HEAD}{' ' * (200 - len(p.prompt))}{body}{TAIL}{s}", p.reference),
                      (f"{HEAD}{' ' * (200 - len(p.prompt))}{body}{TAIL}", p.reference + s)]
    js = js_verify_escaped(node, cases)
    py = [_python_verdict(p, a) for p, a in cases]
    assert len(js) == len(cases) and js == py, [(c, j, q) for c, j, q in zip(cases, js, py) if j != q][:10]


# ---------------------------------------------------------------- against Lean

EDGE_CASES = [(f"{HEAD}{e}{TAIL}", v) for e, v in SEMANTICS if v is not None]


def _stated_value(answer: str):
    """The number an answer states, read as Lean reads it (leading zeros,
    hexadecimal, digit separators, arithmetic such as `0+18`, a decimal point
    with a zero fraction), or None. Lean accepts `1 * 19 = 19.0 := by rfl`:
    the decimal makes the whole equation one about Float, which still says 19."""
    s = answer.split(":=", 1)[0].strip().replace("_", "")
    s = re.sub(r"0[xX][0-9a-fA-F]+", lambda m: str(int(m.group(), 16)), s)
    s = re.sub(r"([0-9]+)\.0*(?![0-9])", r"\1", s)
    s = re.sub(r"\b0+(?=[0-9])", "", s)
    return value(f"{HEAD}{s}{TAIL}")


def test_soundness_against_lean(lean):
    """Every proof form of 500 references, the edge cases of Lean's Nat
    semantics, and every mutant the checker accepts compile. Rejected answers
    that Lean accepts must state the checker's value: the checker may refuse
    a respelling (`007`, a newline before the proof), never a value Lean
    computes differently."""
    valid = [f"{p.prompt} {value(p.prompt)} := {proof}" for p in problems()[:500] for proof in PROOFS]
    valid += [f"{prompt} {v} := {proof}" for prompt, v in EDGE_CASES for proof in PROOFS]
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
    print(f"nat vs Lean: {len(set(valid))} references, proof forms and edge cases, {len(accepted)} accepted mutants; "
          f"of {len(sample)} rejected answers Lean accepts {len(lean_accepts)}")
    for d in lean_accepts[:10]:
        print("   checker rejects, Lean accepts:", repr(d))
    assert not failed_valid, failed_valid[:10]
    assert not failed_accepted, failed_accepted[:10]
    wrong = [d for d in lean_accepts if _stated_value(rejected[d][1]) != value(rejected[d][0])]
    assert not wrong, wrong[:10]
    assert len(lean_accepts) < len(sample) - 400  # most near misses are wrong in Lean too
