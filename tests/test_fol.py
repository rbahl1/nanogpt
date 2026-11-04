"""The fol domain: generated references and documents check and fit the
model's context, sampling is deterministic and fast, split keys ignore
renaming and respelling, the Python checker agrees with web/checker.js on
every case, and Lean accepts everything the checker accepts.

Run with -s to see the measurements (time per sample, agreement counts, how
many answers Lean accepts that the checker rejects)."""

import itertools
import random
import re
import sys
import time

import pytest

import lean_oracle
from formal import fol
from formal.fol import DOMAIN
from js_checker import js_verify
from tokenizer import Tokenizer

TOKENIZER = Tokenizer()


def _problems(n, seed, unique=False):
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        p = DOMAIN.sample(rng)
        key = DOMAIN.split_key(p.prompt)
        if unique and key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


@pytest.fixture(scope="module")
def problems():
    return _problems(2000, seed=0)


def _answer(problem, document):
    assert document.startswith(problem.prompt + " ")
    return document[len(problem.prompt) + 1:]


# ---------------------------------------------------------------- generation

def test_references_check_and_fit(problems):
    for p in problems:
        assert p.domain == "fol"
        assert DOMAIN.claims(p.prompt), p.prompt
        assert DOMAIN.verify(p.prompt, p.reference), (p.prompt, p.reference)
        assert len(p.prompt) <= 160 and len(p.reference) <= 200, p
        ids, n_prompt = TOKENIZER.example_ids(p.prompt, p.reference)
        assert len(ids) <= 256 and n_prompt == len(p.prompt) + 2
        assert not p.reference.startswith("by")
        # hypotheses h0, h1, ... and individuals x, y, z in binding order
        assert DOMAIN.verify(p.prompt, fol._show_term(fol._rename(fol._parse_term(p.reference, ["a", "b"]))))
        assert fol._show_term(fol._rename(fol._parse_term(p.reference, ["a", "b"]))) == p.reference


def test_sampling_is_deterministic_and_fast():
    first = [DOMAIN.sample(random.Random(7)) for _ in range(3)]
    assert first[0] == first[1] == first[2]
    rng_a, rng_b = random.Random(11), random.Random(11)
    a = [DOMAIN.sample(rng_a) for _ in range(300)]
    b = [DOMAIN.sample(rng_b) for _ in range(300)]
    assert a == b
    docs_a = [DOMAIN.document(p, random.Random(i)) for i, p in enumerate(a)]
    docs_b = [DOMAIN.document(p, random.Random(i)) for i, p in enumerate(b)]
    assert docs_a == docs_b
    rng = random.Random(3)
    start = time.perf_counter()
    n = 3000
    for _ in range(n):
        DOMAIN.sample(rng)
    per_sample = (time.perf_counter() - start) / n
    print(f"\nfol: {per_sample * 1e3:.3f} ms per sample()")
    assert per_sample < 5e-3


def test_distribution_has_variety(problems):
    prompts = {p.prompt for p in problems}
    keys = {DOMAIN.split_key(p.prompt) for p in problems}
    shapes = {re.sub(r"\b[ab]\b", "c", p.reference) for p in problems}
    lengths = sorted(len(p.reference) for p in problems)
    print(f"\nfol: {len(prompts)} distinct prompts, {len(keys)} split keys, {len(shapes)} proof shapes in "
          f"{len(problems)}; reference length median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
    assert len(keys) > 0.8 * len(problems)
    assert len(shapes) > 100
    assert lengths[0] < 20 and lengths[-1] > 60  # easy and hard ones


def test_documents_check_and_vary(problems):
    rng = random.Random(1)
    changed = 0
    for p in problems:
        doc = DOMAIN.document(p, rng)
        answer = _answer(p, doc)
        assert DOMAIN.verify(p.prompt, answer), doc
        assert len(TOKENIZER.document_ids(doc)) <= 256
        changed += answer != p.reference
    print(f"\nfol: {changed}/{len(problems)} documents differ from the reference")
    assert changed > 0.8 * len(problems)


def test_document_binders_are_named_for_their_kind(problems, monkeypatch):
    """Documents name hypotheses and individuals from separate schemes, and an
    η-expansion's binder by what it binds: `fun h => t h` for an implication
    t, `fun x => t x` for a ∀ (both check, so the checker cannot tell)."""
    rng = random.Random(4)
    documents = [(p.prompt, _answer(p, DOMAIN.document(p, rng))) for p in problems]
    hyp_names = {scheme(k) for scheme in fol._HYP_SCHEMES for k in range(10)}
    ind_names = {scheme(k) for scheme in fol._IND_SCHEMES for k in range(10)}
    assert not hyp_names & ind_names
    individuals, hypotheses = set(), set()
    new_individual, bind_hyp = fol._new_individual, fol._bind_hyp

    def record_individual(ctx, name, body):
        individuals.add(name)
        return new_individual(ctx, name, body)

    def record_hyp(ctx, binder, ty):
        if isinstance(binder, str):
            hypotheses.add(binder)
        return bind_hyp(ctx, binder, ty)

    monkeypatch.setattr(fol, "_new_individual", record_individual)
    monkeypatch.setattr(fol, "_bind_hyp", record_hyp)
    for prompt, answer in documents:
        individuals.clear()
        hypotheses.clear()
        assert DOMAIN.verify(prompt, answer)
        assert not hypotheses & ind_names and not individuals & hyp_names, (prompt, answer)


# ---------------------------------------------------------------- split keys

_PREC = {"iff": 20, "imp": 25, "or": 30, "and": 35}
_SYMBOL = {"and": "∧", "or": "∨", "imp": "→", "iff": "↔"}
_BOUND_NAMES = ["x", "y", "z", "w", "u", "v", "t", "x'", "s"]


def _noisy(f, rng, rename, bound, prec=0, tail=False):
    """f printed with random bound-variable names, redundant parentheses and
    extra spaces (a test-side printer, independent of fol._show's choices)."""
    if rng.random() < 0.15:
        return "(" + _noisy(f, rng, rename, bound) + ")"
    sp = "  " if rng.random() < 0.1 else " "
    tag = f[0]
    if tag == "pred":
        args = [bound[len(bound) - 1 - x] if isinstance(x, int) else rename.get(x, x) for x in f[2]]
        return sp.join([rename.get(f[1], f[1])] + args)
    if tag in ("top", "bot"):
        return "True" if tag == "top" else "False"
    if tag == "not":
        return "¬" + _noisy(f[1], rng, rename, bound, 40, tail)
    if tag in ("all", "ex"):
        name = rng.choice([v for v in _BOUND_NAMES if v not in bound])
        s = f"{'∀' if tag == 'all' else '∃'} {name},{sp}{_noisy(f[1], rng, rename, bound + [name])}"
        return f"({s})" if tail else s
    p = _PREC[tag]
    if p < prec:
        return "(" + _noisy(f, rng, rename, bound) + ")"
    left = _noisy(f[1], rng, rename, bound, p + 1, True)
    right = _noisy(f[2], rng, rename, bound, p + 1 if tag == "iff" else p, tail)
    return f"{left}{sp}{_SYMBOL[tag]}{sp}{right}"


def _respelled(prompt, rng):
    """The same problem with the symbols of each group permuted, bound
    variables renamed and redundant parentheses and spaces added."""
    goal, _ = fol._goal_of(prompt)
    rename = {}
    for group in fol.GROUPS:
        rename.update(zip(group, rng.sample(group, len(group))))
    preds, used = set(), set()
    fol._symbols(goal, preds, used)
    names = {rename[s] for s in preds | used}
    header = ["example (α : Type)"]
    for group, kind in zip(fol.GROUPS, ("α → Prop", "α → α → Prop", "Prop", "α")):
        declared = sorted(n for n in names if n in group)
        if declared:
            header.append(f"({' '.join(declared)} : {kind})")
    return f"{' '.join(header)} : {_noisy(goal, rng, rename, [])} :="


def test_split_key_ignores_renaming_and_parentheses(problems):
    rng = random.Random(2)
    for p in problems[:1000]:
        key = DOMAIN.split_key(p.prompt)
        assert DOMAIN.claims(key) and DOMAIN.split_key(key) == key
        for _ in range(3):
            other = _respelled(p.prompt, rng)
            assert DOMAIN.claims(other), other
            assert DOMAIN.split_key(other) == key, (p.prompt, other)
    assert DOMAIN.split_key("example : 1 + 1 =") == "example : 1 + 1 ="
    assert DOMAIN.split_key("not a prompt") == "not a prompt"


def test_split_key_separates_different_problems():
    a = "example (α : Type) (P Q : α → Prop) : (∃ x, P x ∧ Q x) → ∃ x, P x :="
    renamed = "example (α : Type) (Q R : α → Prop) : (∃ y, (R y ∧ (Q y))) → ∃ z, R z :="
    assert DOMAIN.split_key(a) == DOMAIN.split_key(renamed)
    for different in ("example (α : Type) (P Q : α → Prop) : (∃ x, P x ∨ Q x) → ∃ x, P x :=",
                      "example (α : Type) (P Q : α → Prop) : (∃ x, P x ∧ Q x) → ∃ x, Q x :=",
                      "example (α : Type) (P Q : α → Prop) : (∀ x, P x ∧ Q x) → ∃ x, P x :="):
        assert DOMAIN.split_key(different) != DOMAIN.split_key(a)


# ---------------------------------------------------------------- mutants

_SWAPS = [("Or.inl", "Or.inr"), (".1", ".2"), (".mp", ".mpr"), (".left", ".right"), (".1", ".mp"), (".2", ".mpr"),
          (".1", ".left"), ("absurd", "False.elim"), ("And.intro", "Iff.intro"), ("Exists.intro", "And.intro"),
          (".elim", ".1"), (".elim", ".2"), ("⟨", "("), ("⟩", ")"), ("trivial", "True.intro"), ("fun", "λ"),
          ("=>", "↦"), ("Or.elim", "Exists.elim"), (".elim", ""), ("(", "⟨"), (")", "⟩"), ("h0", "h1"), ("h1", "h2"),
          ("x", "y"), ("a", "b"), ("Or.inl", "Or.inl ("), ("=>", "=> ⟨_, "), ("trivial", "_")]
_CHARS = list(" ()⟨⟩,._'=>0123456789hxyzabcPQRrpq¬∧∨→↔∀∃") + ["fun ", " => ", "Or.inl ", "Or.inr ", ".elim", ".1",
                                                                 ".2", ".mp", "absurd ", "⟨x, h0⟩", "trivial", "_ "]
_SPACES = [" ", "  ", "", "\t", "\n", " ", "　", " ."]


def _mutate_once(s, rng):
    op = rng.randrange(13)
    i = rng.randrange(len(s) + 1)
    if op == 0 and s:  # delete a character
        return s[:i] + s[i + 1:]
    if op == 1:  # insert a character or token
        return s[:i] + rng.choice(_CHARS) + s[i:]
    if op == 2 and s:  # replace a character
        return s[:i] + rng.choice(_CHARS) + s[i + 1:]
    if op == 3 and len(s) > 1:  # swap neighbours
        i = min(i, len(s) - 2)
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == 4:  # swap constructors, fields, names (one occurrence)
        a, b = rng.choice(_SWAPS)
        if rng.random() < 0.5:
            a, b = b, a
        spots = [m.start() for m in re.finditer(re.escape(a), s)]
        if spots:
            k = rng.choice(spots)
            return s[:k] + b + s[k + len(a):]
        return s
    if op == 5:  # wrong individual, or a name renamed everywhere
        words = sorted(set(re.findall(r"\b[a-z][0-9_']*\b", s)))
        if words:
            w = rng.choice(words)
            new = rng.choice(["a", "b", "x", "y", "z", "h0", "h1", "h2", "_", w + "'"])
            if rng.random() < 0.5:
                return re.sub(rf"(?<![\w.']){re.escape(w)}(?![\w'])", new, s)
            spots = [m.start() for m in re.finditer(rf"(?<![\w.']){re.escape(w)}(?![\w'])", s)]
            if spots:
                k = rng.choice(spots)
                return s[:k] + new + s[k + len(w):]
        return s
    if op == 6:  # drop or reshape a pattern
        pats = list(re.finditer(r"⟨([^⟨⟩]*)⟩", s))
        if pats:
            m = rng.choice(pats)
            items = m.group(1).split(", ")
            choice = rng.randrange(3)
            if choice == 0:
                new = rng.choice(items)
            elif choice == 1 and len(items) > 2:
                new = "⟨" + ", ".join(items[:-1]) + "⟩"
            else:
                new = "⟨" + ", ".join(items[::-1]) + "⟩"
            return s[:m.start()] + new + s[m.end():]
        return s
    if op == 7:  # truncate
        return s[:rng.randrange(len(s) + 1)]
    if op == 8:  # whitespace
        spots = [m.start() for m in re.finditer(" ", s)] or [0]
        k = rng.choice(spots)
        choice = rng.randrange(4)
        if choice == 0:
            return s[:k] + rng.choice(_SPACES) + s[k + 1:]
        if choice == 1:
            return " " + s if rng.random() < 0.5 else s + " "
        if choice == 2:
            return s.replace(" ", "  ")
        return s[:i] + " " + s[i:]
    if op == 9:  # wrap, prefix
        return rng.choice(["(" + s + ")", "by exact " + s, "⟨" + s + "⟩", s + " a", "fun h0 => " + s, s + ".1",
                           "(" + s + ").elim", "by\n  exact " + s, "id " + s, "@" + s])
    if op == 10:  # duplicate or drop a word
        words = s.split(" ")
        k = rng.randrange(len(words))
        if rng.random() < 0.5:
            return " ".join(words[:k] + [words[k]] + words[k:])
        return " ".join(words[:k] + words[k + 1:])
    if op == 11:  # swap two words
        words = s.split(" ")
        if len(words) > 1:
            k, j = rng.sample(range(len(words)), 2)
            words[k], words[j] = words[j], words[k]
        return " ".join(words)
    # swap the two cases of an Or elimination, or the two halves of an ⟨⟩
    m = re.search(r"\((fun [^()]*)\) \((fun [^()]*)\)", s)
    if m:
        return s[:m.start()] + f"({m.group(2)}) ({m.group(1)})" + s[m.end():]
    return s


def _mutants(s, rng, n):
    out = []
    for _ in range(n):
        t = s
        for _ in range(rng.choice((1, 1, 1, 2, 3))):
            t = _mutate_once(t, rng)
        out.append(t)
    return out


_NAMES = ["h", "h0", "h1", "h2", "x", "y", "z", "a", "b", "_", "w"]
_SAME_ARITY = {1: ["Or.inl", "Or.inr", "False.elim", "And.left", "And.right"],
               2: ["absurd", "And.intro", "Iff.intro", "Exists.intro", "Exists.elim"], 3: ["Or.elim"]}


def _rebound(t, rng):
    """t with one binder, and the variables it binds, renamed to a name that
    may capture a variable of an outer binder or a declared individual."""
    lams = [n for n in fol._nodes(t, []) if n[0] == "lam"]
    if not lams:
        return t
    lam, names = rng.choice(lams), []
    fol._binder_names(lam[1], names)
    names = [n for n in names if n != "_"]
    if not names:
        return t
    old, new = rng.choice(names), rng.choice(_NAMES)

    def binder(b):
        return (new if b == old else b) if isinstance(b, str) else ("pat", [binder(x) for x in b[1]])

    def body(u):  # the free occurrences of old in u
        tag = u[0]
        if tag == "var":
            return ("var", new) if u[1] == old else u
        if tag == "lam":
            bound = []
            fol._binder_names(u[1], bound)
            return u if old in bound else ("lam", u[1], body(u[2]))
        if tag == "proj":
            return ("proj", body(u[1]), u[2])
        if tag == "app":
            return ("app", body(u[1]), [body(a) for a in u[2]])
        return ("const", u[1], [body(a) for a in u[2]]) if tag == "const" else ("anon", [body(a) for a in u[1]])

    index = next(k for k, n in enumerate(fol._nodes(t, [])) if n is lam)
    return fol._replace(t, index, ("lam", binder(lam[1]), body(lam[2])))


def _tree_mutants(prompt, answer, rng, n):
    """Structural edits of a proof term: arguments permuted (Exists.elim and
    Or.elim in the wrong order), a binder renamed so that it captures, a
    subterm transplanted, a constant swapped for another of its arity, an
    application re-bracketed, a variable swapped, a projection added or removed."""
    t = fol._parse_term(answer, fol._goal_of(prompt)[1])
    out = []
    for _ in range(n):
        m = t
        for _ in range(rng.choice((1, 1, 2))):
            nodes = fol._nodes(m, [])
            k = rng.randrange(len(nodes))
            node, op = nodes[k], rng.randrange(8)
            new = None
            if op == 0 and node[0] in ("app", "const", "anon"):
                items = list(node[1] if node[0] == "anon" else node[2])
                rng.shuffle(items)
                new = ("anon", items) if node[0] == "anon" else (node[0], node[1], items)
            elif op == 1:
                m = _rebound(m, rng)
            elif op == 2:
                new = rng.choice(nodes)
            elif op == 3 and node[0] == "const" and node[2]:
                new = ("const", rng.choice(_SAME_ARITY[len(node[2])]), node[2])
            elif op == 4 and node[0] == "app" and len(node[2]) > 1:
                j = rng.randrange(1, len(node[2]))
                new = ("app", ("app", node[1], node[2][:j]), node[2][j:]) if rng.random() < 0.5 else \
                    ("app", node[1], node[2][:j - 1] + [("app", node[2][j - 1], node[2][j:])])
            elif op == 5 and node[0] == "var":
                new = ("var", rng.choice(_NAMES))
            elif op == 6:
                new = ("proj", node, rng.choice(["1", "2", "elim", "mp", "mpr", "left", "right"]))
            elif op == 7 and node[0] == "proj":
                new = node[1]
            if new is not None:
                m = fol._replace(m, k, new)
        out.append(fol._show_term(m, rng.random() < 0.8))
    return out


def _space_toggles(s):
    """s with one space deleted or inserted, at each position in turn (`f(x)`,
    `h.2 .1`, `fun h=>h`, `∀x,P x`, ...)."""
    return [s[:i] + s[i + 1:] if s[i:i + 1] == " " else s[:i] + " " + s[i:] for i in range(len(s) + 1)]


_PROMPT_EDITS = [("∧", "∨"), ("∨", "∧"), ("→", "↔"), ("↔", "→"), ("∀", "∃"), ("∃", "∀"), (" a", " b"), (" b", " a"),
                 ("¬", ""), ("(∀", "(¬∀"), (" P ", " Q "), ("Type", "Prop"), ("α", "β"), ("example", "theorem"),
                 (" :=", ""), (", ", ",\t"), (", ", ",  "), ("(P", "(Q"), ("x", "y"), (" x", " a"), ("Q R", "R Q"),
                 (" : ", "  : "), ("Prop)", "Prop) (q : Prop)"), ("(a : α)", "(a b : α)"), ("(α : Type)", "(α : Sort 1)")]


def _prompt_mutants(prompt, rng, n):
    out = []
    for _ in range(n):
        choice = rng.randrange(4)
        if choice == 0:
            a, b = rng.choice(_PROMPT_EDITS)
            spots = [m.start() for m in re.finditer(re.escape(a), prompt)]
            t = prompt
            if spots:
                k = rng.choice(spots)
                t = prompt[:k] + b + prompt[k + len(a):]
        elif choice == 1:
            t = _respelled(prompt, rng)
        elif choice == 2:  # mutate the formula
            k = prompt.index(") : ") + 4
            t = prompt[:k] + _mutate_once(prompt[k:-3], rng) + " :="
        else:
            t = _mutate_once(prompt, rng)
        out.append(t)
    return out


def _edge_cases():
    """Inputs at every limit: nesting, lengths in UTF-16 units, astral
    characters, deep formulas against deep proofs."""
    out = []
    head = "example (α : Type) (P : α → Prop) (p : Prop) (a : α) : "
    for n in (62, 63, 64, 65):
        f = "(" * n + "P a" + ")" * n + " ∨ p"
        out.append((head + f + " :=", "Or.inr"))
        out.append((head + "P a ∨ p :=", "Or.inl " + "(" * n + "_" + ")" * n))
        out.append(("example (α : Type) (p : Prop) : p → p :=", "fun h => " + "(" * n + "h" + ")" * n))
        out.append(("example (α : Type) (p : Prop) : p → p :=", "fun h => " + "(" * (n - 2) + "fun h => h" +
                    ")" * (n - 2) + " h"))
        out.append(("example (α : Type) (p : Prop) : p → p ∧ p :=",
                    "fun h => " + "⟨" * (n - 1) + "h" + ", h⟩" * (n - 1)))
    for n in (20, 30, 60, 64, 65, 66, 120):
        chain = " → ".join(["p"] * n)
        out.append((f"example (α : Type) (p : Prop) : {chain} :=", "fun " + " ".join(["h"] * (n - 1)) + " => h"))
        out.append((f"example (α : Type) (p : Prop) : {chain} :=",
                    "fun " + " ".join(f"h{k}" for k in range(n - 1)) + " => h0"))
        conj = " ∧ ".join(["p"] * n)
        out.append((f"example (α : Type) (p : Prop) : {conj} → p :=", "fun h => h" + ".2" * (n - 1)))
        out.append((f"example (α : Type) (p : Prop) : p → {conj} :=", "fun h => ⟨" + ", ".join(["h"] * n) + "⟩"))
        out.append((f"example (α : Type) (p : Prop) : {conj} → p :=",
                    "fun ⟨" + ", ".join(f"h{k}" for k in range(n)) + "⟩ => h0"))
        out.append((f"example (α : Type) (p : Prop) : {conj} → p :=",
                    "fun ⟨" + ", ".join(["_"] * (n - 1)) + ", h⟩ => h"))
    for n in (100, 200, 260, 266, 267, 268, 280):
        nots = "¬" * n
        out.append((f"example (α : Type) (p : Prop) : {nots}p → {nots}p :=", "fun h => h"))
        out.append((f"example (α : Type) (p : Prop) : {nots}p → {nots}p :=", "fun h h' => h h'"))
        out.append((f"example (α : Type) (p : Prop) : ({nots}p) :=", "trivial"))
    for n in range(1, 30, 4):
        f = "∀ x, P x ∧ " * n + "p"
        out.append((f"example (α : Type) (P : α → Prop) (p : Prop) : ({f}) → p ∨ p :=", "fun h => Or.inl h"))
        quant = "".join("∀ x" + "'" * k + ", " for k in range(n)) + "P x"
        out.append((f"example (α : Type) (P : α → Prop) : {quant} :=", "fun x => x"))
    ok = ("example (α : Type) (P : α → Prop) (a : α) : (∀ x, P x) → P a :=", "fun h => h a")
    for pad in range(385, 392):  # answers around 400 UTF-16 units
        out.append((ok[0], ok[1] + " " * pad))
        out.append((ok[0], ok[1] + " " * (pad - 2) + "𝔸"))
    for pad in range(225, 240):  # prompts around 300
        spaced = ok[0].replace("→ P a", "→" + " " * pad + "P a")
        out.append((spaced, ok[1]))
        out.append((spaced.replace("P a :=", "P a 𝔸 :="), ok[1]))
    out += [(ok[0], ""), (ok[0], " "), ("", ok[1]), (ok[0] + "\n", ok[1]), (ok[0], ok[1] + "\n"),
            (ok[0].replace(" :=", " := "), ok[1]), (ok[0], "fun h => h a -- done"), (ok[0], "fun h => h a\t"),
            (ok[0].replace("(a : α)", "(a : α) "), ok[1]), (ok[0], "fun h => h a　"),
            ("example (α : Type) (P : α → Prop) (a : α) : (∀ x, P x) → P a :=\n", ok[1]),
            ("example (α : Type) (PQ : α → Prop) (a : α) : (∀ x, PQ x) → PQ a :=", ok[1]),
            ("example (α : Type) (Q P : α → Prop) (a : α) : (∀ x, P x ∧ Q x) → P a :=", "fun h => (h a).1"),
            ("example (α : Type) (P : α → Prop) (a : α) : (∀ a, P a) → P a :=", "fun h => h a"),
            ("example (α : Type) (P : α → Prop) (a : α) : (∀ p, P p) → P a :=", "fun h => h a"),
            ("example (α : Type) (P : α → Prop) (p : Prop) (a : α) : (∀ p, P p) → P a ∨ p :=", "fun h => Or.inl (h a)"),
            ("example (α : Type) (P : α → Prop) (a : α) : (∀ x, ∀ x, P x) → P a :=", "fun h => h a a"),
            ("example (α : Type) (P : α → Prop) (a : α) : P a → True :=", "fun _ => trivial"),
            ("example (α : Type) (P : α → Prop) (a : α) : P a → True :=", "fun _ => trivial.elim"),
            ("example (α : Type) (P : α → Prop) (a : α) : False → P a :=", "fun h => h.elim"),
            ("example (α : Type) (P : α → Prop) (a : α) : False → P a :=", "False.elim"),
            ("example (α : Type) (P : α → Prop) (a : α) : P a → P a :=", "fun constructor => constructor"),
            ("example (α : Type) (P : α → Prop) (a : α) : P a → P a :=", "fun __proto__ => __proto__"),
            ("example (α : Type) (P : α → Prop) (a : α) : P a → P a :=", "fun h => toString.h")]
    out += _SEARCH_CORNERS
    return out


# Shortest proofs the search once missed (test_search_corner_cases): ex falso
# from a case analysis, which `.elim` cannot take, and a pattern binding one
# formula twice, whose first copy is unused.
_SEARCH_CORNERS = [
    ("example (α : Type) (P : α → Prop) (b : α) : (∀ x, P x ∨ P x) → ¬P b → P b :=",
     "fun h0 h1 => absurd h0 (fun _ => (h0 b).elim h1 h1)"),
    ("example (α : Type) (r : α → α → Prop) (a : α) : (∀ x, r a x ∨ r x x) → (∀ x, ¬r a x) → ∀ x, r x x :=",
     "fun h0 h1 => absurd h0 (fun _ => (h0 a).elim (h1 a) (h1 a))"),
    ("example (α : Type) (R : α → Prop) : (∀ x, R x → ¬R x) → ¬∃ x, R x ∧ R x :=",
     "fun h0 ⟨x, _, h1⟩ => h0 x h1 h1"),
    ("example (α : Type) (R : α → Prop) : (∃ x, R x ∧ R x) → (∃ x, R x) ∧ ∃ x, R x :=",
     "fun ⟨x, _, h0⟩ => ⟨⟨x, h0⟩, x, h0⟩"),
]


def test_search_corner_cases():
    for prompt, shortest in _SEARCH_CORNERS:
        goal, individuals = fol._goal_of(prompt)
        assert DOMAIN.verify(prompt, shortest)
        assert fol._shortest(goal, individuals, 100) == shortest
        assert fol._shortest(goal, individuals, len(shortest) - 1) is None


@pytest.fixture(scope="module")
def corpus():
    """(prompt, answer, kind) for 600 fresh problems: kind "good" for the
    reference and up to three document variants, "mutant" for mutated answers
    (of the reference and of the variants: textual, structural, spacing) and
    for mutated prompts."""
    rng = random.Random(5)
    out = []
    for p in _problems(600, seed=5, unique=True):
        docs = sorted({_answer(p, DOMAIN.document(p, rng)) for _ in range(3)} - {p.reference})
        out += [(p.prompt, a, "good") for a in [p.reference] + docs]
        out += [(p.prompt, m, "mutant") for m in _mutants(p.reference, rng, 18)]
        out += [(p.prompt, m, "mutant") for d in docs for m in _mutants(d, rng, 2)]
        out += [(p.prompt, m, "mutant") for a in [p.reference] + docs for m in _tree_mutants(p.prompt, a, rng, 2)]
        out += [(p.prompt, m, "mutant") for m in rng.sample(_space_toggles(p.reference), 3)]
        out += [(m, p.reference, "mutant") for m in _prompt_mutants(p.prompt, rng, 6)]
        k = p.prompt.index(") : ") + 4  # where the formula starts
        out += [(p.prompt[:k] + m + " :=", p.reference, "mutant") for m in rng.sample(_space_toggles(p.prompt[k:-3]), 2)]
    return out


def _python_verdicts(cases):
    """What LeanCheck.verify returns: null for a prompt no domain claims (the
    prompts here are fol headers or claimed by no domain at all)."""
    return [DOMAIN.verify(p, a) if DOMAIN.claims(p) else None for p, a in cases]


def test_matches_javascript(node, corpus):
    cases = [(p, a) for p, a, _ in corpus] + _edge_cases()
    rng = random.Random(9)
    alphabet = TOKENIZER.alphabet + "𝔸"
    for p, _ in random.Random(4).sample(cases, 2000):  # answers of random characters
        cases.append((p, "".join(rng.choice(alphabet) for _ in range(rng.randrange(30)))))
    py = _python_verdicts(cases)
    js = js_verify(cases)
    bad = [(c, x, y) for c, x, y in zip(cases, py, js) if x != y]
    print(f"\nfol vs web/checker.js: {len(cases)} cases, {sum(v is True for v in py)} accepted, "
          f"{sum(v is None for v in py)} unclaimed, {len(bad)} disagreements")
    assert len(cases) >= 20000
    assert not bad, bad[:10]


def _renamed(t, mapping):
    """The proof term t with binders and variables renamed by `mapping`."""
    tag = t[0]
    if tag == "lam":
        def binder(b):
            return mapping.get(b, b) if isinstance(b, str) else ("pat", [binder(x) for x in b[1]])
        return ("lam", binder(t[1]), _renamed(t[2], mapping))
    if tag == "var":
        return ("var", mapping.get(t[1], t[1]))
    if tag == "proj":
        return ("proj", _renamed(t[1], mapping), t[2])
    if tag == "app":
        return ("app", _renamed(t[1], mapping), [_renamed(a, mapping) for a in t[2]])
    return (tag, t[1], [_renamed(a, mapping) for a in t[2]]) if tag == "const" else \
        ("anon", [_renamed(a, mapping) for a in t[1]])


def test_references_are_shortest(corpus, monkeypatch):
    """No answer the checker accepts in the corpus (document variants and
    mutants) is shorter than the reference once its binders are named as the
    reference's are: hypotheses h0, h1, ..., individuals x, y, z, unused `_`.
    Which binders are individuals is recorded while the checker runs."""
    individuals, hypotheses = set(), set()
    new_individual, bind_hyp = fol._new_individual, fol._bind_hyp

    def record_individual(ctx, name, body):
        individuals.add(name)
        return new_individual(ctx, name, body)

    def record_hyp(ctx, binder, ty):
        if isinstance(binder, str):
            hypotheses.add(binder)
        return bind_hyp(ctx, binder, ty)

    monkeypatch.setattr(fol, "_new_individual", record_individual)
    monkeypatch.setattr(fol, "_bind_hyp", record_hyp)
    references = {}
    for p, a, kind in corpus:
        if kind == "good":
            references.setdefault(p, a)  # each problem's first answer is its reference
    checked = 0
    for p, a, _ in corpus:
        if p not in references:  # a mutated prompt
            continue
        individuals.clear()
        hypotheses.clear()
        # Skip answers that use a name for both kinds or shadow a declared individual.
        if not DOMAIN.verify(p, a) or individuals & hypotheses or (individuals | hypotheses) & {"a", "b"}:
            continue
        # _rename tells hypotheses from individuals by the prefix "h".
        mapping = {n: "h" + n for n in hypotheses} | {n: "x" + n for n in individuals}
        mapping.pop("_", None)
        canonical = fol._show_term(fol._rename(_renamed(fol._parse_term(a, ["a", "b"]), mapping)))
        assert DOMAIN.verify(p, canonical), (p, a, canonical)
        assert len(canonical) >= len(references[p]), (p, references[p], a)
        checked += 1
    print(f"\nfol: {checked} accepted answers, none shorter than its reference")
    assert checked > 3000


def test_never_raises():
    rng = random.Random(6)
    weird = ["\ud800", "\udfff", "𝔸", "\x00", "⟨", "⟩", "fun", " => ", ".", "∀", "∃"]
    ok = "example (α : Type) (P : α → Prop) (a : α) : (∀ x, P x) → P a :="
    for _ in range(3000):
        s = "".join(rng.choice(weird) for _ in range(rng.randrange(12)))
        assert DOMAIN.verify(ok, s) is False or DOMAIN.verify(ok, s) is True
        assert DOMAIN.verify(ok[:-3] + s + " :=", "fun h => h a") in (True, False)
        assert DOMAIN.claims(s) is False and DOMAIN.split_key(s) == s
    for p, a in _edge_cases():
        assert DOMAIN.verify(p, a) in (True, False)
    assert DOMAIN.verify(None, "fun h => h a") is False and DOMAIN.verify(ok, None) is False
    assert DOMAIN.claims(None) is False


def test_never_raises_on_a_nearly_full_stack():
    """The deepest prompts within the limits, at every depth of the caller's
    stack: claims and verify turn a RecursionError into a rejection, and
    split_key cannot meet one after parsing (printing recurses less)."""
    head = "example (α : Type) (q : Prop) : "  # q, not p, so that split_key renames
    deep = [head + "¬" * 262 + "q :=", head + "(" * 64 + "q" + ")" * 64 + " → q :=", head + " → ".join(["q"] * 66) + " :=",
            "example (α : Type) (Q : α → Prop) : " + "∀ y, Q y ∧ " * 23 + "True :="]
    assert all(DOMAIN.claims(p) for p in deep)
    escaped = []

    def call(p):
        try:
            DOMAIN.claims(p), DOMAIN.verify(p, "fun h => h"), DOMAIN.split_key(p)
        except RecursionError:
            escaped.append(p)

    def at_depth(n, p):
        return call(p) if n == 0 else at_depth(n - 1, p)

    for depth in range(0, sys.getrecursionlimit(), 10):
        for p in deep:
            try:
                at_depth(depth, p)
            except RecursionError:  # the helper itself ran out of stack
                pass
    assert not escaped


# ---------------------------------------------------------------- Lean

def test_lean_accepts_what_the_checker_accepts(lean, corpus):
    good = sorted({f"{p} {a}" for p, a, kind in corpus if kind == "good"})
    cases = [(p, a) for p, a, kind in corpus if kind == "mutant"] + _edge_cases()
    verdicts = _python_verdicts(cases)
    accepted = sorted({f"{p} {a}" for (p, a), v in zip(cases, verdicts) if v} - set(good))
    rejected = sorted({f"{p} {a}" for (p, a), v in zip(cases, verdicts) if v is False})
    rejected = random.Random(0).sample(rejected, min(2000, len(rejected)))
    start = time.perf_counter()
    ok = lean_oracle.check_in_chunks(good + accepted + rejected, lean, chunk=2000)
    seconds = time.perf_counter() - start
    n_good, n_accepted = len(good), len(good) + len(accepted)
    ok_good, ok_accepted, ok_rejected = ok[:n_good], ok[n_good:n_accepted], ok[n_accepted:]
    lean_only = [d for d, o in zip(rejected, ok_rejected) if o]
    n_problems = len({p for p, _, kind in corpus if kind == "good"})
    print(f"\nfol vs Lean ({seconds:.0f}s): {sum(ok_good)}/{len(good)} references and documents of {n_problems} "
          f"problems compile; {sum(ok_accepted)}/{len(accepted)} checker-accepted mutants compile; Lean accepts "
          f"{len(lean_only)}/{len(rejected)} checker-rejected mutants")
    for d in lean_only[:8]:
        print("   Lean only:", d.replace("\n", "\\n"))
    assert n_problems >= 300
    assert all(ok_good), [d for d, o in zip(good, ok_good) if not o][:10]
    assert all(ok_accepted), [d for d, o in zip(accepted, ok_accepted) if not o][:10]


# ---------------------------------------------------------------- exotic prompts

_SPELLING_NAMES = ["x", "y", "z", "w", "a", "b", "p", "q", "r", "x'", "u"]


def _random_formula(rng, depth, n_bound, symbols, individuals):
    """A random formula over `symbols`; bound variables are indices below n_bound."""
    terms = list(range(n_bound)) + individuals
    r = rng.random()
    if depth == 0 or r < 0.25:
        options = [s for s in symbols if fol.ARITY[s] == 0 or terms]
        if not options or rng.random() < 0.07:
            return rng.choice((fol.TOP, fol.BOT))
        s = rng.choice(options)
        return ("pred", s, tuple(rng.choice(terms) for _ in range(fol.ARITY[s])))
    if r < 0.4:
        return ("not", _random_formula(rng, depth - 1, n_bound, symbols, individuals))
    if r < 0.6:
        body = _random_formula(rng, depth - 1, n_bound + 1, symbols, individuals)
        if fol._mentions(body):
            return (rng.choice(("all", "ex")), body)
    op = rng.choice(("and", "or", "imp", "imp", "iff"))
    return (op, _random_formula(rng, depth - 1, n_bound, symbols, individuals),
            _random_formula(rng, depth - 1, n_bound, symbols, individuals))


def _gap(rng):
    return rng.choice(["", " ", " ", " ", "  "])


def _spelled(f, rng, bound, prec=0, tail=False):
    """f printed unlike fol._show: bound variables named anything, shadowing
    outer ones or a declared individual (the caller rejects a spelling that
    captures), redundant parentheses, no or double spaces between tokens."""
    if rng.random() < 0.12:
        return "(" + _spelled(f, rng, bound) + ")"
    tag = f[0]
    if tag == "pred":
        args = [bound[len(bound) - 1 - x] if isinstance(x, int) else x for x in f[2]]
        return rng.choice([" ", "  "]).join([f[1]] + args)
    if tag in ("top", "bot"):
        return "True" if tag == "top" else "False"
    if tag == "not":
        return "¬" + _gap(rng) * (rng.random() < 0.3) + _spelled(f[1], rng, bound, 40, tail)
    if tag in ("all", "ex"):
        name = rng.choice(_SPELLING_NAMES + bound[-2:])
        s = f"{'∀' if tag == 'all' else '∃'}{_gap(rng) or ' '}{name},{_gap(rng)}{_spelled(f[1], rng, bound + [name])}"
        return f"({s})" if tail else s
    p = _PREC[tag]
    if p < prec:
        return "(" + _spelled(f, rng, bound) + ")"
    left = _spelled(f[1], rng, bound, p + 1, True)
    right = _spelled(f[2], rng, bound, p + 1 if tag == "iff" else p, tail)
    return f"{left}{_gap(rng)}{_SYMBOL[tag]}{_gap(rng)}{right}"


def _parenthesised(f, level=0):
    """f with every subformula in parentheses and bound variables v0, v1, ...
    by level: a spelling Lean cannot read two ways."""
    tag = f[0]
    if tag == "pred":
        return "(" + " ".join([f[1]] + [f"v{level - 1 - x}" if isinstance(x, int) else x for x in f[2]]) + ")"
    if tag in ("top", "bot"):
        return "True" if tag == "top" else "False"
    if tag == "not":
        return f"(¬{_parenthesised(f[1], level)})"
    if tag in ("all", "ex"):
        return f"({'∀' if tag == 'all' else '∃'} v{level}, {_parenthesised(f[1], level + 1)})"
    return f"({_parenthesised(f[1], level)} {_SYMBOL[tag]} {_parenthesised(f[2], level)})"


def _exotic(rng):
    """(prompt, goal): a provable formula over random parts A, B (closed) and
    C (one loose variable), each occurrence spelled on its own, or None if a
    spelling captured a variable or the prompt is too long."""
    symbols = rng.sample(list(fol.ARITY), rng.randrange(1, 5))
    individuals = rng.sample(list(fol.INDIVIDUALS), rng.randrange(0, 3))
    a = _random_formula(rng, rng.randrange(0, 4), 0, symbols, individuals)
    b = _random_formula(rng, rng.randrange(0, 3), 0, symbols, individuals)
    c = _random_formula(rng, rng.randrange(1, 4), 1, symbols, individuals)
    if not fol._mentions(c):
        return None
    goals = [("imp", a, a), ("imp", ("and", a, b), ("and", b, a)), ("imp", ("or", a, b), ("or", b, a)), ("iff", a, a),
             ("imp", ("iff", a, b), ("iff", b, a)), ("imp", a, ("not", ("not", a))), ("imp", ("and", a, ("not", a)), b),
             ("all", ("imp", c, c)), ("imp", ("ex", c), ("not", ("all", ("not", c)))),
             ("imp", ("all", ("imp", c, a)), ("imp", ("ex", c), a)), ("imp", ("and", ("all", c), a), ("all", ("and", c, a)))]
    if individuals:
        w = rng.choice(individuals)
        goals += [("imp", ("all", c), fol._open(c, w)), ("imp", fol._open(c, w), ("ex", c))]
    goal = rng.choice(goals)
    declaration = fol._declaration(goal)
    prompt = f"{declaration[:declaration.index(') : ') + 1]} : {_spelled(goal, rng, [])} :="
    parsed = fol._goal_of(prompt)
    return (prompt, goal) if parsed is not None and parsed[0] == goal else None


def test_exotic_prompts(lean, node):
    """Prompts spelled unlike the generator's. Lean must read each formula as
    the checker does, the split key must ignore the spelling, and proofs the
    search finds, documents of them and mutants of both must compile whenever
    the checker accepts them."""
    rng = random.Random(8)
    exotic = []
    while len(exotic) < 400:
        got = _exotic(rng)
        if got is not None:
            exotic.append(got)
    readings, good, cases = [], [], []
    for prompt, goal in exotic:
        assert DOMAIN.split_key(prompt) == DOMAIN.split_key(fol._declaration(goal)), prompt
        k = prompt.index(") : ") + 1  # the end of the binders (a formula has no colon)
        readings.append(f"{prompt[:k]} (hyp : {_parenthesised(goal)}){prompt[k:]} hyp")
        try:
            proof = fol._shortest(goal, fol._goal_of(prompt)[1], 200)
        except fol._GiveUp:
            continue
        assert proof is not None, prompt
        p = fol.Problem("fol", prompt, proof)
        answers = sorted({proof} | {_answer(p, DOMAIN.document(p, rng)) for _ in range(2)})
        good += [(prompt, a) for a in answers]
        cases += [(prompt, m) for a in answers for m in _tree_mutants(prompt, a, rng, 3)]
        cases += [(prompt, m) for m in rng.sample(_space_toggles(proof), 3)]
    py = _python_verdicts(good + cases)
    assert py[:len(good)] == [True] * len(good)
    assert py == js_verify(good + cases)
    accepted = sorted({f"{q} {a}" for (q, a), v in zip(good + cases, py) if v})
    ok = lean_oracle.check_in_chunks(readings + accepted, lean, chunk=2000)
    print(f"\nfol exotic prompts: Lean reads {sum(ok[:len(readings)])}/{len(readings)} as the checker does; "
          f"{sum(ok[len(readings):])}/{len(accepted)} checker-accepted proofs, documents and mutants compile")
    assert all(ok[:len(readings)]), [d for d, o in zip(readings, ok) if not o][:5]
    assert all(ok[len(readings):]), [d for d, o in zip(accepted, ok[len(readings):]) if not o][:5]
