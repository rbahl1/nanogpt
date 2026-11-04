"""The prop domain: its generator and split keys, its checker against
web/checker.js (the specification: they must agree on every case, here
references, variants, many kinds of mutants, simulated tactic proofs and odd
inputs, except where Lean runs out of recursion and prop.py alone refuses),
and against Lean 4 (soundness: every answer the checker accepts compiles)."""

import functools
import json
import os
import random
import re
import subprocess
import sys
import time

from formal import prop
from formal.prop import DOMAIN
from js_checker import js_verify
import lean_oracle
from tokenizer import Tokenizer

N_PROBLEMS = 300
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@functools.lru_cache(maxsize=None)
def problems(n=N_PROBLEMS, seed=0) -> tuple:
    rng = random.Random(seed)
    return tuple(DOMAIN.sample(rng) for _ in range(n))


@functools.lru_cache(maxsize=None)
def variants() -> dict:
    return {p.prompt: DOMAIN.variants(p.prompt) for p in problems()}


# ---------------------------------------------------------------- generator

def test_references_verify_and_fit():
    tok = Tokenizer()
    for p in problems():
        assert DOMAIN.claims(p.prompt) and DOMAIN.verify(p.prompt, p.reference), p
        assert len(p.prompt) <= 120 and len(p.reference) <= 200, p
        ids, _ = tok.example_ids(p.prompt, p.reference)
        assert len(ids) <= 256, p
        # The header declares exactly the atoms used, alphabetically.
        atoms = prop.atoms_in_order(prop.goal_of(p.prompt))
        binders = re.match(r"example(?: \(([a-z ]+) : Prop\))? :", p.prompt).group(1)
        assert (binders.split() if binders else []) == sorted(atoms), p
        # Prompts print with the fewest parentheses: reprinting changes nothing.
        assert prop.header(prop.goal_of(p.prompt)) == p.prompt


def test_old_examples_are_in_the_language():
    old = {"example (p r s : Prop) : False → ¬r ∧ p ∧ s :=": None,
           "example (q : Prop) : q → ((q → q) ∧ q) ∧ q :=": None,
           "example (p : Prop) : p ∨ p ↔ p :=": None,
           "example (s : Prop) : ¬(s → s) → s ∧ s :=": None,
           "example (q : Prop) : (q → q → ¬q) → q → q → ¬q :=": "fun h0 => h0",
           "example (p q : Prop) : p ∧ q → q ∧ p :=": "fun h0 => ⟨h0.2, h0.1⟩"}
    for prompt, reference in old.items():
        assert prop.header(prop.goal_of(prompt)) == prompt
        best, _ = prop.solve(prop.unfold(prop.goal_of(prompt)))
        assert DOMAIN.verify(prompt, prop.show_proof(best))
        if reference:
            assert prop.show_proof(best) == reference
            assert DOMAIN.verify(prompt, reference)


_SAMPLE_IN_SUBPROCESS = """
import json, random
from formal.prop import DOMAIN
rng = random.Random(7)
ps = [DOMAIN.sample(rng) for _ in range(50)]
print(json.dumps([[p.prompt, p.reference, DOMAIN.document(p, rng)] for p in ps]))
"""


def test_generator_is_deterministic():
    assert list(problems(50, 7)) == list(problems.__wrapped__(50, 7))
    assert list(problems(50, 7)) != list(problems(50, 8))
    rng1, rng2 = random.Random(3), random.Random(3)
    assert [DOMAIN.document(p, rng1) for p in problems()[:100]] == [DOMAIN.document(p, rng2) for p in problems()[:100]]
    # Across processes too, whatever the string hash seed: data.py relies on it.
    rng = random.Random(7)
    here = [[p.prompt, p.reference, DOMAIN.document(p, rng)] for p in [DOMAIN.sample(rng) for _ in range(50)]]
    for hash_seed in ("1", "2"):
        out = subprocess.run([sys.executable, "-c", _SAMPLE_IN_SUBPROCESS], cwd=ROOT, capture_output=True, text=True,
                             check=True, env={**os.environ, "PYTHONHASHSEED": hash_seed})
        assert json.loads(out.stdout) == here


def _size(f) -> int:
    """Connectives in f, ¬ included."""
    return 0 if f[0] in ("atom", "top", "bot") else 1 + sum(_size(g) for g in f[1:])


def test_distribution():
    ps = problems(2000, 1)
    goals = [prop.goal_of(p.prompt) for p in ps]
    atoms = [len(prop.atoms_in_order(g)) for g in goals]
    counts = {k: atoms.count(k) / len(ps) for k in range(5)}
    assert counts[1] > 0.2 and counts[2] > 0.2 and counts[3] > 0.1 and 0.02 < counts[4] < 0.2, counts
    tactic = sum(p.reference.startswith("by") for p in ps) / len(ps)
    assert 0.02 < tactic < 0.3, tactic
    constants = sum("True" in p.prompt or "False" in p.prompt for p in ps) / len(ps)
    assert 0.05 < constants < 0.35, constants  # True and False occasionally
    for construct in ("Or.elim", "absurd", "Or.inl", "⟨", "fun h0 h1"):
        assert sum(construct in p.reference for p in ps) > 20, construct
    assert sum(g[0] == "iff" for g in goals) > 40
    assert len({DOMAIN.split_key(p.prompt) for p in ps}) > 0.8 * len(ps)
    sizes, lengths = sorted(_size(g) for g in goals), sorted(len(p.reference) for p in ps)
    print(f"prop: atoms {', '.join(f'{k}: {v:.0%}' for k, v in counts.items() if v)}; connectives median "
          f"{sizes[len(ps) // 2]}, 90th percentile {sizes[9 * len(ps) // 10]}, max {sizes[-1]}; reference length "
          f"mean {sum(lengths) / len(ps):.1f}, median {lengths[len(ps) // 2]}, max {lengths[-1]}; "
          f"tactic references {tactic:.1%}; True or False in {constants:.0%} of prompts")


def test_sample_speed():
    rng = random.Random(11)
    start = time.perf_counter()
    for _ in range(500):
        DOMAIN.sample(rng)
    per_sample = (time.perf_counter() - start) / 500
    print(f"prop: {per_sample * 1000:.2f} ms per sample")
    assert per_sample < 0.005


def test_documents_verify():
    rng = random.Random(5)
    forms = set()
    for p in problems():
        for _ in range(3):
            doc = DOMAIN.document(p, rng)
            assert doc.startswith(p.prompt + " "), doc
            answer = doc[len(p.prompt) + 1:]
            assert DOMAIN.verify(p.prompt, answer), doc
            assert len(doc) + 2 <= 256
            forms.add("term" if not answer.startswith("by") else "block" if "\n" in answer else "line")
    assert forms == {"term", "line", "block"}
    for prompt, answers in variants().items():
        assert all(DOMAIN.verify(prompt, a) for a in answers)


# ---------------------------------------------------------------- split keys

def _reparenthesise(f, rng) -> str:
    """f printed with random redundant parentheses and spacing, and ¬A
    sometimes spelled A → False."""
    tag = f[0]
    if tag == "atom":
        s = f[1]
    elif tag == "top":
        s = "True"
    elif tag == "bot":
        s = "False"
    elif tag == "not" and rng.random() < 0.3:
        s = "((" + _reparenthesise(f[1], rng) + ") → False)"
    elif tag == "not":
        s = "¬(" + _reparenthesise(f[1], rng) + ")"
    else:
        s = f"({_reparenthesise(f[1], rng)}) {prop.SYMBOL[tag]}{' ' * rng.randint(1, 2)}({_reparenthesise(f[2], rng)})"
    return f"({s})" if rng.random() < 0.3 else s


def test_split_key_invariance():
    assert DOMAIN.split_key("example (r s : Prop) : s → r → s :=") == \
        DOMAIN.split_key("example (p q : Prop) : p → q → p :=")
    assert DOMAIN.split_key("example (p q : Prop) : (p → q) → p :=") != \
        DOMAIN.split_key("example (p q : Prop) : p → q → p :=")
    assert DOMAIN.split_key("example (p q : Prop) : ((p) → (q → p)) :=") == \
        DOMAIN.split_key("example (p q : Prop) : p → q → p :=")
    # ¬A unfolds to A → False: one problem, two spellings.
    assert DOMAIN.split_key("example (p q : Prop) : (q → False) → p → q → p :=") == \
        DOMAIN.split_key("example (r s : Prop) : ¬r → s → r → s :=") == "example (p q : Prop) : ¬p → q → p → q :="
    assert DOMAIN.split_key("example : False → False :=") == DOMAIN.split_key("example : ¬False :=")
    assert DOMAIN.split_key("example (p q : Prop) : p → q :=") != DOMAIN.split_key("example (p : Prop) : p → False :=")
    assert DOMAIN.split_key("not a prompt") == "not a prompt"
    rng = random.Random(2)
    names = list("abcdefgpqrsuvwxyz") + ["foo", "bar", "pp"]
    for p in problems():
        key = DOMAIN.split_key(p.prompt)
        assert DOMAIN.claims(key) and DOMAIN.split_key(key) == key
        f = prop.goal_of(p.prompt)
        atoms = prop.atoms_in_order(f)
        fresh = rng.sample(names, len(atoms))
        renamed = prop._rename(f, dict(zip(atoms, fresh)))
        variants = [prop.header(renamed), f"example ({' '.join(reversed(fresh))} : Prop) : {prop.show_formula(renamed)} :="
                    if fresh else prop.header(renamed),
                    "example" + (f" ({' '.join(sorted(atoms))} : Prop)" if atoms else "") + f" : {_reparenthesise(f, rng)} :="]
        for v in variants:
            assert DOMAIN.claims(v), v
            assert DOMAIN.split_key(v) == key, (v, p.prompt)


def test_split_keys_never_collide():
    """A key parses back to the canonical form of its problem (atoms renamed in
    order of first occurrence, A → False folded to ¬A), so two different
    problems never share a key: here generated prompts and arbitrary formulas
    of up to six atoms, True and False included."""
    rng = random.Random(4)
    prompts = [p.prompt for p in problems(2000, 3)]
    prompts += [prop.header(prop._random_formula(rng, rng.sample("pqrstu", rng.randint(1, 6)), rng.randint(0, 9)))
                for _ in range(4000)]
    seen = {}
    for prompt in prompts:
        f = prop.goal_of(prompt)
        rename = {a: prop._CANONICAL[k] for k, a in enumerate(prop.atoms_in_order(f))}
        canonical = prop._rename(prop._fold(f), rename)
        key = DOMAIN.split_key(prompt)
        assert prop.goal_of(key) == canonical, (prompt, key)
        assert seen.setdefault(key, canonical) == canonical
    assert len(seen) > 0.7 * len(prompts)


# ---------------------------------------------------------------- the checker

# Tactic proofs that exercise the rarer paths of the tactic checker.
HANDWRITTEN = [
    ("example (p q : Prop) : (p ↔ q) → p → q :=", "by\n  intro h0 h1\n  cases h0 with\n  | intro a b =>\n    apply a\n    exact h1"),
    ("example (p q : Prop) : (p ↔ q) → p → q :=", "by\n  intro h0\n  cases h0 with\n  | intro a b => apply a"),
    ("example (p q : Prop) : (p ↔ q) → p → q :=", "by\n  intro h0\n  cases h0 with\n  | intro a b => exact a"),
    ("example (p q : Prop) : (p ↔ q) → p → q :=", "by intro h0; exact h0.mp"),
    ("example (p q : Prop) : (p ↔ q) → p → q :=", "by intro h0; apply h0.1"),
    ("example (p q : Prop) : (p ↔ q) → q → p :=", "by intro h0 h1; apply h0.2; exact h1"),
    ("example (p q : Prop) : (p ↔ (q → p)) → q → p → p :=", "by\n  intro h0\n  cases h0 with\n  | intro a b =>\n    intro h1 h2\n    apply b\n    · intro h3\n      exact h2"),
    ("example (p q : Prop) : (p → q → p) → p → q → p :=", "by intro h0 h1 h2; apply h0; exact h1; exact h2"),
    ("example (p q : Prop) : (p → q → p) → p → q → p :=", "by intro h0 h1; apply h0; exact h1"),
    ("example (p q : Prop) : (p → q → p) → p → q → p :=", "by intro h0 h1; apply h0 h1"),
    ("example (p : Prop) : False → p :=", "by apply False.elim"),
    ("example (p : Prop) : ¬p → p → False :=", "by intro h0 h1; apply False.elim; exact h0 h1"),
    ("example (p q : Prop) : ¬p → p → q :=", "by intro h0 h1; apply False.elim; exact h0 h1"),
    ("example (p q : Prop) : p → p ∨ q :=", "by intro h0; apply Or.inl; exact h0"),
    ("example (p q : Prop) : q → p ∨ q :=", "by intro h0; apply Or.inr; assumption"),
    ("example (p q : Prop) : p → q → p ∧ q :=", "by intro h0 h1; apply And.intro; exact h0; exact h1"),
    ("example (p q : Prop) : p → q → p ∧ q :=", "by\n  intro h0 h1\n  apply And.intro\n  · exact h0\n  · exact h1"),
    ("example (p : Prop) : p ↔ p :=", "by apply Iff.intro; intro h0; exact h0; intro h0; exact h0"),
    ("example (p : Prop) : p ↔ p :=", "by constructor <;> intro h0 <;> exact h0"),
    ("example : True :=", "by constructor"),
    ("example : True ∧ True :=", "by constructor; constructor; trivial"),
    ("example (p q : Prop) : p ∨ q → q ∨ p :=", "by\n  intro h0\n  cases h0 with\n  | inr b => left; exact b\n  | inl a => right; exact a"),
    ("example (p q : Prop) : p ∨ q → q ∨ p :=", "by\n  intro h0\n  cases h0 with\n  | inl a => right; exact a\n  | inl b => left; exact b"),
    ("example (p q : Prop) : p ∨ q → q ∨ p :=", "by intro h0; rcases h0 with a | b; right; exact a; left; exact b"),
    ("example (p q : Prop) : p ∨ q → q ∨ p :=", "by intro h0; obtain a | a := h0; exact Or.inr a; exact Or.inl a"),
    ("example (p q : Prop) : p ∨ q → q ∨ p :=", "by intro h0; cases h0 with | inl a => exact Or.inr a | inr a => exact Or.inl a"),
    ("example (p q r : Prop) : (p ∧ q) ∨ r → r ∨ p :=", "by intro h0; rcases h0 with ⟨a, b⟩ | c; right; exact a; left; exact c"),
    ("example (p q r : Prop) : (p ∧ q) ∨ r → r ∨ p :=", "by intro h0; rcases h0 with (⟨a, b⟩ | c); right; exact a; left; exact c"),
    ("example (p q r : Prop) : p ∧ q ∧ r → r :=", "by intro h0; obtain ⟨a, b, c⟩ := h0; exact c"),
    ("example (p q r : Prop) : p ∧ q ∧ r → r :=", "by intro h0; obtain ⟨a, ⟨b, c⟩⟩ := h0; exact c"),
    ("example (p q r : Prop) : p ∧ q ∧ r → r :=", "by intro h0; rcases h0 with ⟨a, a, c⟩; exact c"),
    ("example (p q r : Prop) : p ∧ q ∧ r → r :=", "by intro h0; cases h0 with | intro a b => exact b.2"),
    ("example (p q : Prop) : p ∧ q → p :=", "by intro h0; cases h0 with | intro a b => exact h0.1"),
    ("example (p q : Prop) : p ∧ q → p :=", "by intro h0; obtain ⟨h0, h1⟩ := h0; exact h0"),
    ("example (p q : Prop) : p → p ∧ q → p :=", "by intro h h; obtain ⟨a, b⟩ := h; exact h"),
    ("example (p q : Prop) : p → p ∧ q → p :=", "by intro h h; obtain ⟨a, b⟩ := h; exact a"),
    ("example (p q : Prop) : (p ↔ q) → q → p :=", "by intro h0 h1; obtain ⟨a, b⟩ := h0; exact b h1"),
    ("example (p q : Prop) : False → p :=", "by intro h0; cases h0"),
    ("example (p q : Prop) : ¬p → p → q :=", "by intro h0 h1; cases h0 h1"),
    ("example (p q : Prop) : ¬p → p → q :=", "by intro h0 h1; contradiction"),
    ("example (p q : Prop) : (p → False) → p → q :=", "by intro h0 h1; contradiction"),
    ("example (p q : Prop) : ¬p → p → q :=", "by intro h0 h1; exfalso; exact h0 h1"),
    ("example (p q : Prop) : ¬p → p → q :=", "by intro h0 h1; exact absurd h1 h0"),
    ("example (p : Prop) : p → p ∧ p ∧ p :=", "by intro h0; trivial"),
    ("example (p : Prop) : p → p ∧ p ∧ p :=", "by intro h0; constructor; assumption; constructor; assumption; assumption"),
    ("example (p : Prop) : p → p ∧ p :=", "by\n  intro h0\n  constructor\n  · exact h0\n  · exact h0"),
    ("example (p : Prop) : p → p ∧ p :=", "by\n  intro h0\n  constructor\n  · exact h0\n   · exact h0"),
    ("example (p : Prop) : p → p ∧ p :=", "by\n  intro h0\n  constructor\n  · exact h0\n· exact h0"),
    ("example (p : Prop) : p → p ∧ p :=", "by\n  intro h0\n  constructor\n  exact h0\n  exact h0"),
    ("example (p : Prop) : p → p ∧ p :=", "by\n  intro h0\n  constructor\n   exact h0\n  exact h0"),
    ("example (p : Prop) : p → p ∧ p :=", "by\n  intro h0\n  constructor\n  · exact h0\n  · exact h0\n  · exact h0"),
    ("example (p : Prop) : p → p ∧ p :=", "by intro h0\n  constructor; exact h0; exact h0"),
    ("example (p : Prop) : p → p ∧ p :=", "by\n intro h0\n constructor; exact h0; exact h0"),
    ("example (p : Prop) : p → p ∧ p :=", "by\n  intro h0;\n  exact ⟨h0, h0⟩"),
    ("example (p : Prop) : p → p ∧ p :=", "by\n  intro h0\n  exact ⟨h0,\n    h0⟩"),
    ("example (p : Prop) : p → p :=", "by\n\n  intro h0\n\n  exact h0"),
    ("example (p : Prop) : p → p :=", "by intro h0; exact h0;"),
    ("example (p : Prop) : p → p :=", "by intro h0; · exact h0"),
    ("example (p : Prop) : p → p :=", "by · intro h0; exact h0"),
    ("example (p : Prop) : p → p :=", "by intro h0;; exact h0"),
    ("example (p : Prop) : p → p :=", "by intro left; exact left"),
    ("example (p : Prop) : p → p :=", "by intro h'; exact h'"),
    ("example (p : Prop) : p → p :=", "fun rfl => rfl"),
    ("example (p : Prop) : p → p :=", "fun id => id"),
    ("example (p : Prop) : p → p :=", "fun _ => _"),
    ("example (p : Prop) : p → p :=", "fun x => (x)"),
    ("example (p : Prop) : p → p :=", "fun x=>x"),
    ("example (p : Prop) : p → p :=", "fun(x) => x"),
    ("example (p : Prop) : p → p :=", "(fun x => x)"),
    ("example (p q : Prop) : p → q → p :=", "fun x _ => x"),
    ("example (p q : Prop) : p → q → p :=", "fun x x => x"),
    ("example (p : Prop) : p → p → p :=", "fun x x => x"),
    ("example (p q : Prop) : p ∧ q → q :=", "fun h => And.right h"),
    ("example (p q : Prop) : p ∧ q → q :=", "fun h => (And.intro h.1 h.2).2"),
    ("example (p q : Prop) : p ∧ q → q ∧ p :=", "fun h => And.intro h.2 h.1"),
    ("example (p q : Prop) : p ∧ q → q ∧ p :=", "fun h => And.intro h.right h.left"),
    ("example (p q : Prop) : p ∧ q → q ∧ p :=", "fun h => And.intro(h.2) h.1"),
    ("example (p q : Prop) : (p ↔ q) → (q ↔ p) :=", "fun h => Iff.intro h.2 h.1"),
    ("example (p q : Prop) : (p ↔ q) → (q ↔ p) :=", "fun h => (Iff.intro h.2 h.1)"),
    ("example (p q : Prop) : (p ↔ q) → (q ↔ p) :=", "fun h => ⟨h.mpr, h.mp⟩"),
    ("example (p q : Prop) : (p ↔ q) → q → p :=", "fun h => (Iff.intro h.2 h.1).1"),
    ("example (p q r : Prop) : (p → r) → (q → r) → p ∨ q → r :=", "fun f g h => Or.elim h f g"),
    ("example (p q r : Prop) : (p → r) → (q → r) → p ∨ q → r :=", "fun f g h => (Or.elim h f g)"),
    ("example (p q : Prop) : p ∨ q → (p → q) → q :=", "fun h f => Or.elim h f (fun x => x)"),
    ("example (p q : Prop) : p ∨ q → (p → q) → q :=", "fun h f => (Or.elim h f (fun x => x))"),
    ("example (p q : Prop) : p ∨ q → (p → q) → q ∧ q :=", "fun h f => ⟨Or.elim h f fun x => x, Or.elim h f id⟩"),
    ("example (p : Prop) : p → ¬¬p :=", "fun h n => n h"),
    ("example (p : Prop) : p → ¬¬p :=", "fun h n => absurd h n"),
    ("example (p : Prop) : ¬¬(p ∨ ¬p) :=", "fun h => h (Or.inr fun x => h (Or.inl x))"),
    ("example (p : Prop) : ¬¬(p ∨ ¬p) :=", "fun h => h (Or.inr (fun x => h (Or.inl x)))"),
    ("example : True :=", "trivial"),
    ("example : True :=", "trivial.1"),
    ("example : ¬False :=", "fun h => h"),
    ("example : ¬False :=", "fun h => False.elim h"),
    ("example : True ∧ True :=", "⟨trivial, trivial⟩"),
    ("example : True ∧ True :=", "⟨trivial,trivial⟩"),
    ("example : True ∧ True :=", "⟨trivial , trivial⟩"),
    ("example : True ∧ True :=", "⟨trivial⟩"),
    ("example : True ∧ True ∧ True :=", "⟨trivial, trivial, trivial⟩"),
    ("example : True ∧ True ∧ True :=", "⟨trivial, ⟨trivial, trivial⟩⟩"),
    ("example : (True ∧ True) ∧ True :=", "⟨trivial, trivial, trivial⟩"),
    ("example (p q : Prop) : p ∧ q → q :=", "fun h => h.2"),
    ("example (p q : Prop) : p ∧ q → q :=", "fun h => (h).2"),
    ("example (p q : Prop) : p ∧ q → q :=", "fun h => (h) .2"),
    ("example (p q : Prop) : p ∧ q → q :=", "fun h => h.right"),
    ("example (p q : Prop) : p ∧ q → q :=", "fun h => h.mpr"),
    ("example (p q : Prop) : p ∧ q → q :=", "fun h => h.3"),
    ("example (p q : Prop) : p ∧ (p → q) → q :=", "fun h => h.2 h.1"),
    ("example (p q : Prop) : p ∧ (p → q) → q :=", "fun h => (h.2 h.1)"),
    ("example (p q : Prop) : p ∧ (p → q) → q :=", "fun h => h.2(h.1)"),
    ("example (p q : Prop) : p ∧ (p → q) → q ∧ q :=", "fun h => ⟨h.2 h.1, (h.2 h.1)⟩"),
    ("example (p q : Prop) : (p ∧ (p → q)) ∧ p → q :=", "fun h => (h.1.2 h.2)"),
    ("example (p q : Prop) : (p → q ∧ p) → p → p :=", "fun f x => (f x).2"),
    ("example (p q : Prop) : (p → q ∧ p) → p → p :=", "fun f x => (f x).2.1"),
    ("example (p q : Prop) : (p → q ∧ p) → p → p :=", "fun f x => ⟨f x, x⟩.2"),
    ("example (p : Prop) : p → p :=", "by\n  intro h0\n  exact h0\n"),
    ("example (p : Prop) : p → p :=", "by  intro h0  ;  exact h0"),
    ("example (p : Prop) : p → p :=", "by intro h0\texact h0"),
    ("example (p : Prop) : p → p :=", "by intro h0\r\n  exact h0"),
]


def _tokens(s: str) -> list:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_'.]*|\s+|.", s)


_SNIPPETS = [" ", "  ", "\n", "\n  ", "\n    ", "(", ")", "⟨", "⟩", ",", ".", ".1", ".2", ".mp", ".left", "h0",
             "h1", "h2", "_", "'", "fun", "fun h3 =>", "=>", ";", "; ", "|", "· ", ":=", "with", "intro",
             "exact", "apply", "trivial", "absurd", "Or.inl", "Or.inr", "Or.elim", "False.elim", "And.intro",
             "Iff.intro", "And.left", "constructor", "left", "right", "cases", "rcases", "obtain", "exfalso",
             "assumption", "contradiction", "inl", "inr", "\t", "\r", "𝔸", "¬", "→", "x", "by ", "<;>", "0"]
_NAMES = ["h0", "h1", "h2", "h9", "h10", "_", "x", "h'", "fun", "in", "at", "left", "rfl", "trivial", "absurd",
          "Or", "True", "intro", "inl", "id", "h.1", "x1", "α", "p", "q", "this", "end", "h_in", "And"]
_TACTICS = ["intro", "exact", "apply", "constructor", "left", "right", "exfalso", "assumption", "contradiction",
            "trivial", "cases", "rcases", "obtain", "rfl", "simp", "decide"]
_SWAPS = [(".1", ".2"), (".left", ".right"), (".mp", ".mpr"), (".1", ".mp"), ("Or.inl", "Or.inr"),
          ("left", "right"), ("inl", "inr"), ("⟨", "("), ("⟩", ")"), ("(", "⟨"), ("And.intro", "Iff.intro"),
          ("exact", "apply"), ("rcases", "obtain"), ("cases", "rcases"), ("trivial", "assumption"),
          ("absurd", "False.elim"), ("h0", "h1"), ("; ", "\n  "), ("\n  ", "; "), ("  ", " "), ("· ", "")]


def mutate(s: str, rng: random.Random) -> str:
    kind = rng.randrange(12)
    i = rng.randrange(len(s) + 1)
    toks = _tokens(s)
    j = rng.randrange(len(toks)) if toks else 0
    if kind == 0 and s:  # delete a character
        return s[:i] + s[i + 1:]
    if kind == 1:  # insert a snippet
        return s[:i] + rng.choice(_SNIPPETS) + s[i:]
    if kind == 2 and toks:  # delete, duplicate or replace a token
        op = rng.randrange(3)
        new = [] if op == 0 else [toks[j], toks[j]] if op == 1 else [rng.choice(_SNIPPETS)]
        return "".join(toks[:j] + new + toks[j + 1:])
    if kind == 3 and len(toks) > 1:  # swap two tokens (adjacent or not)
        k = min(j + 1, len(toks) - 1) if rng.random() < 0.5 else rng.randrange(len(toks))
        toks[j], toks[k] = toks[k], toks[j]
        return "".join(toks)
    if kind == 4:  # rename a hypothesis, everywhere or once
        found = list(re.finditer(r"(?<![A-Za-z0-9_.'])(?:h\d+|h'*|[a-z])(?![A-Za-z0-9_'])", s))
        if found:
            old, new = rng.choice(found).group(), rng.choice(_NAMES)
            spots = [m for m in found if m.group() == old]
            if rng.random() < 0.6:
                for m in reversed(spots):
                    s = s[:m.start()] + new + s[m.end():]
                return s
            m = rng.choice(spots)
            return s[:m.start()] + new + s[m.end():]
    if kind == 5:  # swap paired constructs, one occurrence or all
        a, b = rng.choice(_SWAPS)
        if a in s:
            if rng.random() < 0.5:
                return s.replace(a, "\0").replace(b, a).replace("\0", b)
            k = rng.choice([m.start() for m in re.finditer(re.escape(a), s)])
            return s[:k] + b + s[k + len(a):]
    if kind == 6:  # truncate
        return s[:i]
    if kind == 7:  # extra or missing spacing
        if rng.random() < 0.5 and " " in s:
            k = rng.choice([m.start() for m in re.finditer(" ", s)])
            return s[:k] + s[k + 1:]
        return s[:i] + rng.choice([" ", "  ", "\n", "\n  ", "\n   "]) + s[i:]
    if kind == 8:  # tactic edits: replace a tactic word, or append a tactic
        words = [m for m in re.finditer(r"\b(?:" + "|".join(_TACTICS) + r")\b", s)]
        if words and rng.random() < 0.6:
            m = rng.choice(words)
            return s[:m.start()] + rng.choice(_TACTICS) + s[m.end():]
        return s + rng.choice(["; trivial", "; exact h0", "\n  exact h0", "\n  trivial", ";", "\n", " h0"])
    if kind == 9:  # re-indent a line
        lines = s.split("\n")
        if len(lines) > 1:
            k = rng.randrange(1, len(lines))
            lines[k] = " " * rng.choice([0, 1, 2, 3, 4, 6]) + lines[k].lstrip(" ")
            return "\n".join(lines)
    if kind == 10:  # a term as a tactic proof, or the reverse
        if s.startswith("by "):
            return s[3:]
        return rng.choice(["by exact ", "by\n  exact ", "by intro h0; exact ", "by "]) + s
    if kind == 11:  # wrap or unwrap parentheses
        if rng.random() < 0.5:
            return f"({s})"
        k = s.find("(")
        if k >= 0:
            return s[:k] + s[k + 1:]
    return s[:i] + rng.choice(_SNIPPETS) + s[i:]


def mutate_prompt(prompt: str, rng: random.Random) -> str:
    kind = rng.randrange(8)
    if kind == 0:
        return mutate(prompt, rng)
    m = re.match(r"example(?: \(([a-z ]+) : Prop\))? : (.*) :=$", prompt)
    names, formula = (m.group(1) or "").split(), m.group(2)
    binders = lambda ns: f" ({' '.join(ns)} : Prop)" if ns else ""
    if kind == 1 and names:  # drop or duplicate a declared atom
        k = rng.randrange(len(names))
        ns = names[:k] + names[k + 1:] if rng.random() < 0.5 else names + [names[k]]
        return f"example{binders(ns)} : {formula} :="
    if kind == 2:  # declare another atom, perhaps a keyword
        return f"example{binders(names + [rng.choice(['t', 'q', 'at', 'fun', 'x', 'r'])])} : {formula} :="
    if kind == 3:  # change a connective
        ops = [m.start() for m in re.finditer("[∧∨→↔¬]", formula)]
        if ops:
            k = rng.choice(ops)
            return f"example{binders(names)} : {formula[:k]}{rng.choice('∧∨→↔¬')}{formula[k + 1:]} :="
    if kind == 4:  # swap two atoms in the formula
        atoms = sorted(set(re.findall(r"\b[a-z]\b", formula)))
        if len(atoms) > 1:
            a, b = rng.sample(atoms, 2)
            swapped = re.sub(rf"\b[{a}{b}]\b", lambda x: b if x.group() == a else a, formula)
            return f"example{binders(names)} : {swapped} :="
    if kind == 5:  # add redundant parentheses, or drop some
        if rng.random() < 0.5:
            return f"example{binders(names)} : ({formula}) :="
        return prompt.replace("(", "", 1).replace(")", "", 1) if formula.count("(") else prompt.replace(" :=", ":=")
    if kind == 6:  # spacing and line breaks
        return rng.choice([prompt.replace(" : ", "  : ", 1), prompt + "\n", prompt.replace(" →", "→"),
                           prompt.replace(" :=", " :=\n"), " " + prompt, prompt.replace("Prop", "Type")])
    return prompt.replace(" :=", " := " + rng.choice(["", "by", "fun"]), 1)


_SOUP = _SNIPPETS + _NAMES + _TACTICS + [" "] * 20


def soup(rng: random.Random) -> str:
    """A random string of the checker's own vocabulary: what the lexers and
    parsers see from an untrained model."""
    text = "".join(rng.choice(_SOUP) + rng.choice(("", " ")) for _ in range(rng.randint(1, 24)))
    return rng.choice(("", "", "by ", "by ", "by\n  ")) + text


# Tactic proofs built by simulating the checker's own goal states. References
# and their mutants rarely take hypotheses apart or nest bullets and
# alternatives; these do little else, so most of them check and Lean judges
# the parts of the tactic checker the other cases barely reach.

_HYP_NAMES = ["h0", "h1", "h2", "a", "b", "c", "x", "y", "h", "h'", "this", "p", "id", "_x"]


def _visible(ctx) -> list:
    """(name, type) of each hypothesis a name reaches, innermost first."""
    seen, out = set(), []
    while ctx is not None:
        if ctx[0] is not None and ctx[0] not in seen:
            seen.add(ctx[0])
            out.append((ctx[0], ctx[1]))
        ctx = ctx[2]
    return out


def _arg(text, atomic):
    return text if atomic else f"({text})"


class _Simulator:
    def __init__(self, rng):
        self.rng = rng

    def infer(self, ctx, depth):
        """(text, type, atomic) of a random term whose type the checker infers, or None."""
        rng, hyps = self.rng, _visible(ctx)
        if not hyps:
            return None
        name, ty = rng.choice(hyps)
        t = (name, ty, True)
        other = self.infer(ctx, depth - 1) if depth > 0 and rng.random() < 0.25 else None
        if other is not None:  # a constant where its type is inferred
            if rng.random() < 0.5:
                t = (f"(And.intro {name} {_arg(other[0], other[2])})", ("and", ty, other[1]), True)
            elif ty[0] == "or" and other[1][0] == "imp" and other[1][1] == ty[1]:
                g = self.check(ctx, ("imp", ty[2], other[1][2]), depth - 1)
                if g is not None:
                    t = (f"(Or.elim {name} {_arg(other[0], other[2])} {_arg(*g)})", other[1][2], True)
        for _ in range(rng.randrange(3) if depth > 0 else 0):
            text, ty, atomic = t
            if ty[0] in ("and", "iff"):
                k = rng.choice((1, 2))
                if ty[0] == "and" and rng.random() < 0.15:
                    t = (f"And.{('left', 'right')[k - 1]} {_arg(text, atomic)}", ty[k], False)
                    continue
                field = rng.choice(((("1", "left"), ("2", "right")) if ty[0] == "and" else (("1", "mp"), ("2", "mpr")))[k - 1])
                t = (f"{_arg(text, atomic)}.{field}", prop._fields(ty)[k - 1], True)
            elif ty[0] == "imp":
                a = self.check(ctx, ty[1], depth - 1)
                if a is None:
                    break
                t = (f"{text} {_arg(*a)}", ty[2], False)
        return t

    def check(self, ctx, ty, depth):
        """(text, atomic) of a random term that checks against ty, or None."""
        rng, hyps = self.rng, _visible(ctx)
        options = (["var"] if any(h == ty for _, h in hyps) else []) + (["trivial"] if ty == prop.TOP else [])
        if depth > 0:
            options += {"imp": ["fun"] * 3, "and": ["anon", "anon", "intro"], "iff": ["anon", "intro"],
                        "or": ["inj", "inj"]}.get(ty[0], [])
            if hyps:
                options += ["infer", "infer", "efq", "absurd", "elim"]
        rng.shuffle(options)
        for opt in options[:4]:
            r = self.check_by(ctx, ty, depth, opt)
            if r is not None:
                return r
        return None

    def check_by(self, ctx, ty, depth, opt):
        rng = self.rng
        if opt == "var":
            return rng.choice([n for n, h in _visible(ctx) if h == ty]), True
        if opt == "trivial":
            return "trivial", True
        if opt == "fun":
            names = []
            while ty[0] == "imp" and (not names or rng.random() < 0.4):
                names.append(rng.choice(_HYP_NAMES + ["_"]))
                ctx, ty = (None if names[-1] == "_" else names[-1], ty[1], ctx), ty[2]
            body = self.check(ctx, ty, depth - 1)
            return None if body is None else (f"fun {' '.join(names)} => {body[0]}", False)
        if opt in ("anon", "intro"):
            first, second = prop._fields(ty)
            if opt == "anon" and second[0] in ("and", "iff") and rng.random() < 0.4:  # ⟨a, b, c⟩ flattens
                a, rest = self.check(ctx, first, depth - 1), self.check_by(ctx, second, depth, "anon")
                return None if a is None or rest is None else ("⟨" + a[0] + ", " + rest[0][1:], True)
            a, b = self.check(ctx, first, depth - 1), self.check(ctx, second, depth - 1)
            if a is None or b is None:
                return None
            if opt == "anon":
                return "⟨" + rng.choice((", ", ",", " , ")).join((a[0], b[0])) + "⟩", True
            return f"{'And' if ty[0] == 'and' else 'Iff'}.intro {_arg(*a)} {_arg(*b)}", False
        if opt == "inj":
            k = rng.choice((1, 2))
            a = self.check(ctx, ty[k], depth - 1)
            return None if a is None else (f"Or.{('inl', 'inr')[k - 1]} {_arg(*a)}", False)
        for _ in range(4):
            t = self.infer(ctx, depth - 1)
            if t is None:
                continue
            if opt == "infer" and t[1] == ty:
                return t[0], t[2]
            if opt == "efq" and t[1] == prop.BOT:
                return f"False.elim {_arg(t[0], t[2])}", False
            if opt == "absurd":
                n = self.check(ctx, ("imp", t[1], prop.BOT), depth - 1)
                if n is not None:
                    return f"absurd {_arg(t[0], t[2])} {_arg(*n)}", False
            if opt == "elim" and t[1][0] == "or":
                f = self.check(ctx, ("imp", t[1][1], ty), depth - 1)
                g = self.check(ctx, ("imp", t[1][2], ty), depth - 1)
                if f is not None and g is not None:
                    return f"Or.elim {_arg(t[0], t[2])} {_arg(*f)} {_arg(*g)}", False
        return None

    def pattern(self, ty, depth):
        """(text, the hypotheses of each goal) of a random rcases pattern for ty."""
        rng = self.rng
        if depth > 0 and ty[0] in ("and", "iff") and rng.random() < 0.8:
            (pa, ba), (pb, bb) = (self.pattern(f, depth - 1) for f in prop._fields(ty))
            flat = pb.startswith("⟨") and rng.random() < 0.5
            return "⟨" + pa + ", " + (pb[1:] if flat else pb + "⟩"), [x + y for x in ba for y in bb]
        if depth > 0 and ty[0] == "or" and rng.random() < 0.8:
            (pa, ba), (pb, bb) = self.pattern(ty[1], depth - 1), self.pattern(ty[2], depth - 1)
            return (f"({pa} | {pb})" if rng.random() < 0.3 else f"{pa} | {pb}"), ba + bb
        name = rng.choice(_HYP_NAMES[:11])
        return name, [[(name, ty)]]

    def steps(self, ctx, target, budget):
        """(steps, whether they close the goal): a step is ("tac", text),
        ("·", steps) or ("alts", "cases e with", [(label, steps), ...])."""
        goals, out = [(ctx, target)], []
        while goals and budget > 0:
            budget -= 1
            (ctx, target), rest = goals[0], goals[1:]
            if rest and self.rng.random() < 0.3:
                sub, closed = self.steps(ctx, target, budget)
                step, new = ("·", sub), []
            else:
                made = self.tactic(ctx, target, budget)
                if made is None:
                    return out, False
                (step, new), closed = made, True
            if not closed:
                return out, False
            out.append(step)
            goals = new + rest
        return out, not goals

    def tactic(self, ctx, target, budget):
        """(step, the goals it leaves) for a random tactic that applies, or None."""
        rng = self.rng
        for _ in range(8):
            kind = rng.choices(["intro", "exact", "apply", "rule", "cases", "alts", "rcases", "obtain", "constructor",
                                "side", "exfalso", "close"], weights=[3, 2, 3, 1, 1, 4, 3, 3, 2, 1, 1, 1])[0]
            if kind == "intro" and target[0] == "imp":
                names = []
                while target[0] == "imp" and (not names or rng.random() < 0.5):
                    names.append(rng.choice(_HYP_NAMES))
                    ctx, target = (names[-1], target[1], ctx), target[2]
                return ("tac", "intro " + " ".join(names)), [(ctx, target)]
            if kind == "exact":
                t = self.check(ctx, target, 3)
                if t is not None:
                    return ("tac", "exact " + t[0]), []
            if kind == "rule":
                name = rng.choice(prop._INTRO_RULES)
                made = prop._intro_rule(name, target)
                if made is not None:
                    return ("tac", "apply " + name), [(ctx, g) for g in made]
            if kind in ("apply", "cases", "alts", "rcases", "obtain"):
                t = self.infer(ctx, 2 if kind in ("apply", "cases") else 1)
                if t is None:
                    continue
                text, ty, _ = t
                if kind == "apply" and prop._apply_goals(ty, target) is not None:
                    return ("tac", "apply " + text), [(ctx, g) for g in prop._apply_goals(ty, target)]
                if kind == "cases" and ty == prop.BOT:
                    return ("tac", "cases " + text), []
                if kind in ("apply", "cases") or ty[0] not in ("or", "and", "iff"):
                    continue
                left = prop._clear(ctx, text) if text in dict(_visible(ctx)) else ctx
                if kind == "alts":
                    if ty[0] == "or":
                        wanted = [("inl", [ty[1]]), ("inr", [ty[2]])][::rng.choice((1, 1, -1))]
                    else:
                        parts = prop._fields(ty)
                        wanted = [("intro", [prop._Uninstantiated(f) for f in parts] if ty[0] == "iff" else list(parts))]
                    arms = []
                    for ctor, parts in wanted:
                        bound = [rng.choice(_HYP_NAMES) for _ in parts]
                        arm = left
                        for name, f in zip(bound, parts):
                            arm = (name, f, arm)
                        sub, closed = self.steps(arm, target, budget - 1)
                        if not closed:
                            return None
                        arms.append((f"| {ctor} {' '.join(bound)} =>", sub))
                    return ("alts", f"cases {text} with", arms), []
                pat, branches = self.pattern(ty, 3)
                if not pat.startswith(("⟨", "(")) and "|" not in pat or any(len(dict(b)) != len(b) for b in branches):
                    continue
                goals = []
                for branch in branches:
                    c = left
                    for name, f in branch:
                        c = (name, f, c)
                    goals.append((c, target))
                return ("tac", f"rcases {text} with {pat}" if kind == "rcases" else f"obtain {pat} := {text}"), goals
            if kind == "constructor" and (target[0] in ("and", "iff") or target == prop.TOP):
                return ("tac", "constructor"), [(ctx, g) for g in (prop._fields(target) if target != prop.TOP else [])]
            if kind == "side" and target[0] == "or":
                k = rng.choice((1, 2))
                return ("tac", ("left", "right")[k - 1]), [(ctx, target[k])]
            if kind == "exfalso" and target != prop.BOT and rng.random() < 0.3:
                return ("tac", "exfalso"), [(ctx, prop.BOT)]
            if kind == "close":
                options = (["assumption"] if target in prop._hyps(ctx) else []) + \
                          (["contradiction"] if prop._contradiction(ctx) else []) + \
                          (["trivial"] if prop._trivial(ctx, target) else [])
                if options:
                    return ("tac", rng.choice(options)), []
        return None

    def line(self, steps):
        """The steps on one line, or None: alternatives take the rest of their line."""
        parts = []
        for k, (kind, *rest) in enumerate(steps):
            if kind == "tac":
                parts.append(rest[0])
                continue
            if kind == "alts" and k < len(steps) - 1:
                return None
            inner = [self.line(sub) for sub in ([rest[0]] if kind == "·" else [s for _, s in rest[1]])]
            if None in inner or kind == "alts" and any(s[0] == "alts" for _, sub in rest[1] for s in sub):
                return None
            if kind == "·":
                parts.append(("· " if k == len(steps) - 1 and self.rng.random() < 0.5 else "") + inner[0])
            else:
                parts.append(rest[0] + " " + " ".join(f"{label} {s}" for (label, _), s in zip(rest[1], inner)))
        return self.rng.choice(("; ", " ; ")).join(parts)

    def block(self, steps, col):
        """The steps as lines at column col: bullets, alternatives at or right of
        their `cases`, short arms on the line of their `|`, some tactics joined by `;`."""
        rng, out, k = self.rng, [], 0
        while k < len(steps):
            kind, *rest = steps[k]
            if kind == "tac":
                text = rest[0]
                while k + 1 < len(steps) and steps[k + 1][0] == "tac" and rng.random() < 0.2:
                    k += 1
                    text += "; " + steps[k][1]
                out.append(" " * col + text)
            elif kind == "·":
                inner = self.block(rest[0], col + 2)
                out += [" " * col + "· " + inner[0][col + 2:]] + inner[1:]
            else:
                out.append(" " * col + rest[0])
                at = col + rng.choice((0, 0, 0, 2))
                for label, sub in rest[1]:
                    inner = self.block(sub, at + rng.choice((2, 2, 4)))
                    if len(inner) == 1 and rng.random() < 0.6:
                        out.append(" " * at + label + " " + inner[0].lstrip())
                    else:
                        out += [" " * at + label] + inner
            k += 1
        return out


def _taut_with_hypotheses(rng) -> str:
    """A header `h1 → ... → c` whose hypotheses are worth taking apart."""
    while True:
        atoms = rng.sample("pqrs", rng.randint(2, 3))
        f = prop._random_formula(rng, atoms, rng.randint(1, 3))
        for _ in range(rng.randint(1, 3)):
            h = prop._random_formula(rng, atoms, rng.randint(1, 3))
            f = ("imp", h if h[0] in ("or", "and", "iff") else ("or", h, ("atom", rng.choice(atoms))), f)
        if prop._tautology(f) and len(prop.header(f)) <= 200:
            return prop.header(f)


@functools.lru_cache(maxsize=None)
def simulated_proofs(n=2500, seed=0) -> tuple:
    """(prompt, answer) for n simulated tactic proofs, half on one line and half as blocks."""
    rng = random.Random(seed)
    sim, out = _Simulator(rng), []
    while len(out) < n:
        prompt = _taut_with_hypotheses(rng) if rng.random() < 0.6 else DOMAIN.sample(rng).prompt
        steps, closed = sim.steps(None, prop.unfold(prop.goal_of(prompt)), 16)
        body = sim.line(steps) if rng.random() < 0.5 else "\n" + "\n".join(sim.block(steps, 2))
        if closed and steps and body is not None:
            out.append((prompt, "by" + ("" if body.startswith("\n") else " ") + body))
    return tuple(out)


def layout_mutant(answer: str, rng: random.Random) -> str:
    """answer with its lines re-indented, joined, split at `;` or `|`, blank
    lines added, bullets added or removed, or turned to or from one line."""
    lines = answer.split("\n")
    kind = rng.randrange(8)
    k = rng.randrange(1, len(lines)) if len(lines) > 1 else 0
    body = lines[k].lstrip(" ")
    indent = len(lines[k]) - len(body)
    if kind == 0 and k:
        lines[k] = " " * max(0, indent + rng.choice((-2, -1, 1, 2, 4))) + body
    elif kind == 1 and k > 1:
        lines[k - 1:k + 1] = [lines[k - 1] + rng.choice(("; ", " ", ";")) + body]
    elif kind == 2:
        spots = [(j, m.start(), m.group()) for j, ln in enumerate(lines) for m in re.finditer(r"; | (?=\| )", ln)]
        if spots:
            j, at, sep = rng.choice(spots)
            col = len(lines[j]) - len(lines[j].lstrip(" ")) + rng.choice((0, 0, 2, -2, 1))
            lines[j:j + 1] = [lines[j][:at], " " * max(0, col) + lines[j][at + len(sep):]]
    elif kind == 3:
        lines.insert(rng.randrange(1, len(lines) + 1), rng.choice(("", "  ")))
    elif kind == 4 and k:
        lines[k] = " " * indent + (body[2:] if body.startswith("· ") else "· " + body)
    elif kind == 5:
        return answer.replace("· ", rng.choice(("·  ", "·", "· · ", ". ")), 1)
    elif kind == 6 and len(lines) > 1:
        return "by " + "; ".join(ln.strip() for ln in lines[1:] if ln.strip())
    elif kind == 6:
        return "by\n" + "\n".join("  " + part for part in answer[3:].split("; "))
    elif kind == 7 and k:
        lines[k - 1:k + 1] = [lines[k - 1] + " " + body] if body.startswith("|") else [lines[k - 1], lines[k]]
    return "\n".join(lines)


# Inputs at the edges of the lexers and parsers, for the differential test:
# exotic whitespace and Unicode, every reserved word where a name goes, odd
# fields and fun binders, and odd binder lists in prompts.
_ODD_SPACES = ["\t", "\r", "\r\n", " ", " ", "　", "﻿", "​", "\x0b", "\x0c", " ", "\x85",
               "\n", "\n\n", "\n  ", " \n"]
_ODD_CHARS = ["α", "é", "₁", "𝔸", "😀", "•", "〈", "〉", "⇒", "↦", "λ", "~", "，", "（", "ｈ", "́", "١", "²", "ß", "ﬁ"]
_ODD_WORDS = sorted(prop.RESERVED) + sorted(prop._TACTIC_WORDS) + [
    "lemma", "admit", "native", "nativex", "xrun_x", "runner", "in_", "at'", "h_in", "h'in", "this", "__proto__",
    "constructor", "toString", "hasOwnProperty", "length", "_", "__", "h''", "mp", "elim"]
_ODD_FIELDS = ["0", "3", "01", "10", "1e2", "١", "²", "1_", "1'", "symm", "elim", "toString", "__proto__", "_", "1.2"]
_ODD_FUNS = ["fun ⟨a, b⟩ => a", "fun (a) => a", "fun (a, b) => a", "fun x : p => x", "fun {x} => x", "fun x =>",
             "fun => x", "fun x.1 => x", "fun | x => x", "λ x => x", "fun x ↦ x", "fun x => ⟨x⟩", "fun x=>x.1"]


def odd_answer(answer: str, rng: random.Random) -> str:
    i = rng.randrange(len(answer) + 1)
    kind = rng.randrange(6)
    if kind == 0:
        return answer[:i] + rng.choice(_ODD_SPACES) + answer[i:]
    if kind == 1:
        return answer[:i] + rng.choice(_ODD_CHARS) + answer[i:]
    if kind == 2:  # a name renamed, everywhere, to a word that is no hypothesis name
        names = sorted(set(re.findall(r"(?<![A-Za-z0-9_.'])[a-z][A-Za-z0-9_']*", answer)) - {"fun", "by"})
        if names:
            old = re.escape(rng.choice(names))
            return re.sub(rf"(?<![A-Za-z0-9_.']){old}(?![A-Za-z0-9_'])", rng.choice(_ODD_WORDS), answer)
    if kind == 3:
        spots = list(re.finditer(r"\.(1|2|left|right|mp|mpr)\b", answer))
        if spots:
            m = rng.choice(spots)
            return answer[:m.start() + 1] + rng.choice(_ODD_FIELDS) + answer[m.end():]
    if kind == 4:
        return rng.choice(_ODD_FUNS)
    return answer[:i] + rng.choice(_ODD_WORDS) + answer[i:]


def odd_prompt(prompt: str, rng: random.Random) -> str:
    m = re.match(r"example(?: \(([a-z ]+) : Prop\))? : (.*) :=$", prompt)
    names, formula = (m.group(1) or "").split(), m.group(2)
    kind = rng.randrange(6)
    if kind == 0:  # unsorted, duplicated, missing or extra atoms (none the oracle refuses to compile)
        names = rng.choice([names[::-1], names + names[:1], names[1:], names + [rng.choice(["t", "x", "pq", "this"])]])
    elif kind == 1:
        return prompt.replace(" : Prop)", rng.choice(["  : Prop)", " :  Prop)", ": Prop)", " : Prop )"]))
    elif kind == 2:
        return prompt.replace(" ", rng.choice(_ODD_SPACES + ["  "]), rng.randint(1, 3))
    elif kind == 3:
        i = rng.randrange(len(prompt) + 1)
        return prompt[:i] + rng.choice(_ODD_CHARS) + prompt[i:]
    elif kind == 4:
        formula = re.sub(r"\b([a-z])\b", lambda a: a.group(1).upper() if rng.random() < 0.3 else a.group(1), formula)
    else:
        formula = rng.choice(["(", "¬", "¬¬", "¬ "]) + formula.replace("¬", "¬ ") + rng.choice([")", "", " "])
    return "example" + (f" ({' '.join(names)} : Prop)" if names else "") + f" : {formula} :="


@functools.lru_cache(maxsize=None)
def differential_cases() -> tuple:
    """(prompt, answer, origin) for references, variants, hand-written proofs
    and their mutants, mutated prompts, answers to other prompts, random
    strings, simulated tactic proofs and their layout mutants, and odd inputs."""
    rng = random.Random(0)
    cases, seeds = [], []
    for p in problems():
        answers = variants()[p.prompt]
        cases += [(p.prompt, a, "valid") for a in [p.reference] + answers]
        seeds += [(p.prompt, p.reference)] * 8 + [(p.prompt, a) for a in rng.sample(answers, min(len(answers), 4))] * 3
    cases += [(p, a, "handwritten") for p, a in HANDWRITTEN]
    seeds += HANDWRITTEN * 25
    for prompt, answer in seeds:
        a = answer
        for _ in range(rng.choice((1, 1, 1, 2, 3))):
            a = mutate(a, rng)
        cases.append((prompt, a, "mutant"))
    ps = problems()
    for p in ps:
        for _ in range(6):
            cases.append((mutate_prompt(p.prompt, rng), p.reference, "prompt mutant"))
        other = rng.choice(ps)
        cases.append((p.prompt, other.reference, "other answer"))
        cases += [(p.prompt, soup(rng), "random") for _ in range(5)]
    rng = random.Random(1)
    sims = simulated_proofs()
    cases += [(p, a, "simulated") for p, a in sims]
    for _ in range(3000):
        p, a = rng.choice(sims)
        for _ in range(rng.choice((1, 1, 2))):
            a = layout_mutant(a, rng)
        cases.append((p, a, "layout"))
    for _ in range(4000):
        p, a = rng.choice(sims)
        cases.append((odd_prompt(p, rng), a, "odd") if rng.random() < 0.3 else (p, odd_answer(a, rng), "odd"))
    return tuple(cases)


def _python_verdict(prompt, answer):
    return DOMAIN.verify(prompt, answer) if DOMAIN.claims(prompt) else None


def _conj(n: int) -> str:
    return " ∧ ".join(["p"] * n)


# (prompt, answer, prop.py's verdict, Lean's): Lean 4.34 runs out of recursion
# (maxRecDepth, which no tactic recovers from) on a header with a chain of 254
# ¬, and when `trivial` splits 64 nested ∧, or 40 under 62 bullets. prop.py and
# web/checker.js refuse these, and the nearby cases their limits also refuse
# although Lean compiles them.
RECURSION_CASES = [
    ("example (p : Prop) : True ∨ " + "¬" * 254 + "p :=", "Or.inl trivial", False, False),
    ("example (p : Prop) : " + "¬" * 254 + "p → True :=", "fun _ => trivial", False, False),
    ("example (p : Prop) : True ∨ " + "¬" * 240 + "p :=", "Or.inl trivial", False, True),
    ("example (p : Prop) : True ∨ " + "¬" * 199 + "p :=", "Or.inl trivial", True, True),
    ("example (p : Prop) : True ∨ " + "(" * 64 + "¬" * 135 + "p" + ")" * 64 + " :=", "Or.inl trivial", True, True),
    ("example (p : Prop) : p → " + _conj(65) + " :=", "by intro h; trivial", False, False),
    ("example (p : Prop) : False → " + _conj(65) + " :=", "by intro h; trivial", False, False),
    ("example (p : Prop) : p → " + _conj(40) + " :=", "by intro h; " + "· " * 62 + "trivial", False, False),
    ("example (p : Prop) : p → " + _conj(34) + " :=", "by intro h; trivial", False, True),
    ("example (p : Prop) : p → " + _conj(33) + " :=", "by intro h; " + "· " * 63 + "trivial", True, True),
    ("example (p : Prop) : (" + _conj(33) + ") → " + _conj(33) + " :=", "by intro h; " + "· " * 63 + "trivial", True, True),
    ("example (p : Prop) : p → " + _conj(65) + " :=", "fun h => ⟨" + ", ".join(["h"] * 65) + "⟩", True, True),
    ("example (p : Prop) : p → " + _conj(60) + " :=", "by intro h; " + "· " * 63 + "exact ⟨" + ", ".join(["h"] * 60) + "⟩",
     True, True),
    ("example (p : Prop) : p → p :=", "by intro h; " + "· " * 63 + "exact " + "(" * 62 + "h" + ")" * 62, True, True),
    ("example (p : Prop) : (p → p) → p → p :=", "by intro f x; " + "· " * 62 + "exact " + "f (" * 61 + "x" + ")" * 61,
     True, True),
    ("example (p q : Prop) : " + "¬" * 249 + "False → p → ¬p → q :=", "by intro a b c; " + "· " * 63 + "contradiction",
     False, True),
]


def test_recursion_limits():
    for prompt, answer, verdict, _ in RECURSION_CASES:
        assert len(prompt) <= 300 and DOMAIN.claims(prompt), prompt
        assert DOMAIN.verify(prompt, answer) == verdict, (prompt, answer)


def test_recursion_limits_against_lean(lean):
    verdicts = lean_oracle.check_declarations([p + " " + a for p, a, _, _ in RECURSION_CASES], lean)
    for (prompt, answer, verdict, expected), compiled in zip(RECURSION_CASES, verdicts):
        assert compiled == expected, (prompt, answer)
        assert compiled or not verdict


def test_differential_against_javascript(node):
    cases = differential_cases() + tuple((p, a, "recursion") for p, a, _, _ in RECURSION_CASES)
    assert len(cases) >= 30_000
    js = js_verify([(p, a) for p, a, _ in cases])
    py = [_python_verdict(p, a) for p, a, _ in cases]
    mismatches = [(c, j, q) for c, j, q in zip(cases, js, py) if j != q]
    counts = {}
    for (_, _, origin), verdict in zip(cases, py):
        counts.setdefault(origin, [0, 0, 0])[{True: 0, False: 1, None: 2}[verdict]] += 1
    print(f"prop differential: {len(cases)} cases, {len(mismatches)} disagreements; "
          + "; ".join(f"{o}: {a} accepted, {r} rejected, {u} unclaimed" for o, (a, r, u) in counts.items()))
    assert not mismatches, mismatches[:10]
    assert all(v is True for (_, _, origin), v in zip(cases, py) if origin == "valid")
    simulated = [v for (_, _, origin), v in zip(cases, py) if origin == "simulated"]
    assert sum(simulated) > 0.95 * len(simulated)  # the simulation follows the checker closely


def test_checker_never_raises():
    deep = "example (p : Prop) : " + "(" * 70 + "p" + ")" * 70 + " :="
    weird_prompts = [None, 3, "", "example : True :=", "example (p : Prop) : p → p :=\n", deep,
                     "example (p : Prop) : " + "¬" * 285 + "p :=", "example (p : Prop) : " + "p → " * 70 + "p :=",
                     "example (\ud800 : Prop) : True :=", "example : True \ud800 :=", "example (p p : Prop) : p :=",
                     "example (at : Prop) : at :=", "example (p : Prop) : 𝔸 → p :=" + "x" * 300]
    weird_answers = [None, 7, "", "by", "by ", "by\n", "\ud800", "fun \udfff => x", "fun h => " + "(" * 80 + "h" + ")" * 80,
                     "⟨" * 70 + "trivial" + "⟩" * 70, "fun h => h" + ".1" * 190, "fun " + "h " * 190 + "=> h",
                     "by " + "· " * 70 + "trivial", "by intro h; rcases h with " + "⟨a, " * 60 + "b" + "⟩" * 60,
                     "trivial", "fun h => h", "by trivial", "x" * 500, "𝔸" * 250]
    for p in weird_prompts:
        DOMAIN.claims(p)
        DOMAIN.split_key(p) if isinstance(p, str) else None
        for a in weird_answers:
            assert DOMAIN.verify(p, a) in (True, False)
    # The length limits, at their edges.
    assert DOMAIN.verify("example : True :=", "trivial" + " " * 393)  # 400 characters
    assert not DOMAIN.verify("example : True :=", "trivial" + " " * 394)
    assert DOMAIN.claims("example : True" + " " * 283 + " :=")  # 300 characters
    assert not DOMAIN.claims("example : True" + " " * 284 + " :=")


# ---------------------------------------------------------------- against Lean

def test_soundness_against_lean(lean):
    """Every reference and variant of N_PROBLEMS problems, and every other
    differential case the checker accepts (mutants, simulated tactic proofs,
    layout and odd variants), compiles. Also reports how many of the cases the
    checker rejects Lean accepts (how conservative it is)."""
    valid = []
    for p in problems():
        valid += [p.prompt + " " + a for a in [p.reference] + variants()[p.prompt]]
    accepted, rejected = set(), set()
    for prompt, answer, origin in differential_cases():
        if origin != "valid" and DOMAIN.claims(prompt):
            (accepted if DOMAIN.verify(prompt, answer) else rejected).add(prompt + " " + answer)
    accepted -= set(valid)
    sample = sorted(r for r in rejected if lean_oracle.is_plain_example(r))
    sample = random.Random(0).sample(sample, min(len(sample), 2000))
    decls = list(dict.fromkeys(valid)) + sorted(accepted) + sample
    # Lean's time per declaration grows with the length of the file, so files
    # of a thousand declarations take a fraction of the time of one big file.
    verdicts = dict(zip(decls, lean_oracle.check_in_chunks(decls, lean, chunk=1000)))
    failed_valid = [d for d in dict.fromkeys(valid) if not verdicts[d]]
    failed_accepted = [d for d in sorted(accepted) if not verdicts[d]]
    lean_accepts = [d for d in sample if verdicts[d]]
    print(f"prop vs Lean: {len(set(valid))} references and variants, {len(accepted)} other accepted cases; "
          f"of {len(sample)} rejected cases Lean accepts {len(lean_accepts)}")
    for d in lean_accepts[:10]:
        print("   checker rejects, Lean accepts:", repr(d))
    assert not failed_valid, failed_valid[:10]
    assert not failed_accepted, failed_accepted[:10]
