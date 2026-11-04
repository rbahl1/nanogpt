"""First-order logic over one type α, proved in Lean 4 term mode.

    example (α : Type) (P Q : α → Prop) : (∃ x, P x ∧ Q x) → ∃ x, P x :=
                                        the model writes:   fun ⟨x, h0⟩ => ⟨x, h0.1⟩

The header declares predicates P Q R : α → Prop, a relation r : α → α → Prop,
propositions p q : Prop and individuals a b : α: exactly the symbols the
formula uses, each once, in that order. An answer is a proof term (no `by`):
`fun` over names or anonymous-constructor patterns, application, ⟨...⟩,
projections .1 .2 .left .right .mp .mpr, `.elim` on ∨ ∃ False, and Or.inl,
Or.inr, Or.elim, Exists.intro, Exists.elim, And.intro, And.left, And.right,
Iff.intro, False.elim, absurd, trivial.

The checker is a port of the Fol section of web/checker.js (itself a port of
the old Python checker, which was fuzzed against Lean) and agrees with it on
every input: limits are measured in UTF-16 code units, regexes keep their
JavaScript meaning, and where the JavaScript catches a RangeError this catches
a RecursionError. Formulas are locally nameless (a bound variable is a de
Bruijn index), so equality of formulas is alpha-equivalence and instantiation
cannot capture; proofs are checked bidirectionally with ¬A unfolded to A → False.

Problems come from inference schemas (instantiation, ∀ over ∧, ∃ introduction
and elimination, quantifier shifts and swaps, syllogisms through ∀, ↔ between
quantified statements, contradiction, case analysis on ∨, ...). A schema is a
formula over slots (A B C unary, E binary, S T closed) and individuals c d,
with a proof that never looks inside a slot, so it proves every instance.
Each slot is filled with a random formula, mostly from symbols no other slot
uses; some problems repeat symbols, and a few slots hold True or False. Then
the slots can coincide or contradict each other and open a shorter route than
the schema's, so a search for the shortest proof of the instance (see _Search)
has the last word. The reference has hypotheses h0, h1, ... and individuals
x, y, z named in binding order and unused binders written `_`.
"""

import itertools
import random
import re

from .base import Domain, Problem

MAX_ANSWER, MAX_PROMPT, MAX_NESTING = 400, 300, 64
# The generator's own bounds: a task sequence <bos> prompt <ans> answer <eos>
# must fit the model's 256-token context.
MAX_SAMPLE_PROMPT, MAX_REFERENCE, MAX_SEQUENCE = 160, 200, 256

TOP, BOT = ("top",), ("bot",)
BINARY = ("and", "or", "imp", "iff")
QUANTIFIER = ("all", "ex")
SYMBOL = {"and": "∧", "or": "∨", "imp": "→", "iff": "↔"}
TIGHTER = {"imp": "or", "or": "and", "and": None}
QUANTIFIERS = {"∀": "all", "∃": "ex"}
ARITY = {"P": 1, "Q": 1, "R": 1, "r": 2, "p": 0, "q": 0}
INDIVIDUALS = ("a", "b")
GROUPS = ("PQR", "r", "pq", "ab")  # the names each binder group of the header may declare
CONSTANTS = {"trivial": 0, "absurd": 2, "Or.inl": 1, "Or.inr": 1, "Or.elim": 3, "False.elim": 1,
             "And.intro": 2, "And.left": 1, "And.right": 1, "Iff.intro": 2, "Exists.intro": 2, "Exists.elim": 2}
CHECKED = {"trivial", "absurd", "Or.inl", "Or.inr", "Or.elim", "False.elim", "And.intro", "Iff.intro",
           "Exists.intro", "Exists.elim"}
FIELDS = {"1", "2", "left", "right", "mp", "mpr", "elim"}

# Lean reads only plain spaces between a formula's tokens (a tab or any other
# Unicode space is an error), so nothing else separates them here.
_FORMULA_TOKEN = re.compile(r" *([()¬∧∨→↔∀∃,]|[A-Za-z_][A-Za-z0-9_']*)")
_VARIABLE = re.compile(r"[a-z]'*")
# `[^\n]` is any character but a newline in both languages; \Z is JavaScript's
# `$` without the m flag (Python's `$` would also match before a final newline).
_HEADER = re.compile(r"example \(α : Type\)(?: \(([PQR ]+) : α → Prop\))?(?: \((r) : α → α → Prop\))?"
                     r"(?: \(([pq ]+) : Prop\))?(?: \(([ab ]+) : α\))? : ([^\n]+) :=\Z")
_BINDER = re.compile(r"[A-Za-z][0-9_']*|_")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.(?:[A-Za-z_][A-Za-z0-9_']*|[0-9]+))*")
_FIELD = re.compile(r"\.([A-Za-z_][A-Za-z0-9_']*|[0-9]+)")


class _Reject(Exception):
    pass


def _js_len(s: str) -> int:
    """String length as JavaScript counts it (UTF-16 code units); a lone
    surrogate, which a JavaScript string can also hold, counts one."""
    return len(s.encode("utf-16-le", "surrogatepass")) // 2


# ---------------------------------------------------------------- formulas
# Tuples: ("pred", name, args) with each argument a de Bruijn index (an int,
# 0 the innermost quantifier) or a free individual (a str), TOP, BOT,
# ("not", a), (op, a, b), ("all" | "ex", body). Bound variables keep no name,
# so == on formulas is alpha-equivalence. Each recursion over a formula takes
# one stack frame per level, so a prompt's length bounds it.

def _open(f, t, k=0):
    """The body f of a quantifier of a closed formula with its variable (index
    k under k more binders) instantiated with the free individual t: nothing
    in a closed formula can capture t, so no renaming is ever needed."""
    tag = f[0]
    if tag == "pred":
        return ("pred", f[1], tuple(t if x == k else x for x in f[2])) if k in f[2] else f
    if tag in BINARY:
        return (tag, _open(f[1], t, k), _open(f[2], t, k))
    if tag == "not":
        return ("not", _open(f[1], t, k))
    if tag in QUANTIFIER:
        return (tag, _open(f[1], t, k + 1))
    return f


def _mentions(f, k=0):
    """Whether the variable with index k occurs in f: Lean infers a quantified
    variable's type from a use, so an unused one is rejected."""
    tag = f[0]
    if tag == "pred":
        return k in f[2]
    if tag in QUANTIFIER:
        return _mentions(f[1], k + 1)
    if tag == "not":
        return _mentions(f[1], k)
    if tag in BINARY:
        return _mentions(f[1], k) or _mentions(f[2], k)
    return False


def _unfold(f):
    tag = f[0]
    if tag == "not":
        return ("imp", _unfold(f[1]), BOT)
    if tag in BINARY:
        return (tag, _unfold(f[1]), _unfold(f[2]))
    if tag in QUANTIFIER:
        return (tag, _unfold(f[1]))
    return f


def _symbols(f, preds, individuals):
    """Adds the predicate and atom names and the free individuals of f, so that
    the header must declare exactly the symbols the formula uses."""
    tag = f[0]
    if tag == "pred":
        preds.add(f[1])
        for x in f[2]:
            if isinstance(x, str):
                individuals.add(x)
    elif tag in QUANTIFIER or tag == "not":
        _symbols(f[1], preds, individuals)
    elif tag in BINARY:
        _symbols(f[1], preds, individuals)
        _symbols(f[2], preds, individuals)


def _parse_formula(text, arity, individuals):
    """Precedence climbing (↔ below → below ∨ below ∧ below ¬; → ∨ ∧ to the
    right, ↔ not associative) plus quantifiers whose body extends as far right
    as possible. `arity` gives the declared predicates and atoms; only
    parentheses and quantifiers recurse, and their depth is bounded."""
    toks, pos = [], 0
    while (m := _FORMULA_TOKEN.match(text, pos)) is not None:
        toks.append(m.group(1))
        pos = m.end()
    if text[pos:].strip(" "):
        raise _Reject
    scope = []
    i = 0

    def peek():
        return toks[i] if i < len(toks) else ""

    def iff(depth):
        nonlocal i
        a = chain("imp", depth)
        if peek() != "↔":
            return a
        i += 1
        b = chain("imp", depth)
        if peek() == "↔":  # `a ↔ b ↔ c` does not parse in Lean
            raise _Reject
        return ("iff", a, b)

    def chain(op, depth):
        nonlocal i
        if op is None:
            return unary(depth)
        parts = [chain(TIGHTER[op], depth)]
        while peek() == SYMBOL[op]:
            i += 1
            parts.append(chain(TIGHTER[op], depth))
        f = parts.pop()
        while parts:
            f = (op, parts.pop(), f)
        return f

    def unary(depth):
        nonlocal i
        nots = 0
        while peek() == "¬":
            nots += 1
            i += 1
        tok = peek()
        i += 1
        if tok in ("(", "∀", "∃") and depth >= MAX_NESTING:
            raise _Reject
        if tok == "(":
            f = iff(depth + 1)
            if peek() != ")":
                raise _Reject
            i += 1
        elif tok in QUANTIFIERS:
            name = peek()
            i += 1
            if not _VARIABLE.fullmatch(name) or name in arity or peek() != ",":
                raise _Reject
            i += 1
            scope.append(name)
            body = iff(depth + 1)
            scope.pop()
            if not _mentions(body):
                raise _Reject
            f = (QUANTIFIERS[tok], body)
        elif tok == "True":
            f = TOP
        elif tok == "False":
            f = BOT
        elif tok in arity:
            args = []
            for _ in range(arity[tok]):
                args.append(term())
            f = ("pred", tok, tuple(args))
        else:
            raise _Reject
        for _ in range(nots):
            f = ("not", f)
        return f

    # A bound variable, innermost binder first (so it shadows an individual), else a declared individual.
    def term():
        nonlocal i
        name = peek()
        i += 1
        for j in range(len(scope) - 1, -1, -1):
            if scope[j] == name:
                return len(scope) - 1 - j
        if name in individuals:
            return name
        raise _Reject

    f = iff(0)
    if i != len(toks):
        raise _Reject
    return f


def _goal_of(prompt):
    """(goal, declared individuals), or None unless the header declares exactly
    the symbols the formula uses, each once, in the fixed order."""
    if not isinstance(prompt, str) or _js_len(prompt) > MAX_PROMPT:
        return None
    m = _HEADER.match(prompt)
    if m is None:
        return None
    declared = []
    for g, group in enumerate(GROUPS):
        names = m.group(g + 1).split(" ") if m.group(g + 1) else []
        # One-letter names from the group's own set, strictly ascending: the
        # character class alone would also admit `(PQ : α → Prop)` or `(Q P : ...)`.
        for k, name in enumerate(names):
            if len(name) != 1 or name not in group or (k > 0 and not names[k - 1] < name):
                return None
        declared.extend(names)
    individuals = [n for n in declared if n in INDIVIDUALS]
    arity = {n: ARITY[n] for n in declared if n in ARITY}
    preds, used = set(), set()
    try:
        goal = _parse_formula(m.group(5), arity, individuals)
        _symbols(goal, preds, used)
    except (_Reject, RecursionError):
        return None
    if preds != set(arity) or used != set(individuals):
        return None
    return goal, individuals


# ---------------------------------------------------------------- proof terms
# ("var", name), ("const", name, args), ("proj", t, field), ("app", head, args),
# ("lam", binder, body) with a binder a name or ("pat", items), ("anon", items).

def _lex_term(text):
    """Tokens, and for each whether a plain space precedes it: Lean reads an
    argument only after whitespace (`f(x)`, `h.2x` do not parse), and a term
    stays on one line. A projection written right after `)` or `⟩` is a "." token."""
    toks, spaced = [], []
    i, gap, n = 0, False, len(text)
    while i < n:
        c = text[i]
        if c == " ":
            gap = True
            i += 1
            continue
        spaced.append(gap)
        gap = False
        if c in "(⟨,":
            toks.append(("p", c))
            i += 1
        elif c in ")⟩":
            toks.append(("p", c))
            i += 1
            while (m := _FIELD.match(text, i)) is not None:
                toks.append((".", m.group(1)))
                spaced.append(False)
                i = m.end()
        elif text.startswith("=>", i):
            toks.append(("p", "=>"))
            i += 2
        else:
            m = _NAME.match(text, i)
            if m is None:
                raise _Reject
            toks.append(("id", m.group(0)))
            i = m.end()
    return toks, spaced


def _binder_names(binder, out):
    if isinstance(binder, str):
        out.append(binder)
    else:
        for item in binder[1]:
            _binder_names(item, out)


def _parse_term(text, individuals):
    """Recursive descent over the fragment:
      term   := "fun" binder+ "=>" term | head (" " atom)*
      binder := name | "⟨" binder ("," binder)+ "⟩"
      head   := constant (" " atom)^arity | atom
      atom   := name ("." field)* | "(" term ")" ("." field)* | "⟨" term ("," term)+ "⟩" ("." field)*
    Names resolve innermost binder first, then the prompt's individuals."""
    toks, spaced = _lex_term(text)
    i = depth = 0
    scope = list(individuals)
    end = ("end", "")

    def peek():
        return toks[i] if i < len(toks) else end

    def take():
        nonlocal i
        t = peek()
        if t is end:
            raise _Reject
        i += 1
        return t

    def enter():
        nonlocal depth
        depth += 1
        if depth > MAX_NESTING:
            raise _Reject

    def project(t, field):
        if field not in FIELDS:
            raise _Reject
        return ("proj", t, field)

    def term():
        nonlocal i, depth
        enter()
        k0, n0 = peek()
        if k0 == "id" and n0 == "fun":
            i += 1
            binders = []
            while peek() != ("p", "=>"):
                binders.append(binder())
            if not binders:
                raise _Reject
            i += 1
            names = []
            for b in binders:
                _binder_names(b, names)
            # Lean binds a group's pattern variables after its other binders, so
            # with a pattern in the group sequential scoping is only faithful when no name repeats.
            bound = [n for n in names if n != "_"]
            if len(set(bound)) != len(bound) and not all(isinstance(b, str) for b in binders):
                raise _Reject
            scope.extend(names)
            t = term()
            del scope[len(scope) - len(names):]
            for b in reversed(binders):
                t = ("lam", b, t)
        else:
            if k0 == "id" and CONSTANTS.get(n0, 0) > 0:
                i += 1
                args = []
                for _ in range(CONSTANTS[n0]):
                    args.append(argument())
                t = ("const", n0, args)
            else:
                t = atom()
            args = []
            while starts_atom():
                args.append(argument())
            if args:
                t = ("app", t, args)
        depth -= 1
        return t

    def binder():
        nonlocal i, depth
        kind, text = take()
        if kind == "id" and _BINDER.fullmatch(text):
            return text
        if (kind, text) != ("p", "⟨"):
            raise _Reject
        enter()
        items = [binder()]
        while peek() == ("p", ","):
            i += 1
            items.append(binder())
        if take() != ("p", "⟩"):
            raise _Reject
        depth -= 1
        if len(items) < 2:
            raise _Reject
        return ("pat", items)

    def starts_atom():
        kind, text = peek()
        return (kind == "id" and text != "fun") or (kind == "p" and text in ("(", "⟨"))

    def argument():
        if i >= len(toks) or not spaced[i]:
            raise _Reject
        return atom()

    def atom():
        nonlocal i
        kind, text = take()
        if kind == "id":
            parts = text.split(".")
            if parts[0] != "_" and parts[0] in scope:
                t = ("var", parts[0])
                fields = parts[1:]
            else:
                width = 1 if parts[0] in CONSTANTS else 2
                name = ".".join(parts[:width])
                if name not in CONSTANTS or CONSTANTS[name] != 0:  # unknown, or missing its arguments
                    raise _Reject
                t = ("const", name, [])
                fields = parts[width:]
            for f in fields:
                t = project(t, f)
            return t
        if (kind, text) == ("p", "("):
            t = term()
            if take() != ("p", ")"):
                raise _Reject
        elif (kind, text) == ("p", "⟨"):
            items = [term()]
            while peek() == ("p", ","):
                i += 1
                items.append(term())
            if take() != ("p", "⟩"):
                raise _Reject
            if len(items) < 2:
                raise _Reject
            t = ("anon", items)
        else:
            raise _Reject
        while peek()[0] == ".":
            t = project(t, take()[1])
        return t

    t = term()
    if i != len(toks):
        raise _Reject
    return t


# Contexts are linked lists (name, kind, value, outer), so no branch of the
# checker leaks bindings into another. A hypothesis's value is its formula; an
# individual's is its identity, the declared name or a fresh "#n" shared with
# no other individual, which is what enforces the eigenvariable conditions.
HYP, IND = 0, 1
_fresh = itertools.count()


def _lookup(ctx, name):
    while ctx is not None:
        if ctx[0] == name:
            return ctx
        ctx = ctx[3]
    raise _Reject


def _individual(ctx, t):
    """The individual a term names; individuals are only ever variables."""
    if t[0] != "var":
        raise _Reject
    b = _lookup(ctx, t[1])
    if b[1] != IND:
        raise _Reject
    return b[2]


def _new_individual(ctx, name, body):
    c = f"#{next(_fresh)}"
    return (name, IND, c, ctx), _open(body, c)


def _bind(ctx, binder, ty):
    """Binds a `fun` binder against the expected type: a name against ∀ binds a
    fresh individual, anything against → a hypothesis (possibly a pattern)."""
    if isinstance(binder, str) and ty[0] == "all":
        return _new_individual(ctx, binder, ty[1])
    if ty[0] != "imp":
        raise _Reject
    return _bind_hyp(ctx, binder, ty[1]), ty[2]


def _bind_hyp(ctx, binder, ty):
    return (binder, HYP, ty, ctx) if isinstance(binder, str) else _destructure(ctx, binder[1], ty)


def _destructure(ctx, items, ty):
    """The pattern ⟨items⟩ against an ∧ or an ∃; surplus items nest into the last field, as in Lean."""
    if ty[0] == "and":
        ctx = _bind_hyp(ctx, items[0], ty[1])
        rest = ty[2]
    elif ty[0] == "ex" and isinstance(items[0], str):
        ctx, rest = _new_individual(ctx, items[0], ty[1])
    else:
        raise _Reject
    return _bind_hyp(ctx, items[1], rest) if len(items) == 2 else _destructure(ctx, items[1:], rest)


def _check(ctx, t, ty):
    tag = t[0]
    if tag == "lam":
        inner, body = _bind(ctx, t[1], ty)
        _check(inner, t[2], body)
    elif tag == "anon":
        _check_fields(ctx, t[1], ty)
    elif tag == "const" and t[1] in CHECKED:
        _check_const(ctx, t[1], t[2], ty)
    elif tag == "proj" and t[2] == "elim":
        _check_elim(ctx, t[1], [], ty, None)
    elif tag == "app" and t[1][0] == "proj" and t[1][2] == "elim":
        _check_elim(ctx, t[1][1], t[2], ty, None)
    elif _infer(ctx, t) != ty:
        raise _Reject


def _fields(ty):
    if ty[0] == "and":
        return ty[1], ty[2]
    if ty[0] == "iff":
        return ("imp", ty[1], ty[2]), ("imp", ty[2], ty[1])  # Iff.intro mp mpr
    raise _Reject


def _check_fields(ctx, items, ty):
    """⟨a, b⟩ against ∧ or ↔, or ⟨w, h⟩ against ∃ with an individual witness w;
    ⟨a, b, c⟩ nests the surplus into the last field, as in Lean."""
    if ty[0] == "ex":
        rest = _open(ty[1], _individual(ctx, items[0]))
    else:
        first, rest = _fields(ty)
        _check(ctx, items[0], first)
    if len(items) == 2:
        _check(ctx, items[1], rest)
    else:
        _check_fields(ctx, items[1:], rest)


def _check_const(ctx, name, args, ty):
    if name in ("Or.inl", "Or.inr"):
        if ty[0] != "or":
            raise _Reject
        _check(ctx, args[0], ty[1] if name == "Or.inl" else ty[2])
    elif name in ("Or.elim", "Exists.elim"):
        _check_elim(ctx, args[0], args[1:], ty, "or" if name == "Or.elim" else "ex")
    elif name == "Exists.intro":
        if ty[0] != "ex":
            raise _Reject
        _check(ctx, args[1], _open(ty[1], _individual(ctx, args[0])))
    elif name == "False.elim":
        _check(ctx, args[0], BOT)
    elif name == "absurd":
        # Lean elaborates absurd's arguments left to right, so the first one's type must be known on its own.
        _check(ctx, args[1], ("imp", _infer(ctx, args[0]), BOT))
    elif name in ("And.intro", "Iff.intro"):
        if ty[0] != ("and" if name == "And.intro" else "iff"):
            raise _Reject
        _check_fields(ctx, args, ty)
    elif ty != TOP:  # trivial
        raise _Reject


def _check_elim(ctx, subject, args, ty, kind):
    """`subject.elim args` (or Or.elim / Exists.elim when `kind` is given)
    against ty. Lean elaborates the subject first, so its type must be
    inferable; the cases are then checked against the motive ty (closed, so
    it may sit under the ∀)."""
    d = _infer(ctx, subject)
    if kind is not None and d[0] != kind:
        raise _Reject
    if d[0] == "bot" and not args:
        return
    if d[0] == "or" and len(args) == 2:
        _check(ctx, args[0], ("imp", d[1], ty))
        _check(ctx, args[1], ("imp", d[2], ty))
    elif d[0] == "ex" and len(args) == 1:
        _check(ctx, args[0], ("all", ("imp", d[1], ty)))
    else:
        raise _Reject


def _infer(ctx, t):
    """The type of a term determined without an expected type: hypotheses,
    projections, applications and some constants; everything else needs one."""
    tag = t[0]
    if tag == "var":
        b = _lookup(ctx, t[1])
        if b[1] != HYP:
            raise _Reject
        return b[2]
    if tag == "proj":
        ty, field = _infer(ctx, t[1]), t[2]
        if ty[0] == "and" and field in ("1", "left", "2", "right"):
            return ty[1] if field in ("1", "left") else ty[2]
        if ty[0] == "iff" and field in ("1", "mp", "2", "mpr"):
            return ("imp", ty[1], ty[2]) if field in ("1", "mp") else ("imp", ty[2], ty[1])
        raise _Reject
    if tag == "app":
        ty = _infer(ctx, t[1])
        for arg in t[2]:
            if ty[0] == "imp":
                _check(ctx, arg, ty[1])
                ty = ty[2]
            elif ty[0] == "all":
                ty = _open(ty[1], _individual(ctx, arg))
            else:
                raise _Reject
        return ty
    if tag == "const":
        name, args = t[1], t[2]
        if name == "trivial":
            return TOP
        if name in ("And.left", "And.right"):
            ty = _infer(ctx, args[0])
            if ty[0] != "and":
                raise _Reject
            return ty[1] if name == "And.left" else ty[2]
        if name == "And.intro":
            return ("and", _infer(ctx, args[0]), _infer(ctx, args[1]))
        if name == "Iff.intro":
            f, g = _infer(ctx, args[0]), _infer(ctx, args[1])
            if f[0] != "imp" or g != ("imp", f[2], f[1]):
                raise _Reject
            return ("iff", f[1], f[2])
    raise _Reject


def verify(prompt, answer) -> bool:
    if not isinstance(answer, str) or _js_len(answer) > MAX_ANSWER:
        return False
    parsed = _goal_of(prompt)
    if parsed is None:
        return False
    goal, individuals = parsed
    ctx = None
    for name in individuals:
        ctx = (name, IND, name, ctx)
    try:
        _check(ctx, _parse_term(answer, individuals), _unfold(goal))
        return True
    except (_Reject, RecursionError):
        return False


# ---------------------------------------------------------------- printing
_PREC = {"iff": 20, "imp": 25, "or": 30, "and": 35}


def _variable(level):
    """The name of a bound variable by its level (binders above it): x, y, z, x', ..."""
    return "xyz"[level % 3] + "'" * (level // 3)


def _show(f, rename=None, prec=0, tail=False, depth=0):
    """f with the fewest parentheses: `prec` is the binding power the context
    requires, `tail` whether more of the formula follows on the right (a
    quantifier there needs parentheses, or its body would swallow the rest).
    Bound variables are named by level; `rename` maps symbols."""
    rename = rename or {}
    tag = f[0]
    if tag == "pred":
        args = [_variable(depth - 1 - x) if isinstance(x, int) else rename.get(x, x) for x in f[2]]
        return " ".join([rename.get(f[1], f[1])] + args)
    if tag == "top":
        return "True"
    if tag == "bot":
        return "False"
    if tag == "not":
        return "¬" + _show(f[1], rename, 40, tail, depth)
    if tag in QUANTIFIER:
        s = f"{'∀' if tag == 'all' else '∃'} {_variable(depth)}, {_show(f[1], rename, 0, False, depth + 1)}"
        return f"({s})" if tail else s
    p = _PREC[tag]
    if p < prec:
        return f"({_show(f, rename, 0, False, depth)})"
    left = _show(f[1], rename, p + 1, True, depth)
    right = _show(f[2], rename, p + 1 if tag == "iff" else p, tail, depth)
    return f"{left} {SYMBOL[tag]} {right}"


def _declaration(goal, rename=None):
    """The header for `goal`, declaring exactly the symbols it uses."""
    rename = rename or {}
    preds, used = set(), set()
    _symbols(goal, preds, used)
    names = {rename.get(s, s) for s in preds | used}
    parts = ["example (α : Type)"]
    for group, kind in zip(GROUPS, ("α → Prop", "α → α → Prop", "Prop", "α")):
        declared = sorted(n for n in names if n in group)
        if declared:
            parts.append(f"({' '.join(declared)} : {kind})")
    return f"{' '.join(parts)} : {_show(goal, rename)} :="


def _occurrences(f, out):
    """Appends the symbols of f in order of first occurrence in its text."""
    tag = f[0]
    if tag == "pred":
        for s in (f[1],) + tuple(x for x in f[2] if isinstance(x, str)):
            if s not in out:
                out.append(s)
    elif tag in QUANTIFIER or tag == "not":
        _occurrences(f[1], out)
    elif tag in BINARY:
        _occurrences(f[1], out)
        _occurrences(f[2], out)


def _show_binder(b):
    return b if isinstance(b, str) else "⟨" + ", ".join(_show_binder(x) for x in b[1]) + "⟩"


def _show_term(t, curry=True):
    """A proof term as text; with `curry`, directly nested `fun`s share one."""
    tag = t[0]
    if tag == "lam":
        binders, body = [_show_binder(t[1])], t[2]
        while curry and body[0] == "lam":
            binders.append(_show_binder(body[1]))
            body = body[2]
        return f"fun {' '.join(binders)} => {_show_term(body, curry)}"
    if tag == "var":
        return t[1]
    if tag == "const":
        return " ".join([t[1]] + [_show_atom(a, curry) for a in t[2]])
    if tag == "app":
        return " ".join([_show_atom(t[1], curry)] + [_show_atom(a, curry) for a in t[2]])
    if tag == "anon":
        return "⟨" + ", ".join(_show_term(x, curry) for x in t[1]) + "⟩"
    return f"{_show_atom(t[1], curry)}.{t[2]}"  # proj


def _show_atom(t, curry):
    s = _show_term(t, curry)
    return s if t[0] in ("var", "proj", "anon") or (t[0] == "const" and not t[2]) else f"({s})"


def _hyp_name(k):
    return f"h{k}"


def _rename(t, free=None, hyp=_hyp_name, ind=_variable, unused=None):
    """t with its binders renamed by binding order: the k-th named hypothesis
    in scope becomes hyp(k), the k-th named individual ind(k), and a binder
    nothing refers to `_` (or unused(k) for the k-th such, when given). A
    binder is a hypothesis iff its name starts with "h", as in the schemas and
    the references. `free` renames free variables (the declared individuals)."""
    free = free or {}
    used, scope = set(), []
    ids = itertools.count()

    def mark(t):
        tag = t[0]
        if tag == "lam":
            names = []
            _binder_names(t[1], names)
            scope.extend((name, next(ids)) for name in names)
            mark(t[2])
            del scope[len(scope) - len(names):]
        elif tag == "var":
            for name, k in reversed(scope):
                if name == t[1]:
                    used.add(k)
                    break
        elif tag == "proj":
            mark(t[1])
        elif tag == "app":
            mark(t[1])
            for a in t[2]:
                mark(a)
        else:
            for a in t[2] if tag == "const" else t[1]:
                mark(a)

    mark(t)
    ids, spare = itertools.count(), itertools.count()

    def name(b):
        if not isinstance(b, str):
            items = [name(x) for x in b[1]]
            return "_" if all(x == "_" for x in items) else ("pat", items)
        k = next(ids)
        if b == "_" or k not in used:
            new, kind = ("_" if unused is None else unused(next(spare))), None
        else:
            kind = "h" if b.startswith("h") else "i"
            count = sum(1 for _, _, c in scope if c == kind)
            new = hyp(count) if kind == "h" else ind(count)
        scope.append((b, new, kind))
        return new

    def assign(t):
        tag = t[0]
        if tag == "lam":
            n = len(scope)
            binder = name(t[1])
            body = assign(t[2])
            del scope[n:]
            return ("lam", binder, body)
        if tag == "var":
            for old, new, _ in reversed(scope):
                if old == t[1]:
                    return ("var", new)
            return ("var", free.get(t[1], t[1]))
        if tag == "proj":
            return ("proj", assign(t[1]), t[2])
        if tag == "app":
            head = assign(t[1])
            return ("app", head, [assign(a) for a in t[2]])
        if tag == "const":
            return ("const", t[1], [assign(a) for a in t[2]])
        return ("anon", [assign(a) for a in t[1]])

    return assign(t)


# ---------------------------------------------------------------- schemas
# weight | formula over slots A B C (unary), E (binary), S T (closed) and
# individuals c d | its proof(s), which never look inside a slot. Every binder
# is named here (hypotheses h..., individuals x y z); _rename turns the unused
# ones into `_` and numbers the rest. Of two proofs the shorter is the
# reference (the first on a tie). Weights: 3 one-step, 2 a few steps, 1 long.
_SCHEMA_TABLE = """
3 | (∀ x, A x) → A c | fun h0 => h0 c
3 | (∀ x, A x → B x) → A c → B c | fun h0 => h0 c
3 | A c → ∃ x, A x | fun h0 => ⟨c, h0⟩
3 | (∀ x, ∀ y, E x y) → E c d | fun h0 => h0 c d
3 | (∀ x, A x ∧ B x) → A c | fun h0 => (h0 c).1
3 | (∀ x, A x ∧ B x) → ∀ x, B x | fun h0 x => (h0 x).2
3 | (∀ x, A x) → ∀ x, A x ∨ B x | fun h0 x => Or.inl (h0 x)
3 | A c → B c → ∃ x, A x ∧ B x | fun h0 h1 => ⟨c, h0, h1⟩
3 | S → ∀ x, A x ∨ S | fun h0 x => Or.inr h0
3 | ∀ x, A x → ∃ y, A y | fun x h0 => ⟨x, h0⟩
3 | (∀ x, A x ↔ B x) → B c → A c | fun h0 => (h0 c).2
3 | (∀ x, A x) → (∀ x, A x → B x) → B c | fun h0 h1 => h1 c (h0 c)
3 | (∀ x, ∀ y, E x y) → ∀ x, E x x | fun h0 x => h0 x x
3 | ∀ x, ∀ y, E x y → ∃ z, E x z | fun x y h0 => ⟨y, h0⟩
3 | (∀ x, A x) → (∀ x, B x) → ∀ x, A x ∧ B x | fun h0 h1 x => ⟨h0 x, h1 x⟩
3 | (∃ x, A x ∧ B x) → ∃ x, A x | fun ⟨x, h0⟩ => ⟨x, h0.1⟩ | fun ⟨x, h0, h1⟩ => ⟨x, h0⟩
3 | (∃ x, A x ∧ B x) → ∃ x, B x | fun ⟨x, h0⟩ => ⟨x, h0.2⟩ | fun ⟨x, h0, h1⟩ => ⟨x, h1⟩
3 | (∀ x, A x) → ∀ x, ¬¬A x | fun h0 x h1 => h1 (h0 x)
3 | (∀ x, E x x) → ∀ x, ∃ y, E x y | fun h0 x => ⟨x, h0 x⟩
3 | (∀ x, A x → B c) → A c → ∃ x, B x | fun h0 h1 => ⟨c, h0 c h1⟩
3 | ∀ x, (∀ y, E x y) → E x x | fun x h0 => h0 x
3 | (∀ x, True → A x) → ∀ x, A x | fun h0 x => h0 x trivial
2 | (∀ x, A x → B x) → (∀ x, B x → C x) → ∀ x, A x → C x | fun h0 h1 x h2 => h1 x (h0 x h2)
2 | (∀ x, A x → S) → (S → T) → ∀ x, A x → T | fun h0 h1 x h2 => h1 (h0 x h2)
2 | (∀ x, A x → B x) → ¬B c → ¬A c | fun h0 h1 h2 => h1 (h0 c h2)
2 | (∀ x, A x ∨ B x) → ¬A c → B c | fun h0 h1 => (h0 c).elim (fun h2 => (h1 h2).elim) (fun h2 => h2)
2 | (∀ x, A x ∨ B x) → ¬B c → A c | fun h0 h1 => (h0 c).elim (fun h2 => h2) (fun h2 => (h1 h2).elim)
2 | (∃ x, A x) → (∀ x, A x → B x) → ∃ x, B x | fun ⟨x, h0⟩ h1 => ⟨x, h1 x h0⟩
2 | (∃ x, A x ∧ B x) → (∃ x, A x) ∧ ∃ x, B x | fun ⟨x, h0⟩ => ⟨⟨x, h0.1⟩, x, h0.2⟩ | fun ⟨x, h0, h1⟩ => ⟨⟨x, h0⟩, x, h1⟩
2 | (∀ x, A x ∧ B x) → (∀ x, A x) ∧ ∀ x, B x | fun h0 => ⟨fun x => (h0 x).1, fun x => (h0 x).2⟩
2 | (∀ x, A x) ∧ (∀ x, B x) → ∀ x, A x ∧ B x | fun h0 x => ⟨h0.1 x, h0.2 x⟩ | fun ⟨h0, h1⟩ x => ⟨h0 x, h1 x⟩
2 | (∀ x, ¬A x) → ¬∃ x, A x | fun h0 ⟨x, h1⟩ => h0 x h1
2 | (¬∃ x, A x) → ∀ x, ¬A x | fun h0 x h1 => h0 ⟨x, h1⟩
2 | (∃ x, ¬A x) → ¬∀ x, A x | fun ⟨x, h0⟩ h1 => h0 (h1 x)
2 | (∃ x, A x) → ¬∀ x, ¬A x | fun ⟨x, h0⟩ h1 => h1 x h0
2 | (∀ x, A x) → ¬∃ x, ¬A x | fun h0 ⟨x, h1⟩ => h1 (h0 x)
2 | (∀ x, S → A x) → S → ∀ x, A x | fun h0 h1 x => h0 x h1
2 | (S → ∀ x, A x) → ∀ x, S → A x | fun h0 x h1 => h0 h1 x
2 | (∃ x, S ∧ A x) → S ∧ ∃ x, A x | fun ⟨x, h0, h1⟩ => ⟨h0, x, h1⟩
2 | S ∧ (∃ x, A x) → ∃ x, S ∧ A x | fun ⟨h0, x, h1⟩ => ⟨x, h0, h1⟩
2 | (∃ x, A x) → (∀ x, A x → S) → S | fun ⟨x, h0⟩ h1 => h1 x h0
2 | (∀ x, A x → S) → (∃ x, A x) → S | fun h0 ⟨x, h1⟩ => h0 x h1
2 | (∀ x, ∀ y, E x y) → ∀ x, ∀ y, E y x | fun h0 x y => h0 y x
2 | (∃ x, ∀ y, E x y) → ∀ x, ∃ y, E y x | fun ⟨x, h0⟩ y => ⟨x, h0 y⟩
2 | (∃ x, ∃ y, E x y) → ∃ x, ∃ y, E y x | fun ⟨x, y, h0⟩ => ⟨y, x, h0⟩
2 | (∀ x, A x ↔ B x) → ∀ x, B x ↔ A x | fun h0 x => ⟨(h0 x).2, (h0 x).1⟩
2 | (∀ x, A x → B x ∧ C x) → A c → C c | fun h0 h1 => (h0 c h1).2
2 | (∀ x, A x) → ¬A c → S | fun h0 h1 => absurd (h0 c) h1 | fun h0 h1 => (h1 (h0 c)).elim
2 | (∀ x, ∀ y, E x y → E y x) → E c d → E d c | fun h0 => h0 c d
2 | (∀ x, A x → B x) → (∃ x, A x ∧ C x) → ∃ x, B x ∧ C x | fun h0 ⟨x, h1, h2⟩ => ⟨x, h0 x h1, h2⟩
2 | (∀ x, A x ∨ B x) → (∀ x, A x → C x) → (∀ x, B x → C x) → ∀ x, C x | fun h0 h1 h2 x => (h0 x).elim (h1 x) (h2 x)
2 | (∀ x, A x → B x) → (∀ x, A x) → ∀ x, B x | fun h0 h1 x => h0 x (h1 x)
2 | (∀ x, A x) → (∃ x, B x) → ∃ x, A x ∧ B x | fun h0 ⟨x, h1⟩ => ⟨x, h0 x, h1⟩
2 | (∃ x, ∀ y, E x y) → ∃ x, E x x | fun ⟨x, h0⟩ => ⟨x, h0 x⟩
2 | (∀ x, ∃ y, E x y) → (∀ x, ∀ y, E x y → A x) → ∀ x, A x | fun h0 h1 x => (h0 x).elim (h1 x)
2 | ¬(∀ x, A x) → ¬∀ x, A x ∧ B x | fun h0 h1 => h0 (fun x => (h1 x).1)
2 | (∀ x, A x) → (∃ x, ¬A x) → S | fun h0 ⟨x, h1⟩ => absurd (h0 x) h1 | fun h0 ⟨x, h1⟩ => (h1 (h0 x)).elim
2 | (∀ x, A x → False) → A c → S | fun h0 h1 => (h0 c h1).elim
2 | ∀ x, ∃ y, A x → A y | fun x => ⟨x, fun h0 => h0⟩
2 | (∀ x, A x ∧ B x → C x) → ∀ x, A x → B x → C x | fun h0 x h1 h2 => h0 x ⟨h1, h2⟩
2 | (∀ x, A x → B x → C x) → ∀ x, A x ∧ B x → C x | fun h0 x h1 => h0 x h1.1 h1.2 | fun h0 x ⟨h1, h2⟩ => h0 x h1 h2
2 | (∀ x, A x → B x) → (∀ x, A x → C x) → ∀ x, A x → B x ∧ C x | fun h0 h1 x h2 => ⟨h0 x h2, h1 x h2⟩
2 | (∀ x, A x ∧ B x) → ∀ x, B x ∧ A x | fun h0 x => ⟨(h0 x).2, (h0 x).1⟩
2 | (∃ x, A x ∧ B x) → ∃ x, B x ∧ A x | fun ⟨x, h0, h1⟩ => ⟨x, h1, h0⟩
2 | (∀ x, A x → ¬B x) → ¬∃ x, A x ∧ B x | fun h0 ⟨x, h1, h2⟩ => h0 x h1 h2
2 | ¬(∃ x, A x ∧ B x) → ∀ x, A x → ¬B x | fun h0 x h1 h2 => h0 ⟨x, h1, h2⟩
2 | (∃ x, A x ∧ B x) → ¬∀ x, A x → ¬B x | fun ⟨x, h0, h1⟩ h2 => h2 x h0 h1
2 | ¬(∃ x, A x ∨ B x) → ∀ x, ¬A x | fun h0 x h1 => h0 ⟨x, Or.inl h1⟩
2 | (∃ x, S → A x) → S → ∃ x, A x | fun ⟨x, h0⟩ h1 => ⟨x, h0 h1⟩
2 | (∃ x, A x → S) → (∀ x, A x) → S | fun ⟨x, h0⟩ h1 => h0 (h1 x)
2 | (∀ x, A x ↔ B x) → ¬A c → ¬B c | fun h0 h1 h2 => h1 ((h0 c).2 h2)
2 | (∀ x, A x ↔ ¬B x) → B c → ¬A c | fun h0 h1 h2 => (h0 c).1 h2 h1
2 | (∀ x, ∀ y, E x y → E y x) → ∀ x, ∀ y, E x y ↔ E y x | fun h0 x y => ⟨h0 x y, h0 y x⟩
2 | (∀ x, A x → ∀ y, E x y) → A c → E c d | fun h0 h1 => h0 c h1 d
2 | (∀ x, ∀ y, E x y → A x) → (∃ x, ∃ y, E x y) → ∃ x, A x | fun h0 ⟨x, y, h1⟩ => ⟨x, h0 x y h1⟩
2 | (∀ x, A x → B x) → (∀ x, B x → C x) → A c → C c | fun h0 h1 h2 => h1 c (h0 c h2)
2 | (∀ x, A x ∨ B x) → (∀ x, ¬A x) → ∀ x, B x | fun h0 h1 x => (h0 x).elim (fun h2 => (h1 x h2).elim) (fun h2 => h2)
2 | (∀ x, A x ∨ S) → ¬S → ∀ x, A x | fun h0 h1 x => (h0 x).elim (fun h2 => h2) (fun h2 => (h1 h2).elim)
2 | ¬¬¬(∀ x, A x) → ¬∀ x, A x | fun h0 h1 => h0 (fun h2 => h2 h1)
2 | ¬¬¬(∃ x, A x) → ¬∃ x, A x | fun h0 h1 => h0 (fun h2 => h2 h1)
2 | (∃ x, A x ∧ ¬A x) → S | fun ⟨x, h0, h1⟩ => (h1 h0).elim | fun ⟨x, h0, h1⟩ => absurd h0 h1
2 | (∀ x, A x → S) → (∀ x, B x → S) → (∃ x, A x ∨ B x) → S | fun h0 h1 ⟨x, h2⟩ => h2.elim (h0 x) (h1 x)
2 | (∃ x, ¬¬A x) → ¬∀ x, ¬A x | fun ⟨x, h0⟩ h1 => h0 (h1 x)
2 | (∀ x, ∀ y, E x y) → ∀ x, ∃ y, E y x | fun h0 x => ⟨x, h0 x x⟩
2 | (∀ x, A x ∨ B x) → ∀ x, B x ∨ A x | fun h0 x => (h0 x).elim (fun h1 => Or.inr h1) (fun h1 => Or.inl h1)
1 | (∀ x, A x) ∨ (∀ x, B x) → ∀ x, A x ∨ B x | fun h0 x => h0.elim (fun h1 => Or.inl (h1 x)) (fun h1 => Or.inr (h1 x))
1 | (∃ x, A x) ∨ (∃ x, B x) → ∃ x, A x ∨ B x | fun h0 => h0.elim (fun ⟨x, h1⟩ => ⟨x, Or.inl h1⟩) (fun ⟨x, h1⟩ => ⟨x, Or.inr h1⟩)
1 | (∃ x, A x ∨ B x) → (∃ x, A x) ∨ ∃ x, B x | fun ⟨x, h0⟩ => h0.elim (fun h1 => Or.inl ⟨x, h1⟩) (fun h1 => Or.inr ⟨x, h1⟩)
1 | (¬∃ x, A x) ↔ ∀ x, ¬A x | ⟨fun h0 x h1 => h0 ⟨x, h1⟩, fun h0 ⟨x, h1⟩ => h0 x h1⟩
1 | ((∃ x, A x) → S) ↔ ∀ x, A x → S | ⟨fun h0 x h1 => h0 ⟨x, h1⟩, fun h0 ⟨x, h1⟩ => h0 x h1⟩
1 | (∀ x, A x ∧ B x) ↔ (∀ x, A x) ∧ ∀ x, B x | ⟨fun h0 => ⟨fun x => (h0 x).1, fun x => (h0 x).2⟩, fun h0 x => ⟨h0.1 x, h0.2 x⟩⟩
1 | (∀ x, A x ↔ B x) → ((∀ x, A x) ↔ ∀ x, B x) | fun h0 => ⟨fun h1 x => (h0 x).1 (h1 x), fun h1 x => (h0 x).2 (h1 x)⟩
1 | (∀ x, A x ↔ B x) → ((∃ x, A x) ↔ ∃ x, B x) | fun h0 => ⟨fun ⟨x, h1⟩ => ⟨x, (h0 x).1 h1⟩, fun ⟨x, h1⟩ => ⟨x, (h0 x).2 h1⟩⟩
1 | (∀ x, S → A x) ↔ (S → ∀ x, A x) | ⟨fun h0 h1 x => h0 x h1, fun h0 x h1 => h0 h1 x⟩
1 | (∃ x, S ∧ A x) ↔ S ∧ ∃ x, A x | ⟨fun ⟨x, h0, h1⟩ => ⟨h0, x, h1⟩, fun ⟨h0, x, h1⟩ => ⟨x, h0, h1⟩⟩
1 | (∃ x, A x ∨ B x) ↔ (∃ x, A x) ∨ ∃ x, B x | ⟨fun ⟨x, h0⟩ => h0.elim (fun h1 => Or.inl ⟨x, h1⟩) (fun h1 => Or.inr ⟨x, h1⟩), fun h0 => h0.elim (fun ⟨x, h1⟩ => ⟨x, Or.inl h1⟩) (fun ⟨x, h1⟩ => ⟨x, Or.inr h1⟩)⟩
1 | (∀ x, A x ↔ B x) → (∀ x, B x ↔ C x) → ∀ x, A x ↔ C x | fun h0 h1 x => ⟨fun h2 => (h1 x).1 ((h0 x).1 h2), fun h2 => (h0 x).2 ((h1 x).2 h2)⟩
1 | ¬¬(∀ x, A x) → ∀ x, ¬¬A x | fun h0 x h1 => h0 (fun h2 => h1 (h2 x))
1 | (¬∀ x, ¬A x) → ¬¬∃ x, A x | fun h0 h1 => h0 (fun x h2 => h1 ⟨x, h2⟩)
1 | (∃ x, A x ∨ B x) → ∃ x, B x ∨ A x | fun ⟨x, h0⟩ => ⟨x, h0.elim (fun h1 => Or.inr h1) (fun h1 => Or.inl h1)⟩
1 | (∀ x, A x) ∨ S → ∀ x, A x ∨ S | fun h0 x => h0.elim (fun h1 => Or.inl (h1 x)) (fun h1 => Or.inr h1)
1 | (∀ x, A x ∧ B x) ∧ (∃ x, C x) → ∃ x, A x ∧ C x | fun ⟨h0, x, h1⟩ => ⟨x, (h0 x).1, h1⟩
"""

_SLOTS = {"A": 1, "B": 1, "C": 1, "E": 2, "S": 0, "T": 0}
_PLACEHOLDERS = ["c", "d"]


def _load_schemas():
    """(weight, formula, its slots and placeholders in order of first use, reference proof)."""
    out = []
    for line in _SCHEMA_TABLE.strip().splitlines():
        weight, formula, *proofs = (s.strip() for s in line.split("|"))
        goal = _parse_formula(formula, _SLOTS, _PLACEHOLDERS)
        order = []
        _occurrences(goal, order)
        terms = [_rename(_parse_term(p, _PLACEHOLDERS)) for p in proofs]
        best = min(terms, key=lambda t: len(_show_term(t)))  # min keeps the first on a tie
        out.append((int(weight), goal, [s for s in order if s in _SLOTS],
                    [s for s in order if s in _PLACEHOLDERS], best))
    return out


_SCHEMAS = _load_schemas()
_CUM_WEIGHTS = list(itertools.accumulate(s[0] for s in _SCHEMAS))


def _plug(body, args, k=0):
    """A slot's body with its holes (the indices it leaves loose, the last
    argument index 0) filled with `args`, at k binders inside the body."""
    tag = body[0]
    if tag == "pred":
        out = []
        for x in body[2]:
            if isinstance(x, int) and x >= k:
                a = args[len(args) - 1 - (x - k)]
                out.append(a + k if isinstance(a, int) else a)
            else:
                out.append(x)
        return ("pred", body[1], tuple(out))
    if tag in QUANTIFIER:
        return (tag, _plug(body[1], args, k + 1))
    if tag == "not":
        return ("not", _plug(body[1], args, k))
    if tag in BINARY:
        return (tag, _plug(body[1], args, k), _plug(body[2], args, k))
    return body


def _substitute(f, bodies, individuals):
    """The schema formula f with each slot replaced by its body and c, d by `individuals`."""
    tag = f[0]
    if tag == "pred":
        return _plug(bodies[f[1]], tuple(individuals.get(x, x) if isinstance(x, str) else x for x in f[2]))
    if tag in QUANTIFIER or tag == "not":
        return (tag, _substitute(f[1], bodies, individuals))
    if tag in BINARY:
        return (tag, _substitute(f[1], bodies, individuals), _substitute(f[2], bodies, individuals))
    return f


def _abstract(f, name, k=0):
    """f with the individual `name` made the variable of a new outermost ∀."""
    tag = f[0]
    if tag == "pred":
        return ("pred", f[1], tuple(k if x == name else x for x in f[2]))
    if tag in QUANTIFIER:
        return (tag, _abstract(f[1], name, k + 1))
    if tag == "not":
        return ("not", _abstract(f[1], name, k))
    if tag in BINARY:
        return (tag, _abstract(f[1], name, k), _abstract(f[2], name, k))
    return f


def _atom(rng, pool, terms, need, keep):
    """An atom with a symbol taken from `pool` (and removed from it unless
    `keep`) and arguments from `terms` (hole indices and individuals) that
    mentions every index in `need`; None if no symbol in the pool fits. With
    nothing to mention it is sometimes True or False."""
    if not need and rng.random() < 0.04:
        return rng.choice((TOP, BOT))
    fits = [s for s in pool if ARITY[s] >= len(need)]
    if not fits:
        return None
    sym = rng.choice(fits)
    if not keep:
        pool.remove(sym)
    args = list(need) + [rng.choice(terms) for _ in range(ARITY[sym] - len(need))]
    rng.shuffle(args)
    return ("pred", sym, tuple(args))


def _body(rng, pool, holes, size, keep):
    """A random formula from the symbols in `pool` that mentions every hole
    (loose index) in `holes`. Size 0: an atom; 1: a negated atom or two atoms
    joined by ∧ ∨ →; 2: a quantifier over a smaller formula, or an atom joined
    with a size-1 formula. None if the symbols run out."""
    terms = list(holes) * 2 + list(INDIVIDUALS)
    if size == 0:
        return _atom(rng, pool, terms, holes, keep)
    if size == 1 and rng.random() < 0.3:
        a = _atom(rng, pool, terms, holes, keep)
        return a and ("not", a)
    if size == 2 and len(holes) < 2 and rng.random() < 0.5:
        body = _body(rng, pool, [h + 1 for h in holes] + [0], rng.randrange(2), keep)
        return body and (rng.choice(QUANTIFIER), body)
    split = rng.randrange(len(holes) + 1)
    a, b = _body(rng, pool, holes[:split], size - 1, keep), _atom(rng, pool, terms, holes[split:], keep)
    if a is None or b is None:
        return None
    if rng.random() < 0.2:
        b = ("not", b)
    if rng.random() < 0.5:
        a, b = b, a
    return (rng.choice(("and", "or", "imp")), a, b)


# How big slot bodies are (weights of sizes 0, 1, 2) by the number of slots
# in the schema: one slot gets room to vary, three get small fillings.
_SIZE_WEIGHTS = {1: (2, 4, 4), 2: (4, 4, 2), 3: (6, 3, 1)}


# ---------------------------------------------------------------- shortest proofs
# A schema's proof is the shortest for the schema, but the formulas in its
# slots can open shorter routes: with the conclusion ¬B a proof can take B
# first and close False without ex falso, an ∃ hypothesis h1 next to a ∀
# hypothesis h0 of the right shape is eliminated by `h1.elim h0`, shorter than
# a pattern, and a repeated symbol or a False can make a premise redundant or
# the premises contradictory.
# So each reference goes through _Search, which looks for anything shorter:
# depth-first branch and bound over the proof terms the checker accepts,
# costed by printed length (hypotheses 2 characters, individuals 1, an unused
# binder `_`), with a memo per sequent and iterative deepening per subgoal
# (a sample whose search gives up is dropped). It leaves out only forms never
# shorter than one it tries: And.intro, Iff.intro and Exists.intro (⟨⟩
# instead), Or.elim, Exists.elim and False.elim (.elim), And.left, .left, .mp,
# ... (.1 and .2), absurd h h' ((h' h).elim, as long), η-expansions, and a
# second elimination of the same ∨ or ∃. Two approximations: patterns have at
# most four items, and ex falso from a proof of False that `.elim` cannot take
# (a case analysis) is tried as `absurd h (fun _ => ...)` with h a hypothesis,
# not as `absurd t (fun h' => ...)` with t a compound term, which is shorter
# only when the proof of False uses t at least twice.

_TOP, _ARG, _LAM, _TAIL = range(4)  # where a term is printed: alone, as an argument, as a fun body, last in ⟨⟩
_SEARCH_STEPS = 20_000  # sequents searched before giving up (the 99th percentile needs under 1,000)
_INF = float("inf")


class _GiveUp(Exception):
    pass


def _placed(cost, term, pos):
    """The printed length at pos of a term whose length alone is cost: an
    argument needs parentheses unless atomic, a fun in a fun body shares its
    `fun` and `=>`, and a ⟨⟩ last in a ⟨⟩ shares its brackets."""
    tag = term[0]
    if pos == _ARG and (tag in ("app", "lam") or (tag == "const" and term[2])):
        return cost + 2
    if pos == _LAM and tag == "lam":
        return cost - 7
    if pos == _TAIL and tag == "anon":
        return cost - 2
    return cost


def _binder_length(binder):
    if isinstance(binder, str):
        return 2 if binder.startswith("h") else 1
    return sum(_binder_length(x) for x in binder[1]) + 2 * len(binder[1])


def _prune(binder, free):
    """The binder with names the body does not use as `_` (a pattern of
    nothing but `_` is `_`), as _rename prints it. A pattern can bind one
    formula, so one name, twice (⟨x, h, h⟩ for ∃ x, P x ∧ P x); only the last
    is in scope."""
    names = []
    _binder_names(binder, names)
    position = itertools.count()

    def prune(b):
        if isinstance(b, str):
            k = next(position)
            return b if b in free and b not in names[k + 1:] else "_"
        items = tuple(prune(x) for x in b[1])
        return "_" if all(x == "_" for x in items) else ("pat", items)

    return prune(binder)


class _Search:
    """Cheapest proofs of sequents within a budget. A proof is (cost, term,
    the names it uses). Hypotheses are named by their formula, so a memoised
    proof fits every context with the same hypotheses; individuals bound in
    a proof are named by level (x2 is the third individual in scope)."""

    def __init__(self, steps):
        self.steps = steps
        self.memo = {}  # (hypotheses, individuals, goal, position) -> (lower bound, cheapest proof or None)
        self.names = {}
        self.paths_memo = {}
        self.active = {}  # sequents being searched -> depth: a cheapest proof never contains itself
        self.low = _INF  # the shallowest active sequent that a search cut short depends on

    def name(self, f):
        if f not in self.names:
            self.names[f] = f"h{len(self.names)}"
        return self.names[f]

    def prove(self, hyps, inds, goal, pos, budget):
        """A cheapest proof of goal from hyps (a tuple of formulas) and the
        individuals inds, printed at pos, if one costs at most budget."""
        if budget < 2:
            return None
        key = (frozenset(hyps), inds, goal, pos)
        lower, best = self.memo.get(key, (2, None))
        if best is not None:
            return best if best[0] <= budget else None
        if lower > budget:
            return None
        if key in self.active:
            self.low = min(self.low, self.active[key])
            return None
        self.steps -= 1
        if self.steps < 0:
            raise _GiveUp
        depth = len(self.active)
        self.active[key] = depth
        outer, self.low = self.low, _INF
        # Iterative deepening, except at the root, which must search its whole
        # budget anyway: a subgoal's cheap proof is found at a small budget
        # before a large one lets the search wander down dead ends.
        b = lower if depth else budget
        while True:
            best = self.search(hyps, inds, goal, pos, b)
            if best is not None or b >= budget:
                break
            b = min(budget, b + max(8, 2 * b))
        del self.active[key]
        if self.low >= depth:  # not cut short by a sequent still being searched: exact
            self.memo[key] = (best[0], best) if best is not None else (budget + 1, None)
            self.low = _INF
        self.low = min(outer, self.low)
        return best

    def search(self, hyps, inds, goal, pos, bound):
        best = None

        def consider(proof):
            nonlocal best, bound
            if proof is not None and proof[0] <= bound:
                best, bound = proof, proof[0] - 1

        tag = goal[0]
        if goal == TOP:
            consider((7, ("const", "trivial", ()), frozenset()))
        for h in hyps:  # a hypothesis applied, instantiated or projected to the goal
            for ops, ty in self.paths(h, inds):
                if ty == goal:
                    consider(self.spine(hyps, inds, h, ops, None, pos, bound))
        if tag == "imp":
            for binder, added, new in self.patterns(goal[1], inds):
                more = hyps + tuple(f for f in added if f not in hyps)
                consider(self.intro(more, inds + new, binder, goal[2], pos, bound))
        elif tag == "all":
            x = f"x{len(inds)}"
            consider(self.intro(hyps, inds + (x,), x, _open(goal[1], x), pos, bound))
        elif tag in ("and", "iff", "ex"):
            consider(self.pair(hyps, inds, goal, pos, bound))
        elif tag == "or":
            fixed = 7 + (2 if pos == _ARG else 0)
            for side, name in ((1, "Or.inl"), (2, "Or.inr")):
                arg = self.prove(hyps, inds, goal[side], _ARG, bound - fixed)
                if arg is not None:
                    consider((fixed + arg[0], ("const", name, (arg[1],)), arg[2]))
        for h in hyps:  # case analysis on an ∨ or ∃, or ex falso
            for ops, ty in self.paths(h, inds):
                if ty[0] == "or" and ty[1] not in hyps and ty[2] not in hyps:
                    cases = (("imp", ty[1], goal), ("imp", ty[2], goal))
                elif ty[0] == "ex" and all(_open(ty[1], w) not in hyps for w in inds):
                    cases = (("all", ("imp", ty[1], goal)),)
                elif ty == BOT and goal != BOT:
                    cases = ()
                else:
                    continue
                consider(self.spine(hyps, inds, h, ops, cases, pos, bound))
        if goal != BOT and hyps:
            # Ex falso from a proof of False that `.elim` cannot take, because the
            # checker cannot infer its type (a case analysis): absurd h (fun _ => ...)
            # with h a hypothesis the proof uses anyway, as in `(∀ x, P x ∨ P x) →
            # ¬P b → P b`, proved by `fun h0 h1 => absurd h0 (fun _ => (h0 b).elim h1 h1)`.
            fixed = 21 + (2 if pos == _ARG else 0)
            falsum = self.prove(hyps, inds, BOT, _LAM, bound - fixed)
            if falsum is not None:
                h = next((self.name(f) for f in hyps if self.name(f) in falsum[2]), self.name(hyps[0]))
                consider((fixed + falsum[0], ("const", "absurd", (("var", h), ("lam", "_", falsum[1]))),
                          falsum[2] | {h}))
        return best

    def paths(self, f, inds):
        """(steps, type) for every way of using a hypothesis of type f: each
        step instantiates a ∀ ("ind", w), discharges a premise ("arg", A) or
        projects an ∧ or ↔ ("proj", "1" or "2")."""
        key = (f, inds)
        if key not in self.paths_memo:
            out = []

            def walk(ops, ty):
                out.append((ops, ty))
                if ty[0] == "all":
                    for w in inds:
                        walk(ops + (("ind", w),), _open(ty[1], w))
                elif ty[0] == "imp":
                    walk(ops + (("arg", ty[1]),), ty[2])
                elif ty[0] in ("and", "iff"):
                    first, second = _fields(ty)
                    walk(ops + (("proj", "1"),), first)
                    walk(ops + (("proj", "2"),), second)

            walk((), f)
            self.paths_memo[key] = out
        return self.paths_memo[key]

    def spine(self, hyps, inds, h, ops, cases, pos, bound):
        """h along ops, then, unless cases is None, `.elim` with a proof of
        each case: the premises and cases get the slack in turn."""
        fixed, app, goals = 2, False, []
        for op in ops:
            if op[0] == "proj":
                fixed, app = fixed + (4 if app else 2), False
            else:
                fixed, app = fixed + (2 if op[0] == "ind" else 1), True
                if op[0] == "arg":
                    goals.append(op[1])
        if cases is not None:
            fixed, app = fixed + (2 if app else 0) + 5 + len(cases), bool(cases)
            goals += cases
        fixed += 2 if pos == _ARG and app else 0
        slack = bound - fixed - 2 * len(goals)
        if slack < 0:
            return None
        proofs = []
        for g in goals:
            proof = self.prove(hyps, inds, g, _ARG, 2 + slack)
            if proof is None:
                return None
            proofs.append(proof)
            slack -= proof[0] - 2
        name = self.name(h)
        term, args, free = ("var", name), [], {name}
        given = iter(proofs)
        for op in ops:
            if op[0] == "proj":
                if args:
                    term, args = ("app", term, tuple(args)), []
                term = ("proj", term, op[1])
            elif op[0] == "ind":
                args.append(("var", op[1]))
                free.add(op[1])
            else:
                args.append(next(given)[1])
        if args:
            term = ("app", term, tuple(args))
        if cases is not None:
            term = ("proj", term, "elim")
            if cases:
                term = ("app", term, tuple(p[1] for p in proofs[len(proofs) - len(cases):]))
        for p in proofs:
            free |= p[2]
        return fixed + sum(p[0] for p in proofs), term, frozenset(free)

    def intro(self, hyps, inds, binder, body_goal, pos, bound):
        """fun binder => (a proof of body_goal), hyps and inds already extended."""
        fixed = _placed(8, ("lam",), pos)
        body = self.prove(hyps, inds, body_goal, _LAM, bound - fixed - 1)
        if body is None:
            return None
        names = []
        _binder_names(binder, names)
        binder = _prune(binder, body[2])
        return fixed + _binder_length(binder) + body[0], ("lam", binder, body[1]), body[2] - set(names)

    def patterns(self, f, inds, depth=0):
        """(binder, hypotheses, individuals) for each way to bind a hypothesis
        of type f: a name, or a pattern taking an ∧ or ∃ apart."""
        out = [(self.name(f), (f,), ())]
        if depth < 3 and f[0] == "and":
            for first, h1, n1 in self.patterns(f[1], inds, depth + 1):
                for rest, h2, n2 in self.patterns(f[2], inds + n1, depth + 1):
                    items = (first,) + (rest[1] if isinstance(rest, tuple) else (rest,))
                    out.append((("pat", items), h1 + h2, n1 + n2))
        elif depth < 3 and f[0] == "ex":
            x = f"x{len(inds)}"
            for rest, h2, n2 in self.patterns(_open(f[1], x), inds + (x,), depth + 1):
                items = (x,) + (rest[1] if isinstance(rest, tuple) else (rest,))
                out.append((("pat", items), h2, (x,) + n2))
        return out

    def pair(self, hyps, inds, goal, pos, bound):
        """⟨a, b⟩ for ∧ and ↔ or ⟨w, b⟩ for ∃; a ⟨⟩ last in another merges into it."""
        fixed = _placed(4, ("anon",), pos)
        if goal[0] == "ex":
            best = None
            for w in inds:
                rest = self.prove(hyps, inds, _open(goal[1], w), _TAIL, bound - fixed - 1)
                if rest is not None:
                    items = (("var", w),) + (rest[1][1] if rest[1][0] == "anon" else (rest[1],))
                    best = (fixed + 1 + rest[0], ("anon", items), rest[2] | {w})
                    bound = best[0] - 1
            return best
        first, second = _fields(goal)
        a = self.prove(hyps, inds, first, _TOP, bound - fixed - 2)
        if a is None:
            return None
        b = self.prove(hyps, inds, second, _TAIL, bound - fixed - a[0])
        if b is None:
            return None
        items = (a[1],) + (b[1][1] if b[1][0] == "anon" else (b[1],))
        return fixed + a[0] + b[0], ("anon", items), a[2] | b[2]


def _shortest(goal, individuals, budget):
    """A shortest proof of goal if one is at most budget characters long, else
    None; raises _GiveUp if the search runs out of steps first."""
    found = _Search(_SEARCH_STEPS).prove((), tuple(individuals), _unfold(goal), _TOP, budget)
    return None if found is None else _show_term(_rename(found[1]))


class Fol(Domain):
    name = "fol"
    prefer_shorter = True

    def claims(self, prompt: str) -> bool:
        return _goal_of(prompt) is not None

    def verify(self, prompt: str, answer: str) -> bool:
        return verify(prompt, answer)

    def sample(self, rng: random.Random) -> Problem:
        while True:
            problem = self._instance(rng)
            if problem is not None:
                return problem

    def _instance(self, rng):
        """A random instance of a random schema, or None when it does not fit
        the length limits or runs out of symbols."""
        _, goal, slots, placeholders, proof = rng.choices(_SCHEMAS, cum_weights=_CUM_WEIGHTS)[0]
        # Mostly each symbol at most once, so that the slots are independent;
        # sometimes symbols repeat, as in `(∀ x, r x x → P x ∨ r x a) → r a a → P a ∨ r a a`.
        pool, keep = list(ARITY), rng.random() < 0.3
        bodies = {}
        for slot in slots:
            size = rng.choices((0, 1, 2), _SIZE_WEIGHTS[len(slots)])[0]
            body = _body(rng, pool, list(range(_SLOTS[slot] - 1, -1, -1)), size, keep)
            if body is None:
                return None
            bodies[slot] = body
        individuals = dict(zip(placeholders, rng.sample(INDIVIDUALS, len(placeholders))))
        goal = _substitute(goal, bodies, individuals)
        # Variety beyond the schemas: an individual generalised into an outer ∀
        # (which `fun x => ...` then proves), the two sides of an ↔ swapped,
        # an irrelevant first hypothesis.
        if placeholders and rng.random() < 0.15:
            c = rng.choice(placeholders)
            goal = ("all", _abstract(goal, individuals[c]))
            proof = ("lam", c, proof)
        if goal[0] == "iff" and proof[0] == "anon" and rng.random() < 0.5:
            goal = ("iff", goal[2], goal[1])
            proof = ("anon", proof[1][::-1])
        if rng.random() < 0.1:
            extra = _body(rng, pool, [], rng.choice((0, 1, 2)), keep)
            if extra is not None:
                goal = ("imp", extra, goal)
                proof = ("lam", "h", proof)
        prompt = _declaration(goal)
        reference = _show_term(_rename(proof, individuals))
        if _js_len(prompt) > MAX_SAMPLE_PROMPT or _js_len(reference) > MAX_REFERENCE \
                or len(prompt) + len(reference) + 3 > MAX_SEQUENCE:
            return None
        goal, individuals = _goal_of(prompt)
        try:
            shorter = _shortest(goal, individuals, len(reference) - 1)
        except _GiveUp:  # about 1 in 20,000: a shorter proof might exist
            return None
        if shorter is not None and len(shorter) < len(reference):
            reference = shorter
        if not verify(prompt, reference):
            raise RuntimeError(f"fol: a reference does not check: {prompt} {reference}")
        return Problem(self.name, prompt, reference)

    def document(self, problem: Problem, rng: random.Random) -> str:
        return f"{problem.prompt} {_variant(problem.prompt, problem.reference, rng)}"

    def split_key(self, prompt: str) -> str:
        """The prompt with each group's symbols renamed in order of first
        occurrence (P Q R, p q, a b), bound variables named by level and the
        fewest parentheses."""
        parsed = _goal_of(prompt)
        if parsed is None:
            return prompt
        order = []
        _occurrences(parsed[0], order)
        rename = {}
        for group in GROUPS:
            rename.update(zip([s for s in order if s in group], group))
        return _declaration(parsed[0], rename)


# ---------------------------------------------------------------- documents
# A document's proof is the reference with some steps traded for equivalent
# ones (a pattern for `.elim` or projections, `.elim` for Or.elim /
# Exists.elim / False.elim, absurd for `(h' h).elim`, .1 for .left / .mp /
# And.left, ⟨⟩ for a named constructor, a hypothesis for its η-expansion),
# other names, and `fun`s nested instead of merged. Whether a rewrite
# type-checks depends on types the term does not show (∧ or ↔? ∃ or ∧?), so
# each is kept only if the checker accepts the result.

def _nodes(t, out):
    """The subterms of t in pre-order."""
    out.append(t)
    tag = t[0]
    if tag == "lam":
        _nodes(t[2], out)
    elif tag == "proj":
        _nodes(t[1], out)
    elif tag == "app":
        _nodes(t[1], out)
        for a in t[2]:
            _nodes(a, out)
    elif tag != "var":
        for a in t[2] if tag == "const" else t[1]:
            _nodes(a, out)
    return out


def _replace(t, index, new, counter=None):
    """t with its index-th subterm (in pre-order) replaced by `new`."""
    counter = counter if counter is not None else itertools.count()
    if next(counter) == index:
        return new
    tag = t[0]
    if tag == "lam":
        return ("lam", t[1], _replace(t[2], index, new, counter))
    if tag == "proj":
        return ("proj", _replace(t[1], index, new, counter), t[2])
    if tag == "app":
        head = _replace(t[1], index, new, counter)
        return ("app", head, [_replace(a, index, new, counter) for a in t[2]])
    if tag == "const":
        return ("const", t[1], [_replace(a, index, new, counter) for a in t[2]])
    if tag == "anon":
        return ("anon", [_replace(a, index, new, counter) for a in t[1]])
    return t


def _substitute_vars(t, mapping):
    """t with the variables named in `mapping` replaced; a reference never
    reuses a name in scope, so nothing is captured."""
    tag = t[0]
    if tag == "var":
        return mapping.get(t[1], t)
    if tag == "lam":
        return ("lam", t[1], _substitute_vars(t[2], mapping))
    if tag == "proj":
        return ("proj", _substitute_vars(t[1], mapping), t[2])
    if tag == "app":
        return ("app", _substitute_vars(t[1], mapping), [_substitute_vars(a, mapping) for a in t[2]])
    if tag == "const":
        return ("const", t[1], [_substitute_vars(a, mapping) for a in t[2]])
    return ("anon", [_substitute_vars(a, mapping) for a in t[1]])


def _rewrites(t, fresh):
    """Forms equivalent to the node t when the types fit; `fresh` names a new
    hypothesis ("h...") and is used for nothing else."""
    tag = t[0]
    out = []
    if tag == "proj" and t[2] == "elim":
        out.append(("const", "False.elim", [t[1]]))
        if t[1][0] == "app":
            f, args = t[1][1], t[1][2]
            out.append(("const", "absurd", [args[-1], f if len(args) == 1 else ("app", f, args[:-1])]))
    elif tag == "app" and t[1][0] == "proj" and t[1][2] == "elim" and len(t[2]) in (1, 2):
        out.append(("const", "Or.elim" if len(t[2]) == 2 else "Exists.elim", [t[1][1]] + t[2]))
    elif tag == "const" and t[1] == "absurd":
        h, hn = t[2]
        app = ("app", hn[1], hn[2] + [h]) if hn[0] == "app" else ("app", hn, [h])
        out += [("proj", app, "elim"), ("const", "False.elim", [app])]
    elif tag == "proj" and t[2] in ("1", "2"):
        first = t[2] == "1"
        out += [("proj", t[1], "left" if first else "right"), ("proj", t[1], "mp" if first else "mpr"),
                ("const", "And.left" if first else "And.right", [t[1]])]
    elif tag == "anon":
        rest = t[1][1] if len(t[1]) == 2 else ("anon", t[1][1:])
        out += [("const", k, [t[1][0], rest]) for k in ("And.intro", "Iff.intro", "Exists.intro")]
    elif tag == "lam" and not isinstance(t[1], str):
        items, body, e = t[1][1], t[2], ("var", fresh)
        rest = items[1] if len(items) == 2 else ("pat", items[1:])
        out.append(("lam", fresh, ("app", ("proj", e, "elim"), [("lam", items[0], ("lam", rest, body))])))
        if len(items) == 2 and all(isinstance(x, str) for x in items):
            parts = {x: ("proj", e, field) for x, field in zip(items, "12") if x != "_"}
            out.append(("lam", fresh, _substitute_vars(body, parts)))
    return out


def _eta(t, fresh):
    """η-expansions of an inferable node: `fun h => t h` (t an implication)
    or `fun x => t x` (t a ∀). Both check either way, since names do not
    matter to the checker; _variant keeps the one named for its kind."""
    if t[0] not in ("var", "app", "proj"):
        return []
    return [("lam", name, ("app", t, [("var", name)])) for name in (fresh, "x" + fresh[1:])]


def _binds_hypothesis(prompt, term, index):
    """Whether the `fun` at node `index` of a term that checks binds a
    hypothesis rather than an individual: `(And.intro h h).1` can stand for
    h only if h is a hypothesis."""
    lam = _nodes(term, [])[index]
    h = ("var", lam[1])
    body = _substitute_vars(lam[2], {lam[1]: ("proj", ("const", "And.intro", [h, h]), "1")})
    return verify(prompt, _show_term(_replace(term, index, ("lam", lam[1], body))))


_HYP_SCHEMES = (_hyp_name, lambda k: f"h{k + 1}", lambda k: "h" + "'" * k, lambda k: f"h_{k}",
                lambda k: "hkm"[k] if k < 3 else f"h{k}")
_IND_SCHEMES = (_variable, lambda k: "uvw"[k] if k < 3 else f"u{k}", lambda k: f"x{k}",
                lambda k: "yzw"[k] if k < 3 else f"y{k}")


def _variant(prompt, reference, rng):
    """A random proof of `prompt` equivalent to `reference` that fits a
    document (the reference itself if no variant does)."""
    individuals = _goal_of(prompt)[1]
    t = reference_term = _parse_term(reference, individuals)
    for find, skip in ((_rewrites, 0.5), (_eta, 0.85)):
        # From the last node back, so a rewrite leaves the nodes before it in place.
        for index in reversed(range(len(_nodes(t, [])))):
            options = find(_nodes(t, [])[index], f"h{100 + index}")
            if not options or rng.random() < skip:
                continue
            rng.shuffle(options)
            for option in options:
                candidate = _replace(t, index, option)
                # _rename tells the kinds of binders apart by their names (h...)
                if verify(prompt, _show_term(candidate)) and (
                        find is _rewrites or _binds_hypothesis(prompt, candidate, index) == option[1].startswith("h")):
                    t = candidate
                    break
    budget = MAX_SEQUENCE - 3 - len(prompt)  # <bos> prompt " " answer <eos>
    hyp, ind = rng.choice(_HYP_SCHEMES), rng.choice(_IND_SCHEMES)
    spare = (lambda k: f"n{k}") if rng.random() < 0.3 else None
    curry = rng.random() < 0.8
    for term in (t, reference_term):
        answer = _show_term(_rename(term, hyp=hyp, ind=ind, unused=spare), curry)
        if len(answer) <= budget and verify(prompt, answer):
            return answer
    return reference


DOMAIN = Fol()
