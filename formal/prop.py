"""Intuitionistic propositional logic, as Lean 4 checks it.

    example (p q : Prop) : p ∧ q → q ∧ p :=        the model writes:   fun h0 => ⟨h0.2, h0.1⟩

Formulas over the atoms the header declares, with True, False and ¬ ∧ ∨ → ↔
in Lean 4 notation and precedence (¬ binds tightest, then ∧, ∨, →, ↔; ∧ ∨ →
associate to the right, ↔ not at all). An answer is a proof term or a tactic
block (`by intro h0; exact ⟨h0.2, h0.1⟩`).

The checker is a port of the Prop section of web/checker.js, which is its
specification: claims and verify agree with it on every input, so its limits
are measured in UTF-16 code units and its regular expressions keep
JavaScript's semantics. Proof terms are checked bidirectionally with ¬A
unfolded to A → False; tactic proofs are run on their goal as Lean runs
them, over a small fragment (intro, exact, apply, cases, rcases, obtain,
constructor, left, right, exfalso, assumption, contradiction, trivial, `·`).
Anything outside the fragment is rejected, never guessed at.

There is one deliberate exception to the agreement, where web/checker.js is
unsound: Lean gives up on deep nesting (maxRecDepth), which no tactic
recovers from, and checker.js does not model that. Here, a formula nested
more than MAX_FORMULA_DEPTH connectives deep is never proved, and `trivial`
never closes a goal whose conjunctions nest more than TRIVIAL_DEPTH deep.
Generated prompts (at most 120 characters) are far from both limits.

The generator keeps random formulas that are intuitionistically provable. A
search for normal natural-deduction proofs (introductions on the goal,
eliminations from the hypotheses) finds the shortest proof term, and the
reference is the shortest checker-verified answer among it and the one-line
tactic forms built from it. Hypotheses are named h0, h1, ... by binding depth.
"""

import random
import re
from functools import lru_cache

from .base import Domain, Problem

MAX_ANSWER, MAX_PROMPT, MAX_NESTING = 400, 300, 64
TOP, BOT = ("top",), ("bot",)
SYMBOL = {"and": "∧", "or": "∨", "imp": "→", "iff": "↔"}
_TIGHTER = {"imp": "or", "or": "and", "and": None}
CONSTANTS = {"trivial": 0, "absurd": 2, "Or.inl": 1, "Or.inr": 1, "Or.elim": 3, "False.elim": 1,
             "And.intro": 2, "And.left": 1, "And.right": 1, "Iff.intro": 2}
FIELDS = {"1", "2", "left", "right", "mp", "mpr"}
# Never binder names: the constants' heads and core Lean 4.34's keywords.
RESERVED = frozenset((
    "And False Iff Not Or Prop Sort StateRefT True Type _ abbrev absurd add_decl_doc assert_not_exists "
    "assert_not_imported at attribute axiom bif binder_predicate break builtin_cbv_simproc "
    "builtin_cbv_simproc_decl builtin_dsimproc builtin_dsimproc_decl builtin_grind_propagator "
    "builtin_initialize builtin_simproc builtin_simproc_decl by by_elab calc catch cbv_eval cbv_simproc "
    "cbv_simproc_decl class coinductive coinductive_fixpoint continue dbg_trace declare_bitwise_int_theorems "
    "declare_bitwise_uint_theorems declare_int_theorems declare_simp_like_tactic declare_syntax_cat "
    "declare_uint_theorems decreasing_by def deprecated_module deprecated_syntax deriving do docs_to_verso "
    "dsimproc dsimproc_decl elab elab_rules else end eval_prec eval_prio example exists export extends "
    "finally for forall from fun generalizing grind_annotated grind_pattern grind_propagator have haveI "
    "hiding idbg if import in include include_str inductive inductive_fixpoint inferInstanceAs infix infixl "
    "infixr init_grind_norm init_quot initialize instance leading_parser let letI let_delayed let_expr "
    "let_fun let_tmp local logNamedError logNamedErrorAt logNamedWarning logNamedWarningAt macro "
    "macro_rules match match_expr matches max_prec meta mod_cast mut mutual namespace nat_lit no_index "
    "nofun nomatch noncomputable nonrec norm_cast_add_elim notation omit opaque open partial "
    "partial_fixpoint postfix prefix private protected public recommended_spelling "
    "register_error_explanation register_sym_dsimp register_sym_simp register_tactic_tag renaming repeat "
    "return run_cmd run_elab run_meta scoped seal section set_library_suggestions set_option show "
    "show_term show_term_elab simproc simproc_decl sorry structure suffices syntax tactic_alt "
    "tactic_extension tactic_name tactic_tag termination_by then theorem throwNamedError "
    "throwNamedErrorAt trailing_parser trivial try unif_hint universe unless unlock_limits unsafe unseal "
    "until using variable where while with with_weak_namespace without_expected_type"
).split())
_UNCOMPILABLE = re.compile(
    r"run_|native|\b(?:import|prelude|open|export|namespace|section|end|variable|universe|omit|include|"
    r"set_option|in|axiom|theorem|lemma|def|abbrev|instance|example|opaque|structure|class|inductive|"
    r"coinductive|mutual|attribute|notation|infix|infixl|infixr|prefix|postfix|macro|macro_rules|syntax|"
    r"elab|elab_rules|declare_syntax_cat|initialize|builtin_initialize|deriving|noncomputable|unsafe|"
    r"partial|private|protected|local|scoped|where|add_decl_doc|register_simp_attr|simproc|dsimproc|"
    r"binder_predicate|grind_pattern|unif_hint|implemented_by|extern|sorry|admit)\b", re.ASCII)
# JavaScript's /^...$/ is Python's fullmatch (Python's `$` also matches before a
# final newline); its sticky /.../y is Python's pattern.match(text, pos).
_BINDER = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
_HEADER = re.compile(r"example(?: \(([a-z][a-z ]*) : Prop\))? : ([^\n]+) :=")
_FORMULA_TOKEN = re.compile(r" *([()¬∧∨→↔]|[A-Za-z_][A-Za-z0-9_']*)")
_IDENT_START = re.compile(r"[A-Za-z_]")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.(?:[A-Za-z_][A-Za-z0-9_']*|[0-9]+))*")
_FIELD = re.compile(r"\.([A-Za-z_][A-Za-z0-9_']*|[0-9]+)")
_TACTIC_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.(?:[A-Za-z_][A-Za-z0-9_']*|[0-9]+))*|:=|=>|[()⟨⟩,;|·]")
_PROJECTIONS = re.compile(r"(?:\.(?:[A-Za-z_][A-Za-z0-9_']*|[0-9]+))+")
# What ends the term of an exact, apply, cases or obtain; a term never continues on a new line.
_TERM_STOP = {";", "|", ":=", "·", "with"}
# Names Lean accepts, but a hypothesis named like a tactic or an rcases keyword only invites misreadings.
_TACTIC_WORDS = {"intro", "exact", "apply", "constructor", "left", "right", "exfalso", "assumption",
                 "trivial", "contradiction", "cases", "rcases", "obtain", "inl", "inr", "rfl"}
MAX_BRANCHES = 64  # goals one rcases pattern may split into
# Lean 4.34 stops at maxRecDepth (512). Elaborating a formula costs about two
# levels per nested connective and one per parenthesis, so a header with a
# chain of 254 ¬ does not compile, whatever the proof; 200 connectives and the
# 64 parentheses parse_formula allows stay well inside the limit.
MAX_FORMULA_DEPTH = 200
# Lean's `trivial` tries `apply And.intro <;> trivial` before its other
# alternatives (assumption included), so it splits every conjunction of the
# goal, about eight levels each. Alone it fails at 64 nested ∧, under 62
# bullets (three levels each) at 40; 32 leaves room for the deepest nesting of
# bullets and alternatives the tactic checker allows.
TRIVIAL_DEPTH = 32
# `apply c` for a constant whose implicit arguments Lean unifies with the goal.
_INTRO_RULES = ("Or.inl", "Or.inr", "And.intro", "Iff.intro", "False.elim")


class _Reject(Exception):
    pass


def _js_len(s: str) -> int:
    """String length as JavaScript counts it (UTF-16 code units), so the limits
    agree with web/checker.js on every input, lone surrogates included."""
    return len(s.encode("utf-16-le", "surrogatepass")) // 2


def _bindable(name: str) -> bool:
    return _BINDER.fullmatch(name) is not None and name not in RESERVED and not _UNCOMPILABLE.search(name)


def _hyp_name(name: str) -> bool:
    return _bindable(name) and name not in _TACTIC_WORDS


# ---------------------------------------------------------------- formulas
# Tuples: ("atom", name), TOP, BOT, ("not", a), (op, a, b) for op in and, or, imp, iff.

def parse_formula(text: str):
    """The formula `text` denotes; raises _Reject (or RecursionError) if none."""
    toks, pos = [], 0
    while True:
        m = _FORMULA_TOKEN.match(text, pos)
        if m is None:
            break
        toks.append(m.group(1))
        pos = m.end()
    if text[pos:].strip(" "):
        raise _Reject
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
        if peek() == "↔":
            raise _Reject
        return ("iff", a, b)

    def chain(op, depth):
        nonlocal i
        tighter = _TIGHTER[op]
        parts = [unary(depth) if tighter is None else chain(tighter, depth)]
        while peek() == SYMBOL[op]:
            i += 1
            parts.append(unary(depth) if tighter is None else chain(tighter, depth))
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
        if tok == "(":
            if depth >= MAX_NESTING:
                raise _Reject
            f = iff(depth + 1)
            if peek() != ")":
                raise _Reject
            i += 1
        elif tok == "True":
            f = TOP
        elif tok == "False":
            f = BOT
        elif _IDENT_START.match(tok):
            f = ("atom", tok)
        else:
            raise _Reject
        for _ in range(nots):
            f = ("not", f)
        return f

    f = iff(0)
    if i != len(toks):
        raise _Reject
    return f


def unfold(f):
    """f with every ¬A written A → False, as Lean's Not unfolds."""
    tag = f[0]
    if tag == "not":
        return ("imp", unfold(f[1]), BOT)
    if tag in SYMBOL:
        return (tag, unfold(f[1]), unfold(f[2]))
    return f


def atoms_in_order(f) -> list:
    """The atoms of f, each once, in the order they are written."""
    out, stack = [], [f]
    while stack:
        g = stack.pop()
        if g[0] == "atom":
            if g[1] not in out:
                out.append(g[1])
        else:
            stack.extend(reversed(g[1:]))
    return out


def formula_depth(f) -> int:
    """The most connectives (¬ included) on one path from the root of f to a leaf."""
    best, stack = 0, [(f, 0)]
    while stack:
        g, d = stack.pop()
        if g[0] in ("atom", "top", "bot"):
            best = max(best, d)
        else:
            stack.extend((h, d + 1) for h in g[1:])
    return best


def _and_height(f) -> int:
    """How deep the conjunctions at the top of f nest: what Lean's `trivial` splits."""
    return 1 + max(_and_height(f[1]), _and_height(f[2])) if f[0] == "and" else 0


_PREC = {"iff": 1, "imp": 2, "or": 3, "and": 4}


def show_formula(f) -> str:
    """f with the fewest parentheses the parser (and Lean) needs."""
    tag = f[0]
    if tag == "atom":
        return f[1]
    if tag == "top":
        return "True"
    if tag == "bot":
        return "False"
    if tag == "not":
        inner = show_formula(f[1])
        return "¬" + (f"({inner})" if f[1][0] in _PREC else inner)
    prec = _PREC[tag]
    left, right = show_formula(f[1]), show_formula(f[2])
    if _PREC.get(f[1][0], 5) <= prec:
        left = f"({left})"
    if _PREC.get(f[2][0], 5) < prec or (tag == "iff" and f[2][0] == "iff"):
        right = f"({right})"
    return f"{left} {SYMBOL[tag]} {right}"


def header(f) -> str:
    """The prompt for f, declaring exactly its atoms, in alphabetical order."""
    atoms = sorted(atoms_in_order(f))
    binders = f" ({' '.join(atoms)} : Prop)" if atoms else ""
    return f"example{binders} : {show_formula(f)} :="


def goal_of(prompt):
    """The formula a prop header states (¬ not unfolded), or None if the
    prompt is not one."""
    if not isinstance(prompt, str) or _js_len(prompt) > MAX_PROMPT:
        return None
    return _goal_of(prompt)


@lru_cache(maxsize=1 << 14)
def _goal_of(prompt: str):
    m = _HEADER.fullmatch(prompt)
    if m is None:
        return None
    names = [n for n in m.group(1).split(" ") if n] if m.group(1) else []
    try:
        goal = parse_formula(m.group(2))
    except (_Reject, RecursionError):
        return None
    if len(set(names)) != len(names) or not set(atoms_in_order(goal)) <= set(names):
        return None
    if any(name in RESERVED for name in names):
        return None
    return goal


# ---------------------------------------------------------------- proof terms
# Terms: ("lam", name, body), ("anon", items), ("const", name, args),
# ("app", head, args), ("var", name), ("proj", term, field).

def _lex_term(text: str):
    """Tokens, and for each whether whitespace precedes it: Lean reads an
    argument only after whitespace (`f(x)` is not an application). A term
    stays on one line."""
    toks, spaced = [], []
    i, gap, n = 0, True, len(text)
    while i < n:
        c = text[i]
        if c == " ":
            i += 1
            gap = True
            continue
        spaced.append(gap)
        gap = False
        if c == "(" or c == "⟨" or c == ",":
            toks.append(("p", c))
            i += 1
        elif c == ")" or c == "⟩":
            toks.append(("p", c))
            i += 1
            while True:
                m = _FIELD.match(text, i)
                if m is None:
                    break
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
            toks.append(("id", m.group()))
            i = m.end()
    return toks, spaced


_END = ("end", "")


def parse_term(text: str, outer=()):
    """The term `text` denotes; `outer` holds the hypotheses already named
    outside it (a tactic proof's context)."""
    toks, spaced = _lex_term(text)
    n = len(toks)
    i = depth = 0
    scope = list(outer)

    def peek():
        return toks[i] if i < n else _END

    def next_tok():
        nonlocal i
        if i >= n:
            raise _Reject
        i += 1
        return toks[i - 1]

    def project(t, field):
        if field not in FIELDS:
            raise _Reject
        return ("proj", t, field)

    def starts_atom():
        kind, word = peek()
        return (kind == "id" and word != "fun") or (kind == "p" and (word == "(" or word == "⟨"))

    def term():
        nonlocal i, depth
        depth += 1
        if depth > MAX_NESTING:
            raise _Reject
        k0, n0 = peek()
        if k0 == "id" and n0 == "fun":
            i += 1
            names = []
            while peek() != ("p", "=>"):
                kind, name = next_tok()
                if kind != "id" or not (name == "_" or _bindable(name)):
                    raise _Reject
                names.append(name)
            if not names:
                raise _Reject
            i += 1
            scope.extend(names)
            t = term()
            del scope[len(scope) - len(names):]
            for name in reversed(names):
                t = ("lam", name, t)
        else:
            if k0 == "id" and CONSTANTS.get(n0, 0) > 0:
                i += 1
                t = ("const", n0, [arg() for _ in range(CONSTANTS[n0])])
            else:
                t = atom()
            args = []
            while starts_atom():
                args.append(arg())
            if args:
                t = ("app", t, args)
        depth -= 1
        return t

    def arg():
        if i < n and not spaced[i]:
            raise _Reject
        return atom()

    def atom():
        nonlocal i
        kind, word = next_tok()
        if kind == "id":
            parts = word.split(".")
            if parts[0] != "_" and parts[0] in scope:
                t = ("var", parts[0])
                fields = parts[1:]
            else:
                width = 1 if parts[0] in CONSTANTS else 2
                name = ".".join(parts[:width])
                if CONSTANTS.get(name) != 0:
                    raise _Reject
                t = ("const", name, [])
                fields = parts[width:]
            for f in fields:
                t = project(t, f)
            return t
        if (kind, word) == ("p", "("):
            t = term()
            if next_tok() != ("p", ")"):
                raise _Reject
        elif (kind, word) == ("p", "⟨"):
            items = [term()]
            while peek() == ("p", ","):
                i += 1
                items.append(term())
            if next_tok() != ("p", "⟩") or len(items) < 2:
                raise _Reject
            t = ("anon", items)
        else:
            raise _Reject
        while peek()[0] == ".":
            t = project(t, next_tok()[1])
        return t

    t = term()
    if i != n:
        raise _Reject
    return t


# Contexts are linked tuples (name, type, outer), innermost binder first.

def _lookup(ctx, name):
    while ctx is not None:
        if ctx[0] == name:
            return ctx[1]
        ctx = ctx[2]
    raise _Reject


def _fields(ty):
    if ty[0] == "and":
        return ty[1], ty[2]
    if ty[0] == "iff":
        return ("imp", ty[1], ty[2]), ("imp", ty[2], ty[1])
    raise _Reject


def _check(ctx, t, ty):
    tag = t[0]
    if tag == "lam":
        if ty[0] != "imp":
            raise _Reject
        _check((t[1], ty[1], ctx), t[2], ty[2])
    elif tag == "anon":
        _check_fields(ctx, t[1], ty)
    elif tag == "const" and t[1] != "And.left" and t[1] != "And.right":
        _check_const(ctx, t[1], t[2], ty)
    elif _infer(ctx, t) != ty:
        raise _Reject


def _check_fields(ctx, items, ty):
    first, rest = _fields(ty)
    _check(ctx, items[0], first)
    if len(items) == 2:
        _check(ctx, items[1], rest)
    else:
        _check_fields(ctx, items[1:], rest)


def _check_const(ctx, name, args, ty):
    if name == "Or.inl" or name == "Or.inr":
        if ty[0] != "or":
            raise _Reject
        _check(ctx, args[0], ty[1] if name == "Or.inl" else ty[2])
    elif name == "Or.elim":
        d = _infer(ctx, args[0])
        if d[0] != "or":
            raise _Reject
        _check(ctx, args[1], ("imp", d[1], ty))
        _check(ctx, args[2], ("imp", d[2], ty))
    elif name == "False.elim":
        _check(ctx, args[0], BOT)
    elif name == "absurd":
        _check(ctx, args[1], ("imp", _infer(ctx, args[0]), BOT))
    elif name == "And.intro" or name == "Iff.intro":
        if ty[0] != ("and" if name == "And.intro" else "iff"):
            raise _Reject
        _check_fields(ctx, args, ty)
    elif ty != TOP:
        raise _Reject


def _infer(ctx, t):
    tag = t[0]
    if tag == "var":
        return _lookup(ctx, t[1])
    if tag == "proj":
        ty, field = _infer(ctx, t[1]), t[2]
        if ty[0] == "and" and field in ("1", "left", "2", "right"):
            return ty[1] if field in ("1", "left") else ty[2]
        if ty[0] == "iff" and field in ("1", "mp", "2", "mpr"):
            return ("imp", ty[1], ty[2]) if field in ("1", "mp") else ("imp", ty[2], ty[1])
        raise _Reject
    if tag == "app":
        ty = _infer(ctx, t[1])
        for a in t[2]:
            if ty[0] != "imp":
                raise _Reject
            _check(ctx, a, ty[1])
            ty = ty[2]
        return ty
    if tag == "const":
        name, args = t[1], t[2]
        if name == "trivial":
            return TOP
        if name == "And.left" or name == "And.right":
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
        if name == "Or.elim":
            d, f = _infer(ctx, args[0]), _infer(ctx, args[1])
            if d[0] != "or" or f[0] != "imp" or f[1] != d[1]:
                raise _Reject
            _check(ctx, args[2], ("imp", d[2], f[2]))
            return f[2]
    raise _Reject


# ---------------------------------------------------------------- tactics

def _lex_tactics(text: str) -> list:
    """Tokens (text, line, column, start, end, first on its line). As in
    _lex_term, projections written directly after `)` or `⟩` stay attached
    to it. Any other character, a tab included, is refused."""
    toks = []
    i = line = col = 0
    first = True
    while i < len(text):
        c = text[i]
        if c == " ":
            i += 1
            col += 1
        elif c == "\n":
            i += 1
            line += 1
            col = 0
            first = True
        else:
            m = _TACTIC_TOKEN.match(text, i)
            if m is None:
                raise _Reject
            end = m.end()
            if m.group() == ")" or m.group() == "⟩":
                p = _PROJECTIONS.match(text, end)
                if p is not None:
                    end = p.end()
            toks.append((text[i:end], line, col, i, end, first))
            col += end - i  # token characters are all in the BMP: code points are code units
            i = end
            first = False
    return toks


def _hyps(ctx) -> list:
    """Every hypothesis type of a context, the inaccessible ones (named None) included."""
    out = []
    while ctx is not None:
        out.append(ctx[1])
        ctx = ctx[2]
    return out


def _clear(ctx, name):
    """ctx after cases or rcases consumes the hypothesis `name` refers to: that
    one is removed, and older ones it shadowed lose their name too."""
    entries = []
    while ctx is not None:
        entries.append(ctx)
        ctx = ctx[2]
    innermost = next((k for k, e in enumerate(entries) if e[0] == name), -1)
    out = None
    for k in range(len(entries) - 1, -1, -1):
        if k != innermost:
            e = entries[k]
            out = (None if e[0] == name else e[0], e[1], out)
    return out


def _contradiction(ctx) -> bool:
    """What Lean's `contradiction` certainly closes: a hypothesis False, or ¬a (written either way) with a."""
    hs = _hyps(ctx)
    return BOT in hs or any(h[0] == "imp" and h[2] == BOT and h[1] in hs for h in hs)


def _trivial(ctx, goal) -> bool:
    """The parts of Lean's `trivial` trusted here: assumption, contradiction,
    True.intro and `apply And.intro <;> trivial`."""
    if goal == TOP or goal in _hyps(ctx) or _contradiction(ctx):
        return True
    return goal[0] == "and" and _trivial(ctx, goal[1]) and _trivial(ctx, goal[2])


def _arrows(f) -> int:
    n = 0
    while f[0] == "imp":
        n += 1
        f = f[2]
    return n


class _Uninstantiated(tuple):
    """The type `cases ... with | intro h1 h2` gives the fields of an iff: Lean
    records it as `?a → ?b`, assigned but not instantiated. It equals the
    formula it stands for; only _apply_goals tells the two apart."""
    __slots__ = ()


def _apply_goals(ty, goal):
    """The goals Lean's `apply e` leaves for `e : ty`, or None if it fails: the
    first i from arrows(ty) - arrows(goal) up to all of them whose conclusion
    after i arguments is the goal, the arguments becoming the new goals. With
    a metavariable as conclusion (`?a → ?b`, b no implication) only all of them."""
    premises, concl = [], [ty]
    while concl[-1][0] == "imp":
        premises.append(concl[-1][1])
        concl.append(concl[-1][2])
    if isinstance(ty, _Uninstantiated) and ty[2][0] != "imp":
        start = len(premises)
    else:
        start = max(0, len(premises) - _arrows(goal))
    for k in range(start, len(premises) + 1):
        if concl[k] == goal:
            return premises[:k]
    return None


def _intro_rule(name, goal):
    # False.elim : {C : Sort u} → False → C. On a goal False → c, Lean's first
    # try instantiates only C, which unifies and closes the goal outright.
    if name == "False.elim":
        return [] if goal[0] == "imp" and goal[1] == BOT else [BOT]
    if (name == "Or.inl" or name == "Or.inr") and goal[0] == "or":
        return [goal[1] if name == "Or.inl" else goal[2]]
    if (name == "And.intro" or name == "Iff.intro") and goal[0] == ("and" if name == "And.intro" else "iff"):
        return list(_fields(goal))
    return None


def _destruct(pat, ty):
    """The hypotheses each goal gets from an rcases pattern, goals in Lean's order:
    ⟨p1, ..., pn⟩ splits ∧ and ↔, p1 | ... | pn splits ∨, surplus patterns nesting into the last."""
    if pat[0] == "name":
        return [[(pat[1], ty)]]
    items = pat[1]
    rest = items[1] if len(items) == 2 else (pat[0], items[1:])
    if pat[0] == "tuple":
        first, second = _fields(ty)
        left, right = _destruct(items[0], first), _destruct(rest, second)
        out = [a + b for a in left for b in right]
    elif ty[0] == "or":
        out = _destruct(items[0], ty[1]) + _destruct(rest, ty[2])
    else:
        raise _Reject
    if len(out) > MAX_BRANCHES:
        raise _Reject
    return out


def _run_tactics(text: str, goal) -> None:
    """Runs a tactic proof on its goal as Lean would, or raises _Reject.

        proof   := "by" seq
        seq     := tactic ((";" | a new line at the sequence's column) tactic)*
        tactic  := "intro" name+ | "exact" term | "apply" term | "cases" term ["with" alt+]
                 | "rcases" term "with" pattern | "obtain" pattern ":=" term | "·" seq
                 | "constructor" | "left" | "right" | "exfalso" | "assumption" | "contradiction" | "trivial"
        alt     := "|" ("inl" | "inr" | "intro") name* "=>" seq
        pattern := item ("|" item)*     item := name | "⟨" pattern ("," pattern)+ "⟩" | "(" pattern ")"

    A sequence continues on a new line only at the column of its first
    tactic, `;` binds the rest of the line into the innermost sequence, and
    alternatives continue while each `|` is at least as far right as the
    first; a line further right than the sequence it would continue is
    rejected rather than guessed at. A goal is (ctx, target); each tactic acts
    on the first goal and puts the goals it creates in front."""
    toks = _lex_tactics(text)
    n = len(toks)
    i = depth = 0

    def peek():
        return toks[i] if i < n else None

    def next_tok():
        nonlocal i
        if i >= n:
            raise _Reject
        i += 1
        return toks[i - 1]

    def inline(word=None):
        """Whether the next token is on the current line (and reads `word`)."""
        return i < n and not toks[i][5] and (word is None or toks[i][0] == word)

    def seq(goals):
        nonlocal i, depth
        depth += 1
        if depth > MAX_NESTING or i >= n:
            raise _Reject
        col = toks[i][2]
        goals = tactic(goals, col)
        while i < n:
            tok = toks[i]
            if tok[0] == ";" and not tok[5]:
                i += 1
                if not inline():
                    raise _Reject
            elif not tok[5] or tok[2] < col:
                break  # the enclosing construct decides what the token is
            elif tok[2] > col:
                raise _Reject
            goals = tactic(goals, col)
        depth -= 1
        return goals

    def names():
        out = []
        while inline() and _BINDER.fullmatch(toks[i][0]):
            name = next_tok()[0]
            if not _hyp_name(name):
                raise _Reject
            out.append(name)
        return out

    def term_text():
        nonlocal i
        start = i
        while inline() and toks[i][0] not in _TERM_STOP:
            i += 1
        if i == start:
            raise _Reject
        return text[toks[start][3]:toks[i - 1][4]]

    def parse(ctx, source):
        scope = []
        while ctx is not None:
            if ctx[0] is not None:
                scope.append(ctx[0])
            ctx = ctx[2]
        return parse_term(source, scope)

    def subject(ctx):
        """The type of the term a cases or rcases destructs, and the context
        without it: Lean clears a destructed hypothesis, but nothing else."""
        t = parse(ctx, term_text())
        ty = _infer(ctx, t)
        return ty, (_clear(ctx, t[1]) if t[0] == "var" else ctx)

    def tactic(goals, col):
        nonlocal i
        word = next_tok()[0]
        if not goals:
            raise _Reject  # Lean: "no goals to be solved"
        ctx, target = goals[0]
        rest = goals[1:]
        if word == "·":
            if not inline() or seq([goals[0]]):
                raise _Reject
            return rest
        if word == "intro":
            introduced = names()
            if not introduced:
                raise _Reject
            for name in introduced:
                if target[0] != "imp":
                    raise _Reject
                ctx = (name, target[1], ctx)
                target = target[2]
            return [(ctx, target)] + rest
        if word == "exact":
            _check(ctx, parse(ctx, term_text()), target)
            return rest
        if word == "apply":
            source = term_text()
            if source in _INTRO_RULES:
                created = _intro_rule(source, target)
            else:
                created = _apply_goals(_infer(ctx, parse(ctx, source)), target)
            if created is None:
                raise _Reject
            return [(ctx, g) for g in created] + rest
        if word == "cases":
            ty, cleared = subject(ctx)
            if not inline("with"):
                if ty != BOT:
                    raise _Reject  # `cases h` without alternatives: only for h : False
                return rest
            i += 1
            alternatives(cleared, ty, target, col)
            return rest
        if word == "rcases" or word == "obtain":
            if word == "rcases":
                ty, ctx = subject(ctx)
                if not inline("with"):
                    raise _Reject
                i += 1
                pat = pattern()
            else:
                pat = pattern()
                if not inline(":="):
                    raise _Reject
                i += 1
                ty, ctx = subject(ctx)
            if pat[0] == "name":
                raise _Reject
            out = []
            for branch in _destruct(pat, ty):
                if len({name for name, _ in branch}) != len(branch):
                    raise _Reject  # one name twice in a goal
                c = ctx
                for name, f in branch:
                    c = (name, f, c)
                out.append((c, target))
            return out + rest
        if word == "constructor":
            if target[0] == "and" or target[0] == "iff":
                return [(ctx, g) for g in _fields(target)] + rest
            if target != TOP:
                raise _Reject
            return rest
        if word == "left" or word == "right":
            if target[0] != "or":
                raise _Reject
            return [(ctx, target[1] if word == "left" else target[2])] + rest
        if word == "exfalso":
            return [(ctx, BOT)] + rest
        if word == "assumption":
            closed = target in _hyps(ctx)
        elif word == "contradiction":
            closed = _contradiction(ctx)
        elif word == "trivial":
            closed = _and_height(target) <= TRIVIAL_DEPTH and _trivial(ctx, target)
        else:
            raise _Reject
        if not closed:
            raise _Reject
        return rest

    def alternatives(ctx, ty, target, col):
        """`| inl h => seq | inr h => seq` for ∨, `| intro h1 h2 => seq` for ∧
        and ↔, each exactly once, in any order, each closing its goal."""
        nonlocal i
        if ty[0] == "or":
            wanted = {"inl": [ty[1]], "inr": [ty[2]]}
        else:
            parts = _fields(ty)
            wanted = {"intro": [_Uninstantiated(f) for f in parts] if ty[0] == "iff" else list(parts)}
        tok = peek()
        if tok is None or tok[0] != "|" or (tok[5] and tok[2] < col):
            raise _Reject
        anchor = tok[2]
        while i < n and toks[i][0] == "|" and toks[i][2] >= anchor:
            tok = toks[i]
            i += 1
            if not inline():
                raise _Reject
            parts = wanted.pop(next_tok()[0], None)  # None: an unknown or repeated alternative
            bound = names()
            if parts is None or len(bound) != len(parts) or not inline("=>"):
                raise _Reject
            i += 1
            body = peek()
            if body is None or (body[5] and body[2] <= tok[2]):
                raise _Reject
            arm = ctx
            for name, f in zip(bound, parts):
                arm = (name, f, arm)
            if seq([(arm, target)]):
                raise _Reject
        if wanted:
            raise _Reject  # Lean: "alternative has not been provided"

    def pattern():
        nonlocal depth
        depth += 1
        if depth > MAX_NESTING:
            raise _Reject
        items = [pattern_item()]
        while inline("|"):
            next_tok()
            items.append(pattern_item())
        depth -= 1
        return items[0] if len(items) == 1 else ("alt", items)

    def pattern_item():
        if not inline():
            raise _Reject
        tok = next_tok()[0]
        if tok == "⟨":
            items = [pattern()]
            while inline(","):
                next_tok()
                items.append(pattern())
            if len(items) < 2 or not inline("⟩"):
                raise _Reject
            next_tok()
            return ("tuple", items)
        if tok == "(":
            pat = pattern()
            if not inline(")"):
                raise _Reject
            next_tok()
            return pat
        if not _hyp_name(tok):
            raise _Reject
        return ("name", tok)

    if next_tok()[0] != "by" or i >= n:
        raise _Reject
    # The first line also holds the prompt, which shifts its columns, so a
    # multi-line proof keeps `by` alone on it and starts the block at column 2.
    if "\n" in text and (not toks[i][5] or toks[i][2] != 2):
        raise _Reject
    if seq([(None, goal)]) or i != n:
        raise _Reject


def check_answer(goal, answer: str) -> bool:
    """Whether `answer` (a term, or tactics after `by`) proves the unfolded goal."""
    if formula_depth(goal) > MAX_FORMULA_DEPTH:
        return False  # too close to where Lean can no longer elaborate the header
    try:
        if answer.startswith("by ") or answer.startswith("by\n"):
            _run_tactics(answer, goal)
        else:
            _check(None, parse_term(answer), goal)
        return True
    except (_Reject, RecursionError):  # RecursionError where JavaScript would throw RangeError
        return False


# ---------------------------------------------------------------- proof search
# Proofs are tuples (tag, length, atomic, ...): the length of the printed term
# (every hypothesis name counted as two characters, h0 to h9), which the
# search minimises, and whether the term needs no parentheses as an argument.
# Tags and fields: hyp (formula), triv, lam (body, formula bound), pair (a, b),
# inl and inr (a), elim (scrutinee, f, g, disjunction: Or.elim), efq (e:
# False.elim), absurd (a, f), app (f, a), proj (e, 1 or 2, "and" or "iff").
# A hypothesis is named by its formula: printing gives it the name of the
# first binder of that formula in scope.

def _arg_len(t) -> int:
    return t[1] if t[2] else t[1] + 2


def _hyp(f):
    return ("hyp", 2, True, f)


_TRIV = ("triv", 7, True)


def _lam(f, body):
    # fun h0 => fun h1 => b prints as fun h0 h1 => b
    return ("lam", body[1] + (3 if body[0] == "lam" else 10), False, body, f)


def _pair(a, b):
    # ⟨a, ⟨b, c⟩⟩ prints as ⟨a, b, c⟩
    return ("pair", a[1] + b[1] + (2 if b[0] == "pair" else 4), True, a, b)


def _inj(tag, a):
    return (tag, 7 + _arg_len(a), False, a)


def _elim(s, f, g, disjunction):
    return ("elim", 10 + _arg_len(s) + _arg_len(f) + _arg_len(g), False, s, f, g, disjunction)


def _efq(e):
    return ("efq", 11 + _arg_len(e), False, e)


def _absurd(a, f):
    return ("absurd", 8 + _arg_len(a) + _arg_len(f), False, a, f)


def _app(f, a):
    return ("app", f[1] + 1 + _arg_len(a), False, f, a)


def _proj(e, k, kind):
    return ("proj", e[1] + (4 if e[0] == "app" else 2), True, e, k, kind)


_CHILDREN = {"hyp": (), "triv": (), "lam": (3,), "pair": (3, 4), "inl": (3,), "inr": (3,), "elim": (3, 4, 5),
             "efq": (3,), "absurd": (3, 4), "app": (3, 4), "proj": (3,)}
_INFERABLE = ("hyp", "proj", "app")
_FIELD_NAMES = {("and", 1): "left", ("and", 2): "right", ("iff", 1): "mp", ("iff", 2): "mpr"}


def _h(level: int) -> str:
    return f"h{level}"


def _bind(levels: dict, f, depth: int) -> dict:
    """levels (formula -> level of its first binder in scope) after binding f at depth."""
    return levels if f in levels else {**levels, f: depth}


def show_proof(t, name=_h, named_fields=False, depth=0, levels=None) -> str:
    """The proof t as Lean source: the binder at depth d named name(d), a
    hypothesis by the name of the first binder of its formula (`levels` maps
    the formulas already in scope to their levels), projections written .1/.2
    or (named_fields) .left/.right/.mp/.mpr."""
    levels = {} if levels is None else levels
    tag = t[0]

    def show(x):
        return show_proof(x, name, named_fields, depth, levels)

    def arg(x):
        return show(x) if x[2] else f"({show(x)})"

    if tag == "hyp":
        return name(levels[t[3]])
    if tag == "triv":
        return "trivial"
    if tag == "lam":
        binders = []
        while t[0] == "lam":
            levels = _bind(levels, t[4], depth)
            binders.append(name(depth))
            depth += 1
            t = t[3]
        return f"fun {' '.join(binders)} => {show_proof(t, name, named_fields, depth, levels)}"
    if tag == "pair":
        items = []
        while t[0] == "pair":
            items.append(show(t[3]))
            t = t[4]
        return "⟨" + ", ".join(items + [show(t)]) + "⟩"
    if tag == "inl" or tag == "inr":
        return f"Or.{tag} {arg(t[3])}"
    if tag == "elim":
        return f"Or.elim {arg(t[3])} {arg(t[4])} {arg(t[5])}"
    if tag == "efq":
        return f"False.elim {arg(t[3])}"
    if tag == "absurd":
        return f"absurd {arg(t[3])} {arg(t[4])}"
    if tag == "app":
        args = []
        while t[0] == "app":
            args.append(arg(t[4]))
            t = t[3]
        return " ".join([show(t)] + args[::-1])
    base = show(t[3])
    if t[3][0] == "app":
        base = f"({base})"
    return f"{base}.{_FIELD_NAMES[t[5], t[4]] if named_fields else t[4]}"


_MISSING = object()
_NEVER = 1 << 30
_EXPENSIVE = 12  # False.elim, absurd and Or.elim proofs are never shorter than this


class _Search:
    """Shortest normal natural-deduction proofs. A state is (hypotheses,
    goal). A proof is an introduction for the goal's connective, or the
    eliminations of a hypothesis (projections, applications) down to the goal
    itself, to False (then False.elim or absurd) or to a disjunction (then
    Or.elim, each case proving its implication). Every hypothesis is a
    subformula of the root goal, so there are finitely many states; a state
    met again on its own branch is cut, since its proof could replace the
    outer one. A result that depended on a cut of a state still open above it
    is not memoised: it is recomputed where that state is not open.

    Eliminating a hypothesis is tried only when its spine reaches the goal,
    False or a disjunction, and the routes through False and disjunctions only
    when no proof of at most _EXPENSIVE characters is known. Past the budget of
    states every new state counts as unprovable: the result is the best proof
    found so far, if any.

    For each provable state the search keeps the best proof for each place it
    can fill, as (top, arg, body, tail, infer): anywhere; as an argument
    (where parentheses count); as a lambda's body (whose binders a lambda
    there merges into its own); as the last item of ⟨⟩ (where a pair
    flattens); and the best proof whose type is inferred rather than checked
    (absurd's first argument)."""

    def __init__(self, budget: int):
        self.budget = budget
        self.memo, self.open, self.spines = {}, {}, {}

    def reaches(self, f):
        """(the formulas along f's elimination spine, whether False is among them, whether a disjunction is)."""
        hit = self.spines.get(f)
        if hit is None:
            heads = {f}
            if f[0] == "and" or f[0] == "iff":
                for g in _fields(f):
                    heads |= self.reaches(g)[0]
            elif f[0] == "imp":
                heads |= self.reaches(f[2])[0]
            hit = self.spines[f] = (heads, BOT in heads, any(h[0] == "or" for h in heads))
        return hit

    def prove(self, ctx: tuple, goal):
        """(best proofs or None, the lowest open level a cut below reached)."""
        key = (frozenset(ctx), goal)
        hit = self.memo.get(key, _MISSING)
        if hit is not _MISSING:
            return hit, _NEVER
        level = self.open.get(key)
        if level is not None:
            return None, level
        if self.budget <= 0:
            return None, -1  # nothing above is final either
        self.budget -= 1
        level = len(self.open)
        self.open[key] = level
        candidates, low = self.candidates(ctx, goal)
        del self.open[key]
        best = self.pick(candidates)
        if low >= level:
            self.memo[key] = best
            low = _NEVER
        return best, low

    @staticmethod
    def pick(candidates):
        if not candidates:
            return None
        top = min(candidates, key=lambda t: t[1])
        arg = min(candidates, key=_arg_len)
        body = min(candidates, key=lambda t: t[1] - 7 * (t[0] == "lam"))
        tail = min(candidates, key=lambda t: t[1] - 2 * (t[0] == "pair"))
        inferable = [t for t in candidates if t[0] in _INFERABLE]
        return top, arg, body, tail, (min(inferable, key=lambda t: t[1]) if inferable else None)

    def candidates(self, ctx: tuple, goal):
        """The best proof by each rule that applies, and the lowest open level reached."""
        if goal in ctx:
            return [_hyp(goal)], _NEVER  # nothing is shorter, in any place
        out, low = [], _NEVER
        tag = goal[0]
        if tag == "imp":
            body, low = self.prove(ctx if goal[1] in ctx else ctx + (goal[1],), goal[2])
            if body:
                out.append(_lam(goal[1], body[2]))
        elif tag == "and" or tag == "iff":
            first, second = _fields(goal)
            a, low = self.prove(ctx, first)
            if a:
                b, low2 = self.prove(ctx, second)
                low = min(low, low2)
                if b:
                    out.append(_pair(a[0], b[3]))
        elif tag == "or":
            for side, inj in ((goal[1], "inl"), (goal[2], "inr")):
                a, low2 = self.prove(ctx, side)
                low = min(low, low2)
                if a:
                    out.append(_inj(inj, a[1]))
        elif tag == "top":
            out.append(_TRIV)
        for hyp in ctx:
            low = min(low, self.walk(ctx, goal, _hyp(hyp), hyp, out, True))
        if not out or min(t[1] for t in out) > _EXPENSIVE:
            for hyp in ctx:
                low = min(low, self.walk(ctx, goal, _hyp(hyp), hyp, out, False))
        return out, low

    def walk(self, ctx: tuple, goal, e, ty, out: list, direct: bool):
        """Adds to `out` the proofs of goal that eliminate e : ty, down to the
        goal itself (direct) or (not direct) to False or a disjunction;
        returns the lowest open level reached."""
        if ty == goal:
            if direct:
                out.append(e)  # eliminating further only makes it longer
            return _NEVER
        heads, to_bot, to_or = self.reaches(ty)
        if not (goal in heads if direct else to_bot or (to_or and goal[0] != "imp")):
            return _NEVER
        low = _NEVER
        tag = ty[0]
        if tag == "bot":
            out.append(_efq(e))
        elif tag == "or":
            # Never for an implication (to_or above): introducing first loses
            # nothing, and Or.elim would ask for ever longer implications.
            f, low = self.prove(ctx, ("imp", ty[1], goal))
            if f:
                g, low2 = self.prove(ctx, ("imp", ty[2], goal))
                low = min(low, low2)
                if g:
                    out.append(_elim(e, f[1], g[1], ty))
        elif tag == "and" or tag == "iff":
            first, second = _fields(ty)
            low = min(self.walk(ctx, goal, _proj(e, 1, tag), first, out, direct),
                      self.walk(ctx, goal, _proj(e, 2, tag), second, out, direct))
        elif tag == "imp":
            a, low = self.prove(ctx, ty[1])
            if a:
                low = min(low, self.walk(ctx, goal, _app(e, a[1]), ty[2], out, direct))
                if not direct and ty[2] == BOT and goal != BOT and a[4]:
                    out.append(_absurd(a[4], e))
        return low


def solve(goal, budget: int = 1500):
    """(the shortest proof term found for an unfolded goal, or None; the search)."""
    search = _Search(budget)
    best, _ = search.prove((), goal)
    return (best[0] if best else None), search


def _lambdas(t) -> list:
    """The formulas the leading lambdas of t bind."""
    out = []
    while t[0] == "lam":
        out.append(t[4])
        t = t[3]
    return out


def _strip(t, k: int):
    for _ in range(k):
        t = t[3]
    return t


def _premises(goal, k: int):
    """The first k premises of goal, and what is left to prove."""
    hyps = []
    for _ in range(k):
        hyps.append(goal[1])
        goal = goal[2]
    return hyps, goal


def _levels(hyps) -> dict:
    levels = {}
    for level, f in enumerate(hyps):
        levels = _bind(levels, f, level)
    return levels


def tactic_answers(goal, best, name=_h, block=False) -> list:
    """Short tactic forms of a proof: `intro` some leading premises, then close
    the goal with trivial, contradiction, cases on a proof of False, or exact;
    on one line (`by intro h0; trivial`) or as a block (`by` alone on its
    line, then a line per tactic at column 2). Not all of them check."""
    out = []
    lams = len(_lambdas(best))
    for k in range(_arrows(goal) + 1):
        hyps, target = _premises(goal, k)
        ctx = None
        for level, h in enumerate(hyps):
            ctx = (name(level), h, ctx)
        levels = _levels(hyps)
        closers = []
        if _trivial(ctx, target):
            closers.append("trivial")
        if _contradiction(ctx):
            closers.append("contradiction")
            if BOT in hyps:
                closers.append(f"cases {name(hyps.index(BOT))}")
        if k <= lams:
            body = _strip(best, k)
            if body[0] == "efq":
                closers.append(f"cases {show_proof(body[3], name, False, k, levels)}")
            elif body[0] == "absurd":
                closers.append(f"cases {show_proof(_app(body[4], body[3]), name, False, k, levels)}")
            closers.append(f"exact {show_proof(body, name, False, k, levels)}")
        intro = " ".join(name(j) for j in range(k))
        for closer in closers:
            if block:
                out.append(f"by\n  intro {intro}\n  {closer}" if k else f"by\n  {closer}")
            else:
                out.append(f"by intro {intro}; {closer}" if k else f"by {closer}")
    return out


# Tactic scripts: a list of steps, each a tactic (a string), ("·", steps) for a
# focused goal, or ("alts", "cases e with", [("| inl h =>", steps), ...]).

def _uses(t, f):
    """(whether hypothesis f occurs in t under a projection, whether it occurs bare)."""
    if t[0] == "proj" and t[3][0] == "hyp" and t[3][3] == f:
        return True, False
    if t[0] == "hyp":
        return False, t[3] == f
    projected = bare = False
    for k in _CHILDREN[t[0]]:
        p, b = _uses(t[k], f)
        projected, bare = projected or p, bare or b
    return projected, bare


def _split(t, f, parts):
    """t with each projection of hypothesis f replaced by the hypothesis of that part."""
    if t[0] == "proj" and t[3][0] == "hyp" and t[3][3] == f:
        return _hyp(parts[t[4] - 1])
    kids = _CHILDREN[t[0]]
    return tuple(_split(x, f, parts) if k in kids else x for k, x in enumerate(t)) if kids else t


def _steps(t, depth, levels, name, rng, block) -> list:
    """Tactic steps that do what the proof term t does, constructs picked by rng."""
    def show(x):
        return show_proof(x, name, False, depth, levels)

    def arg(x):
        return show(x) if x[2] else f"({show(x)})"

    def sub(x, d=depth, lv=levels):
        return _steps(x, d, lv, name, rng, block)

    tag = t[0]
    if tag == "lam":
        binders = []
        while t[0] == "lam":
            levels = _bind(levels, t[4], depth)
            binders.append(name(depth))
            depth += 1
            t = t[3]
        return [f"intro {' '.join(binders)}"] + _destructure(t, depth, levels, name, rng, block)
    if tag == "pair":
        return ["constructor", ("·", sub(t[3])), ("·", sub(t[4]))]
    if tag == "inl" or tag == "inr":
        return ["left" if tag == "inl" else "right"] + sub(t[3])
    if tag == "triv":
        return [rng.choice(("trivial", "constructor", "exact trivial"))]
    if tag == "efq":
        return rng.choice(([f"cases {show(t[3])}"], ["exfalso", f"exact {show(t[3])}"],
                           [f"exact False.elim {arg(t[3])}"]))
    if tag == "absurd":
        options = [[f"exact absurd {arg(t[3])} {arg(t[4])}"], ["exfalso", f"exact {show(_app(t[4], t[3]))}"],
                   [f"cases {show(_app(t[4], t[3]))}"]]
        if t[3][0] == "hyp" and t[4][0] == "hyp":
            options.append(["contradiction"])
        return rng.choice(options)
    if tag == "elim":
        s, disjunction = show(t[3]), t[6]
        n = name(depth)
        arms = []
        for case, f, part in ((1, t[4], disjunction[1]), (2, t[5], disjunction[2])):
            inner = _bind(levels, part, depth)
            body = f[3] if f[0] == "lam" and f[4] == part else _app(f, _hyp(part))
            arms.append(sub(body, depth + 1, inner))
        style = rng.randrange(3 if block else 2)
        if style == 2:
            return [("alts", f"cases {s} with", [(f"| inl {n} =>", arms[0]), (f"| inr {n} =>", arms[1])])]
        split = f"rcases {s} with {n} | {n}" if style == 0 else f"obtain {n} | {n} := {s}"
        return [split, ("·", arms[0]), ("·", arms[1])]
    head, args = t, []
    while head[0] == "app":
        args.append(head[4])
        head = head[3]
    if args and rng.random() < 0.5:
        return [f"apply {show(head)}"] + [("·", sub(a)) for a in reversed(args)]
    return [f"exact {show(t)}"]


def _destructure(t, depth, levels, name, rng, block) -> list:
    """Steps for t after intro: first, perhaps, take apart a conjunction or
    iff hypothesis that t only ever projects (obtain, rcases, or cases)."""
    for f, level in sorted(levels.items(), key=lambda item: item[1]):
        if f[0] not in ("and", "iff") or _uses(t, f) != (True, False) or rng.random() < 0.5:
            continue
        parts = _fields(f)
        inner = _bind(_bind(levels, parts[0], depth), parts[1], depth + 1)
        rest = _steps(_split(t, f, parts), depth + 2, inner, name, rng, block)
        a, b, h = name(depth), name(depth + 1), name(level)
        style = rng.randrange(3 if block else 2)
        if style == 2:
            return [("alts", f"cases {h} with", [(f"| intro {a} {b} =>", rest)])]
        return [f"obtain ⟨{a}, {b}⟩ := {h}" if style == 0 else f"rcases {h} with ⟨{a}, {b}⟩"] + rest
    return _steps(t, depth, levels, name, rng, block)


def _flat(steps) -> list:
    out = []
    for s in steps:
        if isinstance(s, str):
            out.append(s)
        elif s[0] == "·":
            out += _flat(s[1])
        else:
            raise ValueError("alternatives do not go on one line")
    return out


def _lines(steps, col: int) -> list:
    out = []
    for s in steps:
        if isinstance(s, str):
            out.append(" " * col + s)
        elif s[0] == "·":
            inner = _lines(s[1], col + 2)
            out.append(" " * col + "· " + inner[0][col + 2:])
            out += inner[1:]
        else:
            out.append(" " * col + s[1])
            for label, arm in s[2]:
                inner = _lines(arm, col + 2)
                if len(inner) == 1:
                    out.append(" " * col + label + " " + inner[0].lstrip())
                else:
                    out += [" " * col + label] + inner
    return out


def tactic_script(proof, rng: random.Random, name=_h, block=True) -> str:
    """A tactic proof doing what the proof term does: intro, constructor,
    left/right, cases/rcases/obtain, apply, exact, exfalso..., as a block or
    on one line. It need not check (a case split clears the hypothesis split,
    which a later step may still use): the caller verifies."""
    steps = _steps(proof, 0, {}, name, rng, block)
    if block:
        return "by\n" + "\n".join(_lines(steps, 2))
    return "by " + "; ".join(_flat(steps))


# ---------------------------------------------------------------- the domain

_ATOMS = "pqrs"
_ATOM_COUNTS, _ATOM_WEIGHTS = (1, 2, 3, 4), (34, 34, 23, 9)
_SIZES, _SIZE_WEIGHTS = (1, 2, 3, 4, 5, 6, 7), (10, 18, 22, 20, 14, 9, 7)  # binary connectives
_OPS, _OP_WEIGHTS = ("imp", "and", "or", "iff"), (40, 25, 20, 15)
MAX_GENERATED_PROMPT, MAX_REFERENCE, BLOCK_SIZE = 120, 200, 256

# Hypothesis names for document variants: a level's name under each scheme.
_LETTERS = "abcdefgijklmnotuvwxyz"
_SCHEMES = (
    _h,
    lambda level: f"h{level + 1}",
    lambda level: "h" + "'" * level,
    lambda level: _LETTERS[level] if level < len(_LETTERS) else f"x{level}",
    lambda level: "h" + _LETTERS[level] if level < len(_LETTERS) else f"h{level}",
    lambda level: "xyzuvw"[level] if level < 6 else f"x{level}",
)


def _random_formula(rng: random.Random, atoms, n_ops: int):
    if n_ops == 0:
        r = rng.random()
        # True and False only now and then: about one prompt in five has either.
        f = TOP if r < 0.015 else BOT if r < 0.045 else ("atom", rng.choice(atoms))
    else:
        k = rng.randint(0, n_ops - 1)
        op = rng.choices(_OPS, weights=_OP_WEIGHTS)[0]
        f = (op, _random_formula(rng, atoms, k), _random_formula(rng, atoms, n_ops - 1 - k))
    return ("not", f) if rng.random() < 0.12 else f


def _tautology(f) -> bool:
    """Classical validity, evaluating f on all assignments at once (one bit
    each): intuitionistic provability implies it, and it is cheap."""
    atoms = atoms_in_order(f)
    rows = 1 << len(atoms)
    full = (1 << rows) - 1
    masks = {a: sum(1 << row for row in range(rows) if row >> j & 1) for j, a in enumerate(atoms)}

    def value(g):
        tag = g[0]
        if tag == "atom":
            return masks[g[1]]
        if tag == "top":
            return full
        if tag == "bot":
            return 0
        if tag == "not":
            return full & ~value(g[1])
        a, b = value(g[1]), value(g[2])
        if tag == "and":
            return a & b
        if tag == "or":
            return a | b
        if tag == "imp":
            return full & (~a | b)
        return full & ~(a ^ b)

    return value(f) == full


def _canonical_names() -> list:
    letters = "abcdefghijklmnopqrstuvwxyz"
    names = list(letters[15:] + letters[:15])  # p, q, ..., z, a, ..., o
    names += [a + b for a in letters for b in letters if a + b not in RESERVED and not _UNCOMPILABLE.search(a + b)]
    return names


_CANONICAL = _canonical_names()


def _rename(f, mapping):
    tag = f[0]
    if tag == "atom":
        return ("atom", mapping[f[1]])
    if tag == "not":
        return ("not", _rename(f[1], mapping))
    if tag in SYMBOL:
        return (tag, _rename(f[1], mapping), _rename(f[2], mapping))
    return f


def _fold(f):
    """f with every A → False written ¬A: ¬A unfolds to A → False, so the two
    spellings state the same problem, and one representative of each is kept."""
    tag = f[0]
    if tag == "not" or (tag == "imp" and f[2] == BOT):
        return ("not", _fold(f[1]))
    if tag in SYMBOL:
        return (tag, _fold(f[1]), _fold(f[2]))
    return f


class Prop(Domain):
    name = "prop"
    prefer_shorter = True

    def claims(self, prompt: str) -> bool:
        return goal_of(prompt) is not None

    def verify(self, prompt: str, answer: str) -> bool:
        if not isinstance(answer, str) or _js_len(answer) > MAX_ANSWER:
            return False
        goal = goal_of(prompt)
        return goal is not None and check_answer(unfold(goal), answer)

    def sample(self, rng: random.Random) -> Problem:
        """A random provable formula with exactly k distinct atoms (k drawn
        first, so that provability, rarer with more atoms, does not skew it),
        and its shortest checked answer."""
        k = rng.choices(_ATOM_COUNTS, weights=_ATOM_WEIGHTS)[0]
        sizes = [(n, w) for n, w in zip(_SIZES, _SIZE_WEIGHTS) if n >= k]  # a tautology repeats an atom
        while True:
            atoms = rng.sample(_ATOMS, k)
            f = _random_formula(rng, atoms, rng.choices([n for n, _ in sizes], weights=[w for _, w in sizes])[0])
            if len(atoms_in_order(f)) != k or not _tautology(f):
                continue
            prompt = header(f)
            if len(prompt) > MAX_GENERATED_PROMPT:
                continue
            goal = unfold(f)
            best, _ = solve(goal)
            if best is None:
                continue
            answers = sorted(dict.fromkeys([show_proof(best)] + tactic_answers(goal, best)), key=len)
            for answer in answers:  # the shortest that checks; ties go to the term
                if len(answer) > MAX_REFERENCE or len(prompt) + len(answer) + 3 > BLOCK_SIZE:
                    break
                if check_answer(goal, answer):
                    return Problem(self.name, prompt, answer)

    def _variant_pool(self, prompt: str):
        """(term answers, tactic answers) for a prompt, not yet checked: the
        best proof and the next best ones after its leading introductions,
        under several naming schemes, with numbered or named projections;
        short tactic forms; tactic scripts on one line and as blocks."""
        f = goal_of(prompt)
        if f is None:
            return [], []
        goal = unfold(f)
        best, search = solve(goal)
        if best is None:
            return [], []
        binders = _lambdas(best)
        search.budget = 1500
        others = search.candidates(tuple(dict.fromkeys(binders)), _premises(goal, len(binders))[1])[0]
        proofs = [best]
        for t in sorted(others, key=lambda t: t[1])[:3]:
            for f in reversed(binders):
                t = _lam(f, t)
            proofs.append(t)
        terms, tactics = [], []
        for scheme in _SCHEMES:
            for t in proofs:
                terms += [show_proof(t, scheme), show_proof(t, scheme, named_fields=True)]
        for i, scheme in enumerate(_SCHEMES[:4]):
            tactics += tactic_answers(goal, best, scheme) + tactic_answers(goal, best, scheme, block=True)
            styles = random.Random(f"{prompt}/{i}")
            for t in proofs[:2]:
                tactics += [tactic_script(t, styles, scheme, block=True), tactic_script(t, styles, scheme, block=False)]
        return list(dict.fromkeys(terms)), list(dict.fromkeys(tactics))

    def variants(self, prompt: str) -> list:
        """Every answer of _variant_pool that checks."""
        goal = goal_of(prompt)
        if goal is None:
            return []
        goal = unfold(goal)
        terms, tactics = self._variant_pool(prompt)
        return [a for a in terms + tactics if _js_len(a) <= MAX_ANSWER and check_answer(goal, a)]

    def document(self, problem: Problem, rng: random.Random) -> str:
        """The prompt and a valid proof that fits the context: the reference
        40% of the time, else a checked variant (a term 30%, tactics 30%)."""
        r = rng.random()
        if r >= 0.4:
            terms, tactics = self._variant_pool(problem.prompt)
            pool = terms if r < 0.7 else tactics
            goal = unfold(goal_of(problem.prompt))
            for answer in rng.sample(pool, min(len(pool), 4)):
                if len(problem.prompt) + len(answer) + 3 <= BLOCK_SIZE and check_answer(goal, answer):
                    return f"{problem.prompt} {answer}"
        return f"{problem.prompt} {problem.reference}"

    def split_key(self, prompt: str) -> str:
        """Atoms renamed p, q, r, ... in order of first occurrence, A → False
        written ¬A, the formula printed with the fewest parentheses, under a
        header declaring exactly its atoms."""
        f = goal_of(prompt)
        if f is None:
            return prompt
        atoms = atoms_in_order(f)
        mapping = {a: _CANONICAL[k] for k, a in enumerate(atoms)}
        binders = f" ({' '.join(mapping[a] for a in atoms)} : Prop)" if atoms else ""
        return f"example{binders} : {show_formula(_rename(_fold(f), mapping))} :="


DOMAIN = Prop()
