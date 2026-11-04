"""Closed natural-number arithmetic, as Lean 4 evaluates it.

    example : (7 + 5) * 3 - 4 =        the model writes:   32 := by decide

Lean's `Nat` semantics, not school arithmetic: subtraction truncates at zero
(3 - 7 = 0), division floors, n / 0 = 0 and n % 0 = n. Any of `by decide`,
`rfl`, `by rfl` and `by simp` closes a true equation; the checker accepts
exactly `<value> := <one of those>`.
"""

import random

from .base import Domain, Problem

PROOFS = ("by decide", "rfl", "by rfl", "by simp")
MAX_ANSWER, MAX_VALUE, MAX_PROMPT, MAX_NESTING = 400, 10_000, 200, 32
HEAD, TAIL = "example : ", " ="
_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2}


class _Reject(Exception):
    pass


def _js_len(s: str) -> int:
    """String length as JavaScript counts it (UTF-16 code units), so the limits
    agree with web/checker.js on every input. A lone surrogate, which a
    JavaScript string can also hold, counts one (and must not raise)."""
    return len(s.encode("utf-16-le", "surrogatepass")) // 2


def apply(op: str, a: int, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return max(a - b, 0)
    if op == "*":
        return a * b
    if op == "/":
        return a // b if b else 0
    return a % b if b else a


def _lex(text: str) -> list:
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch == " ":
            i += 1
        elif "0" <= ch <= "9":
            j = i
            while j < len(text) and "0" <= text[j] <= "9":
                j += 1
            if j - i > len(str(MAX_VALUE)) or (ch == "0" and j - i > 1):
                raise _Reject
            out.append(int(text[i:j]))
            i = j
        elif ch in "+-*/%()":
            out.append(ch)
            i += 1
        else:
            raise _Reject
    return out


def _parse(prompt: str):
    """(value, tree) for a nat header, or None if it is not one (malformed, or
    a value or intermediate result above MAX_VALUE). Trees are ints or
    (op, left, right). Mirrors valueOf in web/checker.js."""
    if not isinstance(prompt, str) or _js_len(prompt) > MAX_PROMPT or not prompt.startswith(HEAD) \
            or not prompt.endswith(TAIL) or len(prompt) < len(HEAD) + len(TAIL):
        return None
    try:
        toks = _lex(prompt[len(HEAD):len(prompt) - len(TAIL)])
    except _Reject:
        return None
    pos = 0
    too_big = False

    def peek():
        return toks[pos] if pos < len(toks) else None

    def bound(v):
        nonlocal too_big
        if v > MAX_VALUE:
            too_big = True
        return v

    def expr(depth):
        nonlocal pos
        v, t = term(depth)
        while peek() in ("+", "-"):
            op = toks[pos]
            pos += 1
            v2, t2 = term(depth)
            v, t = bound(apply(op, v, v2)), (op, t, t2)
        return v, t

    def term(depth):
        nonlocal pos
        v, t = atom(depth)
        while peek() in ("*", "/", "%"):
            op = toks[pos]
            pos += 1
            v2, t2 = atom(depth)
            v, t = bound(apply(op, v, v2)), (op, t, t2)
        return v, t

    def atom(depth):
        nonlocal pos
        tok = peek()
        pos += 1
        if isinstance(tok, int):
            return bound(tok), tok
        if tok == "(" and depth < MAX_NESTING:
            v = expr(depth + 1)
            if peek() == ")":
                pos += 1
                return v
        raise _Reject

    try:
        value, tree = expr(0)
    except _Reject:
        return None
    if pos != len(toks) or too_big:
        return None
    return value, tree


def value_of(prompt: str):
    parsed = _parse(prompt)
    return None if parsed is None else parsed[0]


def show(tree) -> str:
    """Prints a tree with the fewest parentheses Lean needs (operators are
    left-associative; * / % bind tighter than + -)."""
    if isinstance(tree, int):
        return str(tree)
    op, a, b = tree
    left, right = show(a), show(b)
    if not isinstance(a, int) and _PREC[a[0]] < _PREC[op]:
        left = f"({left})"
    if not isinstance(b, int) and _PREC[b[0]] <= _PREC[op]:
        right = f"({right})"
    return f"{left} {op} {right}"


def canonical(tree):
    """`tree` with every sum and product flattened and its operands sorted, so
    that spellings of one problem that differ only by commuting or
    reassociating + and * (`3 + 4` and `4 + 3`, `(1 + 2) + 3` and
    `1 + (2 + 3)`) get one split key; Lean gives them one value. Subtraction,
    division and modulo are left as they are."""
    if isinstance(tree, int):
        return tree
    op = tree[0]
    if op not in ("+", "*"):
        return (op, canonical(tree[1]), canonical(tree[2]))
    operands = []

    def collect(t):
        if isinstance(t, tuple) and t[0] == op:
            collect(t[1])
            collect(t[2])
        else:
            operands.append(canonical(t))

    collect(tree)
    operands.sort(key=show)
    out = operands[0]
    for t in operands[1:]:
        out = (op, out, t)
    return out


def _first(tree) -> int:
    """The first number of the printed expression."""
    while not isinstance(tree, int):
        tree = tree[1]
    return tree


def _evaluate(tree):
    """Value of a tree, or None if any intermediate result exceeds MAX_VALUE."""
    if isinstance(tree, int):
        return tree if tree <= MAX_VALUE else None
    op, a, b = tree
    va, vb = _evaluate(a), _evaluate(b)
    if va is None or vb is None:
        return None
    v = apply(op, va, vb)
    return v if v <= MAX_VALUE else None


class Nat(Domain):
    name = "nat"
    prefer_shorter = False

    def claims(self, prompt: str) -> bool:
        return value_of(prompt) is not None

    def verify(self, prompt: str, answer: str) -> bool:
        if not isinstance(answer, str) or _js_len(answer) > MAX_ANSWER:
            return False
        value = value_of(prompt)
        return value is not None and any(answer == f"{value} := {proof}" for proof in PROOFS)

    def _tree(self, rng: random.Random, n_ops: int):
        if n_ops == 0:
            # Mostly small numbers, sometimes 0 or 1 (the edge cases of - / %).
            r = rng.random()
            if r < 0.1:
                return rng.choice((0, 1))
            return rng.randint(0, 20)
        k = rng.randint(0, n_ops - 1)
        op = rng.choices(("+", "-", "*", "/", "%"), weights=(30, 25, 25, 10, 10))[0]
        return (op, self._tree(rng, k), self._tree(rng, n_ops - 1 - k))

    def sample(self, rng: random.Random) -> Problem:
        while True:
            n_ops = rng.choices((1, 2, 3), weights=(40, 35, 25))[0]
            tree = self._tree(rng, n_ops)
            value = _evaluate(tree)
            if value is None:
                continue
            # Truncated subtraction makes 0 the value of ~30% of random
            # expressions; keep about one in seven of those, so that 0 is no
            # more common than 1 (about 6% each) and always answering 0 scores
            # well under 10%. (Keeping a quarter left 0 at 9.8%, too close.)
            if value == 0 and rng.random() < 0.85:
                continue
            # Likewise the first number of the expression was the value of 17%
            # of problems (`5 % 7`, `5 + 0`, `5 * 1`, `5 - 0`, `5 % 0`), so
            # copying it would score 17%; keep two in five of those, which
            # leaves every positional guess (first, last, largest, smallest
            # number) at about 8%.
            if value == _first(tree) and rng.random() < 0.6:
                continue
            prompt = f"{HEAD}{show(tree)}{TAIL}"
            return Problem(self.name, prompt, f"{value} := by decide")

    def document(self, problem: Problem, rng: random.Random) -> str:
        value = value_of(problem.prompt)
        return f"{problem.prompt} {value} := {rng.choice(PROOFS)}"

    def split_key(self, prompt: str) -> str:
        """The canonical expression: the generator writes both `3 + 4` and
        `4 + 3`, and with the operands in the order they were written 22% of
        held-out problems had a commuted twin in the training split."""
        parsed = _parse(prompt)
        return prompt if parsed is None else f"{HEAD}{show(canonical(parsed[1]))}{TAIL}"


DOMAIN = Nat()
