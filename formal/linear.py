"""One linear equation in one integer unknown, closed by `omega`.

    example (x : Int) (h : 3 * x + 5 = 20) : x =        the model writes:   5 := by omega

The checker parses both sides with Lean's grammar for `+ - *` and prefix
minus over Int, reduces each to k * x + m, and accepts `<root> := by omega`
(or `by revert h; omega`) when the equation has exactly one integer root.
"""

import random

from .base import Domain, Problem

PROOFS = ("by omega", "by revert h; omega")
MAX_ANSWER, MAX_PROMPT, MAX_NESTING, MAX_MAGNITUDE = 400, 200, 32, 10 ** 6
HEAD, TAIL = "example (x : Int) (h : ", ") : x ="
_ALLOWED = set("0123456789x+*() -")


class _Reject(Exception):
    pass


def _js_len(s: str) -> int:
    """String length as JavaScript counts it (UTF-16 code units); a lone
    surrogate, which a JavaScript string can also hold, counts one."""
    return len(s.encode("utf-16-le", "surrogatepass")) // 2


def _lex(text: str) -> list:
    if "--" in text:  # a Lean comment, not `- -`
        raise _Reject
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch == " ":
            i += 1
        elif "0" <= ch <= "9":
            j = i
            while j < len(text) and "0" <= text[j] <= "9":
                j += 1
            if j - i > len(str(MAX_MAGNITUDE)) or (ch == "0" and j - i > 1):
                raise _Reject
            out.append(int(text[i:j]))
            i = j
        elif ch in "x+-*()":
            out.append(ch)
            i += 1
        else:
            raise _Reject
    return out


def _bounded(k, m, has_x, tree):
    if abs(k) > MAX_MAGNITUDE or abs(m) > MAX_MAGNITUDE:
        raise _Reject
    return k, m, has_x, tree


def _parse_side(text: str):
    """(k, m, has_x, tree) for one side, k * x + m, mirroring parseLinear in
    web/checker.js. Trees: int, "x", ("neg", t), (op, a, b) for op in + - *."""
    toks = _lex(text)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def expr(depth):
        nonlocal pos
        k, m, hx, t = term(depth)
        while peek() in ("+", "-"):
            op = toks[pos]
            pos += 1
            sign = 1 if op == "+" else -1
            k2, m2, x2, t2 = term(depth)
            k, m, hx, t = _bounded(k + sign * k2, m + sign * m2, hx or x2, (op, t, t2))
        return k, m, hx, t

    def term(depth):
        nonlocal pos
        k, m, hx, t = neg(depth)
        while peek() == "*":
            pos += 1
            k2, m2, x2, t2 = neg(depth)
            if hx and x2:
                raise _Reject
            k, m, hx, t = _bounded(k * m2 + k2 * m, m * m2, hx or x2, ("*", t, t2))
        return k, m, hx, t

    def neg(depth):
        nonlocal pos
        if peek() != "-":
            return atom(depth)
        if depth >= MAX_NESTING:
            raise _Reject
        pos += 1
        k, m, hx, t = neg(depth + 1)
        return -k, -m, hx, ("neg", t)

    def atom(depth):
        nonlocal pos
        tok = peek()
        pos += 1
        if isinstance(tok, int):
            return _bounded(0, tok, False, tok)
        if tok == "x":
            return 1, 0, True, "x"
        if tok == "(" and depth < MAX_NESTING:
            v = expr(depth + 1)
            if peek() == ")":
                pos += 1
                return v
        raise _Reject

    v = expr(0)
    if pos != len(toks):
        raise _Reject
    return v


def _parse(prompt: str):
    """(root, left tree, right tree) or None if the prompt is not a linear
    header with exactly one integer root."""
    if not isinstance(prompt, str) or _js_len(prompt) > MAX_PROMPT:
        return None
    if not (prompt.startswith(HEAD) and prompt.endswith(TAIL)) or len(prompt) <= len(HEAD) + len(TAIL):
        return None
    body = prompt[len(HEAD):len(prompt) - len(TAIL)]
    # The JS header regex: one or more allowed characters, "=", one or more.
    if "\n" in body or not all(c in _ALLOWED or c == "=" for c in body):
        return None
    sides = body.split("=")
    if len(sides) != 2 or not sides[0] or not sides[1]:
        return None
    try:
        k1, m1, _, t1 = _parse_side(sides[0])
        k2, m2, _, t2 = _parse_side(sides[1])
    except (_Reject, RecursionError):
        return None
    if k1 == k2 or (m2 - m1) % (k1 - k2) != 0:
        return None
    return (m2 - m1) // (k1 - k2), t1, t2


def solution_of(prompt: str):
    parsed = _parse(prompt)
    return None if parsed is None else parsed[0]


_PREC = {"+": 1, "-": 1, "*": 2}


def show(tree) -> str:
    """Prints a side with the fewest parentheses Lean needs: + - left
    associative, * tighter, prefix minus tighter than *. `- -` keeps its space
    (`--` would start a comment)."""
    if isinstance(tree, int):
        return str(tree)
    if tree == "x":
        return "x"
    if tree[0] == "neg":
        inner = tree[1]
        s = show(inner)
        if isinstance(inner, tuple) and inner[0] != "neg":
            s = f"({s})"
        return ("- " if s.startswith("-") else "-") + s
    op, a, b = tree
    left, right = show(a), show(b)
    if isinstance(a, tuple) and a[0] != "neg" and _PREC[a[0]] < _PREC[op]:
        left = f"({left})"
    if isinstance(b, tuple) and b[0] != "neg" and _PREC[b[0]] <= _PREC[op]:
        right = f"({right})"
    return f"{left} {op} {right}"


def canonical(tree):
    """`tree` with every sum and product flattened and its operands sorted, so
    that commuted or reassociated spellings of a side (`x * 3 + 5`,
    `5 + 3 * x`) get one split key. Subtraction and prefix minus are left as
    they are."""
    if isinstance(tree, int) or tree == "x":
        return tree
    if tree[0] == "neg":
        return ("neg", canonical(tree[1]))
    op = tree[0]
    if op == "-":
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


class Linear(Domain):
    name = "linear"
    prefer_shorter = False

    def claims(self, prompt: str) -> bool:
        return solution_of(prompt) is not None

    def verify(self, prompt: str, answer: str) -> bool:
        if not isinstance(answer, str) or _js_len(answer) > MAX_ANSWER:
            return False
        root = solution_of(prompt)
        return root is not None and any(answer == f"{root} := {proof}" for proof in PROOFS)

    @staticmethod
    def _const(c: int):
        return c if c >= 0 else ("neg", -c)

    def _side(self, rng, a: int, b: int, form: str):
        """A tree for a*x + b in one of several surface forms."""
        ax = "x" if a == 1 else ("*", a, "x")
        if form == "ax+b":
            if b == 0:
                return ax
            return ("+", ax, b) if b > 0 else ("-", ax, -b)
        if form == "a(x+b)":  # requires b divisible handled by caller: a * (x + c) with b = a * c
            c = b // a
            inner = ("+", "x", c) if c > 0 else ("-", "x", -c)
            return inner if a == 1 else ("*", a, inner)
        raise ValueError(form)

    def sample(self, rng: random.Random) -> Problem:
        while True:
            root = rng.randint(-20, 20)
            kind = rng.choices(("ax+b=c", "c=ax+b", "ax+b=cx+d", "a(x+b)=c"), weights=(40, 15, 30, 15))[0]
            a = rng.randint(1, 9)
            if kind == "a(x+b)=c":
                c0 = rng.choice([v for v in range(-15, 16) if v != 0])
                b = a * c0
                left = self._side(rng, a, b, "a(x+b)")
                right = self._const(a * root + b)
            elif kind == "ax+b=cx+d":
                c = rng.choice([v for v in range(1, 10) if v != a])
                b = rng.randint(-30, 30)
                d = a * root + b - c * root
                left = self._side(rng, a, b, "ax+b")
                right = self._side(rng, c, d, "ax+b")
            else:
                b = rng.choice([v for v in range(-30, 31) if v != 0])
                left = self._side(rng, a, b, "ax+b")
                right = self._const(a * root + b)
                if kind == "c=ax+b":
                    left, right = right, left
            prompt = f"{HEAD}{show(left)} = {show(right)}{TAIL}"
            parsed = _parse(prompt)
            if parsed is None or parsed[0] != root:
                continue
            return Problem(self.name, prompt, f"{root} := by omega")

    def document(self, problem: Problem, rng: random.Random) -> str:
        return f"{problem.prompt} {solution_of(problem.prompt)} := {rng.choice(PROOFS)}"

    def split_key(self, prompt: str) -> str:
        """Canonical sides, in a fixed order: `20 = 3 * x + 5` is the same
        equation as `3 * x + 5 = 20`, and the generator writes both (the
        c=ax+b form is the ax+b=c form swapped), so with the sides in the
        order they were written 15% of held-out problems had their swapped
        twin in the training split."""
        parsed = _parse(prompt)
        if parsed is None:
            return prompt
        left, right = sorted((show(canonical(parsed[1])), show(canonical(parsed[2]))))
        return f"{HEAD}{left} = {right}{TAIL}"


DOMAIN = Linear()
