"""The Lean 4 domains and the functions every stage uses to judge answers.

    formal.verify(prompt, answer)   -> bool: does the declaration compile?
    formal.prefer(prompt, a, b)     -> +1 / -1 / 0: which answer a rater prefers
"""

from .base import Domain, Problem
from .fol import DOMAIN as FOL
from .linear import DOMAIN as LINEAR
from .nat import DOMAIN as NAT
from .prop import DOMAIN as PROP

DOMAINS = {d.name: d for d in (PROP, FOL, NAT, LINEAR)}  # web/checker.js uses the same order


def resolve(spec) -> list:
    """Domain names from "all", "prop,nat" or a list; raises on unknown names."""
    if isinstance(spec, str):
        names = list(DOMAINS) if spec.strip() == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    else:
        names = list(spec)
    unknown = [n for n in names if n not in DOMAINS]
    if unknown or not names:
        raise ValueError(f"unknown domains {unknown}; known: {list(DOMAINS)}")
    return names


def domain_for(prompt: str) -> Domain:
    """The domain that claims `prompt`; raises ValueError if none does."""
    for d in DOMAINS.values():
        if d.claims(prompt):
            return d
    raise ValueError(f"no domain claims this prompt: {prompt!r}")


def verify(prompt: str, answer: str) -> bool:
    for d in DOMAINS.values():
        if d.claims(prompt):
            return d.verify(prompt, answer)
    return False


def prefer(prompt: str, a: str, b: str) -> int:
    """+1 if a rater should prefer answer a, -1 for b, 0 for no preference. An
    answer that checks beats one that does not; between two that check, the
    shorter wins in domains where that is meaningful (proofs)."""
    d = domain_for(prompt)
    ok_a, ok_b = d.verify(prompt, a), d.verify(prompt, b)
    if ok_a != ok_b:
        return 1 if ok_a else -1
    if ok_a and d.prefer_shorter and len(a) != len(b):
        return 1 if len(a) < len(b) else -1
    return 0


__all__ = ["DOMAINS", "Domain", "Problem", "resolve", "domain_for", "verify", "prefer"]
