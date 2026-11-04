"""Problem sampling and the train/val/test split.

A problem's split is a function of its canonical form (`Domain.split_key`)
alone: sha256 of the key, 5% test, 5% val, 90% train. Two spellings of the same
problem (atoms renamed, an expression reparenthesised) share a key and so a
split, which keeps renamed copies of test problems out of training.

Problems are drawn per domain from a generator seeded by (seed, domain), so a
domain's problems do not depend on which other domains are requested.
"""

import hashlib
import random

import formal

TEST_PERCENT, VAL_PERCENT = 5, 5
SPLITS = ("train", "val", "test")


def split_key(prompt: str) -> str:
    return formal.domain_for(prompt).split_key(prompt)


def split_of_key(key: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % 100
    if bucket < TEST_PERCENT:
        return "test"
    if bucket < TEST_PERCENT + VAL_PERCENT:
        return "val"
    return "train"


def split_of(prompt: str) -> str:
    return split_of_key(split_key(prompt))


def _domain_rng(seed: int, domain: str) -> random.Random:
    return random.Random(f"{seed}/{domain}")


def sample_domain(domain: str, n: int, split: str, seed: int, unique: bool = False, max_tries: int = None) -> list:
    """n problems of one domain from `split`. With unique=True no two share a
    split key. Raises if the generator cannot supply them within max_tries
    draws (default 200 * n + 10_000)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    d = formal.DOMAINS[domain]
    rng = _domain_rng(seed, domain)
    out, seen = [], set()
    tries, limit = 0, (max_tries if max_tries is not None else 200 * n + 10_000)
    while len(out) < n:
        tries += 1
        if tries > limit:
            raise RuntimeError(f"{domain}: only {len(out)} of {n} {split} problems after {limit} draws")
        p = d.sample(rng)
        key = d.split_key(p.prompt)
        if split_of_key(key) != split or (unique and key in seen):
            continue
        seen.add(key)
        out.append(p)
    return out


def sample_problems(n: int, split: str, seed: int, domains="all", unique: bool = False) -> list:
    """n problems from `split`, spread evenly over the domains (the first
    n % len(domains) domains get one more), interleaved domain by domain."""
    names = formal.resolve(domains)
    per = [n // len(names) + (1 if i < n % len(names) else 0) for i in range(len(names))]
    lists = [sample_domain(name, k, split, seed, unique) for name, k in zip(names, per)]
    out = []
    for i in range(max(per, default=0)):
        out.extend(lst[i] for lst in lists if i < len(lst))
    return out


def pretrain_documents(n: int, seed: int, domains="all") -> list:
    """n whole declarations from the train split, for next-token pretraining."""
    rng = random.Random(f"{seed}/documents")
    return [formal.DOMAINS[p.domain].document(p, rng) for p in sample_problems(n, "train", seed, domains)]
